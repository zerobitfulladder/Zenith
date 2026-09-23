"""The viewer. One of them, for every drone rig — present and future.

    .venv/bin/python viewer.py

Controllers: human (W/S thrust, A/D tilt), the two autopilots, and the
pupil once something has been trained. Click to set a target, "New
target" to move it without resetting the drone, Gust to kick it. The
fading trail is the easiest way to compare how two controllers approach.

WHERE THE PUPIL COMES FROM. Set VIEW_CKPT to a .npz written by any rig;
it defaults to the newest one in any results folder under experiments/. Beside it
there must be a .json saying how to bring it to life:

    {"kind": "single_layer", "size": 8192, "k": 4096, "enc_seed": 0}

`kind` is dispatched below. To add a rig, do NOT copy this file — give
your rig module a `load_policy(npz, cfg) -> fn(state, target) -> (l, r)`
and write {"kind": "module", "module": "my_rig", "dir": "experiments/..."}
into the json. The viewer will import it and call it.

Env: VIEW_CKPT = <path to .npz> | vision (default when the vision pupil exists) | latest | none
"""

import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SLD = HERE / "experiments" / "2026_08_28" / "single_layer_drone"
sys.path.insert(0, str(SLD))
sys.path.insert(0, str(HERE / "experiments" / "2026_08_29" / "temporal_drone"))

import dearpygui.dearpygui as dpg          # noqa: E402
import sl_drone as W                       # noqa: E402

try:
    import fast_oracle as F                # noqa: E402
    FAST = F.teacher
except Exception:                          # rig not present — hide the option
    FAST = None

# the drone's eye: the egocentric frame of 2026_09_03/vision_drone
sys.path.insert(0, str(HERE / "experiments" / "2026_09_03" / "vision_drone"))
try:
    import box_world as EYE                # noqa: E402
except Exception:
    EYE = None
EYE_PX = 192

CW, CH, SCALE, HEADER_PX, TRAIL_N = 900, 660, 42.0, 104, 240
VIEW = {"w": CW, "h": CH}


# ------------------------------------------------------------ the pupil ---
def _single_layer(npz, cfg):
    """The one-vector rig: sensory bits in, motor bits empty, winner's out."""
    Wt = npz["Wt"]
    enc = W.Encoder(int(cfg["size"]), seed=int(cfg.get("enc_seed", 0)))

    def act(s, tgt, lv_prev):
        q = enc.encode(W.sense(s, tgt, lv_prev), None)
        idx = np.nonzero(q)[0]
        return enc.read_motors(Wt[int(np.argmax(Wt[:, idx] @ q[idx]))])
    return act


def _from_module(npz, cfg):
    """A rig that brings its own loader — see the note at the top."""
    d = cfg.get("dir")
    if d:
        sys.path.insert(0, str(HERE / d))
    mod = __import__(cfg["module"])
    fn = mod.load_policy(npz, cfg)
    return lambda s, tgt, lv_prev: fn(s, tgt, lv_prev)


KINDS = {"single_layer": _single_layer, "module": _from_module}


def load_pupil():
    VISION = HERE / "experiments" / "2026_09_03" / "vision_drone" / "results" / "pupil.npz"
    pick = os.environ.get("VIEW_CKPT", "vision" if VISION.exists() else "latest")
    if pick == "none":
        return None, "pupil disabled"
    if pick == "vision":
        cands = [VISION.resolve()]
    elif pick != "latest":
        p = Path(pick)
        p = p if p.is_absolute() else (HERE / p)
        cands = [p.resolve()] if p.exists() else []
    else:
        cands = sorted((q for q in HERE.glob("experiments/**/*.npz")
                        if any(part.startswith("results") for part in q.parts)),
                       key=lambda q: q.stat().st_mtime, reverse=True)
    for npz_path in cands:
        cfg_path = npz_path.with_suffix(".json")
        if not cfg_path.exists():
            continue
        try:
            cfg = json.load(open(cfg_path))
            kind = cfg.get("kind", "single_layer")
            if kind not in KINDS:
                continue
            act = KINDS[kind](np.load(npz_path), cfg)
            try:
                rel = npz_path.resolve().relative_to(HERE)
            except ValueError:
                rel = npz_path
            return act, (f"{rel}  kind={kind} "
                         f"tick={cfg.get('tick')} score={cfg.get('score')}")
        except Exception as e:                       # keep the viewer usable
            return None, f"could not load {npz_path.name}: {e}"
    return None, "no pupil checkpoint found"


PUPIL, PUPIL_INFO = load_pupil()


def make_pid():
    """Full PID chaser, tuned by random search (2026-08-30): 1.00
    success on 100 fresh strict episodes, median 110 ticks to goal.
    Position PID (P, I with anti-windup, D on true velocity) ->
    desired tilt + collective -> attitude PD -> levels. Integrals
    reset when the target moves or the drone teleports."""
    G = {"KP": 4.811, "KI": 0.011, "KD": 4.084, "ILIM": 1.987,
         "KT": 0.071, "PHIMAX": 0.696, "KAp": 6.426, "KAd": 0.943}
    st = {"I": np.zeros(2), "prev": None}

    def pid(s, tgt):
        key = (float(tgt[0]), float(tgt[1]), float(s[0]), float(s[1]))
        if (st["prev"] is None or key[:2] != st["prev"][:2]
                or np.hypot(key[2] - st["prev"][2],
                            key[3] - st["prev"][3]) > 2.0):
            st["I"][:] = 0.0
        st["prev"] = key
        ex, ey = tgt[0] - s[0], tgt[1] - s[1]
        st["I"][0] = np.clip(st["I"][0] + ex * W.DT,
                             -G["ILIM"], G["ILIM"])
        st["I"][1] = np.clip(st["I"][1] + ey * W.DT,
                             -G["ILIM"], G["ILIM"])
        ax = G["KP"] * ex + G["KI"] * st["I"][0] - G["KD"] * s[2]
        ay = G["KP"] * ey + G["KI"] * st["I"][1] - G["KD"] * s[3]
        ph = W.wrap(s[4])
        ph_des = np.clip(-G["KT"] * ax, -G["PHIMAX"], G["PHIMAX"])
        coll = np.clip(0.5 * (9.8 + ay) / max(np.cos(ph), 0.5),
                       0.3, W.TMAX)
        diff = np.clip(G["KAp"] * (ph_des - ph) - G["KAd"] * s[5],
                       -2.5, 2.5)
        t1 = np.clip(coll - diff, 0, W.TMAX)
        t2 = np.clip(coll + diff, 0, W.TMAX)
        return (int(np.argmin(np.abs(W.LEVELS - t1))),
                int(np.argmin(np.abs(W.LEVELS - t2))))
    return pid


PID = make_pid()

rng = np.random.default_rng()
_s0, _t0 = W.any_init(rng)
state = {"s": _s0, "tgt": _t0, "running": False, "hold": 0, "ticks": 0,
         "lv": (W.NLEV // 2, W.NLEV // 2), "coll": W.HOVER, "diff": 0.0,
         "trail": deque(maxlen=TRAIL_N), "arrived": None,
         "status": "click to set a target, then Run"}


def world_to_px(x, y):
    return VIEW["w"] / 2 + x * SCALE, VIEW["h"] / 2 - y * SCALE


def px_to_world(px, py):
    return (px - VIEW["w"] / 2) / SCALE, (VIEW["h"] / 2 - py) / SCALE


def fit_canvas():
    w = max(320, dpg.get_viewport_client_width() - 16)
    h = max(240, dpg.get_viewport_client_height() - HEADER_PX)
    if (w, h) != (VIEW["w"], VIEW["h"]):
        VIEW["w"], VIEW["h"] = w, h
        dpg.configure_item("canvas", width=w, height=h)


def _restart(keep_drone):
    state["hold"] = state["ticks"] = 0
    state["arrived"] = None
    state["trail"].clear()
    if not keep_drone:
        state["s"], state["tgt"] = W.any_init(rng)
        state["lv"] = (W.NLEV // 2, W.NLEV // 2)
        state["coll"], state["diff"] = W.HOVER, 0.0


def reset():
    _restart(False)
    state["status"] = "reset"


def new_target():
    _restart(True)
    state["tgt"] = np.array([rng.uniform(-6, 6), rng.uniform(-4, 4)])
    state["status"] = "new target"


def gust():
    state["s"][2] += rng.uniform(-2.0, 2.0)
    state["s"][3] += rng.uniform(-2.0, 2.0)
    state["status"] = "GUST"


def toggle():
    state["running"] = not state["running"]
    dpg.set_item_label("btn_run", "Pause" if state["running"] else "Run")


def on_click():
    x, y = px_to_world(*dpg.get_drawing_mouse_pos())
    _restart(True)
    state["tgt"] = np.array([x, y])
    state["status"] = f"target ({x:+.1f}, {y:+.1f})"


def human_levels():
    if dpg.is_key_down(dpg.mvKey_W):
        state["coll"] = min(state["coll"] + 0.18, W.TMAX)
    if dpg.is_key_down(dpg.mvKey_S):
        state["coll"] = max(state["coll"] - 0.18, 0.0)
    if dpg.is_key_down(dpg.mvKey_A):
        state["diff"] = max(state["diff"] - 0.12, -2.5)
    elif dpg.is_key_down(dpg.mvKey_D):
        state["diff"] = min(state["diff"] + 0.12, 2.5)
    else:
        state["diff"] *= 0.9
    t1 = np.clip(state["coll"] - state["diff"], 0, W.TMAX)
    t2 = np.clip(state["coll"] + state["diff"], 0, W.TMAX)
    return (int(np.argmin(np.abs(W.LEVELS - t1))),
            int(np.argmin(np.abs(W.LEVELS - t2))))


def step():
    s, tgt, ctrl = state["s"], state["tgt"], dpg.get_value("ctrl")
    if ctrl == "pupil" and PUPIL is None:
        ctrl = "oracle-fast" if FAST else "oracle-PD"
        state["status"] = PUPIL_INFO
    if ctrl == "human":
        lv = human_levels()
    elif ctrl == "oracle-PD":
        lv = W.teacher(s, tgt)
    elif ctrl == "oracle-fast":
        lv = FAST(s, tgt)
    elif ctrl == "PID":
        lv = PID(s, tgt)
    else:
        lv = PUPIL(s, tgt, state["lv"])
        state["status"] = PUPIL_INFO
    state["lv"] = lv
    state["s"] = W.physics(s, lv)
    state["ticks"] += 1
    state["trail"].append((float(state["s"][0]), float(state["s"][1])))
    if W.at_goal(state["s"], tgt):
        state["hold"] += 1
        if state["hold"] >= 10 and state["arrived"] is None:
            state["arrived"] = state["ticks"]
    else:
        state["hold"] = 0


def redraw():
    dpg.delete_item("canvas", children_only=True)
    hx = int(VIEW["w"] / (2 * SCALE)) + 2
    hy = int(VIEW["h"] / (2 * SCALE)) + 2
    for gx in range(-hx - hx % 2, hx + 1, 2):
        dpg.draw_line(world_to_px(gx, -hy), world_to_px(gx, hy),
                      color=(45, 45, 55), parent="canvas")
    for gy in range(-hy - hy % 2, hy + 1, 2):
        dpg.draw_line(world_to_px(-hx, gy), world_to_px(hx, gy),
                      color=(45, 45, 55), parent="canvas")

    trail = list(state["trail"])
    for i in range(1, len(trail)):
        f = i / len(trail)
        dpg.draw_line(world_to_px(*trail[i - 1]), world_to_px(*trail[i]),
                      color=(70 + 150 * f, 90 + 80 * f, 200, 40 + 180 * f),
                      thickness=2, parent="canvas")

    tx, ty = world_to_px(*state["tgt"])
    dpg.draw_circle((tx, ty), 0.25 * SCALE, color=(90, 220, 120, 90),
                    parent="canvas")
    dpg.draw_circle((tx, ty), 9, color=(90, 220, 120), thickness=2,
                    parent="canvas")
    dpg.draw_line((tx - 13, ty), (tx + 13, ty), color=(90, 220, 120),
                  parent="canvas")
    dpg.draw_line((tx, ty - 13), (tx, ty + 13), color=(90, 220, 120),
                  parent="canvas")

    s = state["s"]
    cx, cy = world_to_px(s[0], s[1])
    if not (0 < cx < VIEW["w"] and 0 < cy < VIEW["h"]):
        dpg.draw_circle((float(np.clip(cx, 14, VIEW["w"] - 14)),
                         float(np.clip(cy, 14, VIEW["h"] - 14))), 9,
                        color=(230, 90, 90), thickness=3, parent="canvas")
    ph = W.wrap(s[4])
    dxa, dya = np.cos(ph), np.sin(ph)
    ax = W.ARM_L * SCALE * 2.2
    lx, ly = cx - ax * dxa, cy + ax * dya
    rx, ry = cx + ax * dxa, cy - ax * dya
    fx, fy = np.sin(ph), np.cos(ph)
    for (bx, by), lvl in (((lx, ly), state["lv"][0]),
                          ((rx, ry), state["lv"][1])):
        tt = W.LEVELS[lvl] / W.TMAX
        dpg.draw_line((bx, by), (bx + 34 * tt * fx, by + 34 * tt * fy),
                      color=(90, 130, 240), thickness=7, parent="canvas")
        dpg.draw_line((bx, by), (bx + 22 * tt * fx, by + 22 * tt * fy),
                      color=(190, 220, 255), thickness=3, parent="canvas")
    dpg.draw_line((lx, ly), (rx, ry), color=(230, 200, 90), thickness=6,
                  parent="canvas")
    dpg.draw_circle((cx, cy), 5, fill=(240, 130, 80), parent="canvas")

    if EYE is not None:
        fr = EYE.render(s, state["tgt"]).astype(np.float32)
        rgba = np.stack([fr, fr, fr, np.ones_like(fr)], -1).reshape(-1)
        dpg.set_value("eye_tex", rgba)
        x0, y0 = VIEW["w"] - EYE_PX - 12, 12
        dpg.draw_rectangle((x0 - 2, y0 - 2), (x0 + EYE_PX + 2, y0 + EYE_PX + 2),
                           color=(120, 120, 140), parent="canvas")
        dpg.draw_image("eye_tex", (x0, y0), (x0 + EYE_PX, y0 + EYE_PX), parent="canvas")
        dpg.draw_text((x0, y0 + EYE_PX + 4), "what the drone sees", color=(160, 160, 180),
                      size=13, parent="canvas")

    spd = float(np.hypot(s[2], s[3]))
    dist = float(np.hypot(s[0] - state["tgt"][0], s[1] - state["tgt"][1]))
    arr = (f"ARRIVED {state['arrived']} ticks ({state['arrived'] * W.DT:.2f}s)"
           if state["arrived"] else f"t={state['ticks']}")
    dpg.set_value("status",
                  f"dist {dist:5.2f}   speed {spd:4.2f}   tilt "
                  f"{np.rad2deg(ph):+6.1f}deg   thrust "
                  f"[{W.LEVELS[state['lv'][0]]:.1f},"
                  f"{W.LEVELS[state['lv'][1]]:.1f}]   {arr}   "
                  f"{state['status']}")


def main():
    opts = ["pupil", "PID", "oracle-fast", "oracle-PD", "human"]
    if FAST is None:
        opts.remove("oracle-fast")
    dpg.create_context()
    if EYE is not None:
        with dpg.texture_registry():
            dpg.add_raw_texture(EYE.SIZE, EYE.SIZE, np.zeros(EYE.SIZE * EYE.SIZE * 4, np.float32),
                                format=dpg.mvFormat_Float_rgba, tag="eye_tex")
    with dpg.window(tag="main", label="Drone"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset", callback=lambda: reset())
            dpg.add_button(label="New target", callback=lambda: new_target())
            dpg.add_button(label="Gust", callback=lambda: gust())
            dpg.add_button(label="Run", tag="btn_run", callback=lambda: toggle())
            dpg.add_combo(items=opts, default_value=opts[0], tag="ctrl",
                          width=130)
            dpg.add_text("W/S thrust, A/D tilt, click = target")
        dpg.add_text("", tag="status")
        dpg.add_text(f"pupil: {PUPIL_INFO}", color=(140, 140, 160))
        with dpg.drawlist(width=CW, height=CH, tag="canvas"):
            pass
    with dpg.item_handler_registry(tag="canvas_h"):
        dpg.add_item_clicked_handler(callback=lambda: on_click())
    dpg.bind_item_handler_registry("canvas", "canvas_h")
    dpg.create_viewport(title="Zenith drone viewer",
                        width=CW + 16, height=CH + HEADER_PX)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)
    while dpg.is_dearpygui_running():
        fit_canvas()
        if state["running"]:
            step()
        redraw()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
