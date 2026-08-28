"""Planar drone viewer (Dear PyGui) — fly it yourself, or watch.

Controllers (dropdown):
  human   W/S = collective thrust up/down, A/D = tilt left/right
          (differential thrust). You fly it.
  oracle  the LQR teacher.
  pupil   the trained three-track memory (needs results/unified weights;
          falls back to oracle with a status note until they exist).

Click anywhere on the canvas to set the target. Gust = random velocity
kick. Reset = random state + random target.

Run from a terminal WITH a display:
  .venv/bin/python experiments/2026_08_28/drone/drone_viewer.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dearpygui.dearpygui as dpg          # noqa: E402
import run_drone as D                      # noqa: E402

OUT = HERE / "results" / "unified"
CW, CH = 900, 660                           # initial canvas size
SCALE = 42.0                                # px per meter
HEADER_PX = 78                              # buttons + status line
VIEW = {"w": CW, "h": CH}                   # live canvas size

CAS = HERE / "results" / "cascade"
GROW = HERE / "results" / "cascade_grow"
BOUND = HERE / "results" / "cascade_bound"
PUPIL = None
PUPIL_INFO = "no pupil weights yet"
_cands = [(OUT / "weights_best.npz", OUT / "weights_best.json"),
          (OUT / "checkpoint.npz", OUT / "checkpoint.json"),
          (CAS / "weights_best.npz", CAS / "weights_best.json"),
          (CAS / "checkpoint.npz", CAS / "checkpoint.json"),
          (GROW / "weights_best.npz", GROW / "weights_best.json"),
          (GROW / "checkpoint.npz", GROW / "checkpoint.json"),
          (BOUND / "weights_best.npz", BOUND / "weights_best.json"),
          (BOUND / "checkpoint.npz", BOUND / "checkpoint.json"),
          (HERE / "results" / "cascade_bound_t1" / "weights_best.npz",
           HERE / "results" / "cascade_bound_t1" / "weights_best.json"),
          (HERE / "results" / "cascade_bound_t1" / "checkpoint.npz",
           HERE / "results" / "cascade_bound_t1" / "checkpoint.json"),
          (HERE.parent / "drone_continual" / "results" / "weights_best.npz",
           HERE.parent / "drone_continual" / "results" / "weights_best.json"),
          (HERE.parent / "drone_continual" / "results" / "checkpoint.npz",
           HERE.parent / "drone_continual" / "results" / "checkpoint.json"),
          (HERE / "results" / "cascade_sliced" / "weights_best.npz",
           HERE / "results" / "cascade_sliced" / "weights_best.json"),
          (HERE / "results" / "cascade_sliced" / "checkpoint.npz",
           HERE / "results" / "cascade_sliced" / "checkpoint.json")]
_cands += [(HERE.parent / "drone_continual" / "results" / "weights_final.npz",
            HERE.parent / "drone_continual" / "results" / "weights_final.json")]
# DV_CKPT: "latest" (default, newest file on disk — the living creature),
# "best" (peak weights only), or an explicit .npz path.
_pick = os.environ.get("DV_CKPT", "latest")
if _pick not in ("latest", "best"):
    _p = Path(_pick)
    _cands = [(_p, _p.with_suffix(".json"))]
elif _pick == "best":
    _cands = [c for c in _cands if "weights_best" in c[0].name]
_cands = [(n, j) for n, j in _cands if n.exists() and j.exists()]
if _cands:
    _npz, _json = max(_cands, key=lambda c: c[0].stat().st_mtime)
    w = np.load(_npz)
    cfg = json.load(open(_json))
    if cfg.get("cascade"):
        import run_drone_cascade as C

        class _B2:
            def __init__(self, W):
                self.W = W

            def code(self, x):
                return np.maximum(self.W @ x, 0.0)

        class _T:
            def __init__(self, W):
                self.W = W

            def forward(self, x):
                return self.W @ x

        _assoc = cfg.get("assoc", "tally")
        _m = {"assoc": _assoc,
              "bo": None if w["Wbo"].shape == (1, 1) else _B2(w["Wbo"]),
              "bi": None if w["Wbi"].shape == (1, 1) else _B2(w["Wbi"]),
              "cb": {"p": _B2(w["Wcp"]), "c": _B2(w["Wcc"]),
                     "d": _B2(w["Wcd"])},
              "t_out": _T(w["Wto"]), "t_in": _T(w["Wti"]),
              "ty_out": w["tyo"], "ty_in": w["tyi"],
              "kod": w["Wto"].shape[1] - 2 * 16,
              "kid": w["Wti"].shape[1] - 16}
        _m["okey"] = ((lambda x: x) if _m["bo"] is None
                      else (lambda x: D.cn(_m["bo"].code(x))))
        _m["ikey"] = ((lambda x: x) if _m["bi"] is None
                      else (lambda x: D.cn(_m["bi"].code(x))))
        PUPIL = {"cascade": True, "m": _m}
        PUPIL_INFO = (f"{_npz.parent.name}/{_npz.stem}: CASCADE "
                      f"round={cfg.get('round')} score={cfg.get('score')}")
        arm = "cascade"
    else:
        arm = cfg["arm"]
        PUPIL_INFO = (f"{_npz.stem}: arm={arm} round={cfg.get('round')} "
                      f"score={cfg.get('score')}")

if _cands and PUPIL is None and "Wp" in w.files:
    class _B:
        def __init__(self, W):
            self.W = W

        def code(self, x):
            return np.maximum(self.W @ x, 0.0)

    bp, bt = _B(w["Wp"]), _B(w["Wt"])
    flat = w["Wm"]
    if arm == "joint":
        bm = [_B(flat.reshape(D.K_MJ, 2 * D.NB_F))]
    else:
        h = D.K_MF * D.NB_F
        bm = [_B(flat[:h].reshape(D.K_MF, D.NB_F)),
              _B(flat[h:].reshape(D.K_MF, D.NB_F))]

    def _skey(p, tg):
        return D.cn(np.concatenate([D.cn(bp.code(p)), D.cn(bt.code(tg))]))

    PUPIL = {"arm": arm, "bp": bp, "bt": bt, "bm": bm,
             "top": type("T", (), {"forward": lambda self, x: w["Wtop"] @ x,
                                   "W": w["Wtop"]})(),
             "tally": w["tally"], "skey": _skey}

rng = np.random.default_rng()
_s0, _t0 = D.any_init(rng)
state = {"s": _s0, "tgt": _t0, "running": False, "hold": 0,
         "hist": D.Hist(), "lv": (4, 4), "coll": D.HOVER, "diff": 0.0,
         "status": "ready — click canvas to set target"}


def world_to_px(x, y):
    return VIEW["w"] / 2 + x * SCALE, VIEW["h"] / 2 - y * SCALE


def px_to_world(px, py):
    return ((px - VIEW["w"] / 2) / SCALE,
            (VIEW["h"] / 2 - py) / SCALE)


def fit_canvas():
    """Canvas fills everything below the buttons/status row."""
    w = max(320, dpg.get_viewport_client_width() - 16)
    h = max(240, dpg.get_viewport_client_height() - HEADER_PX)
    if (w, h) != (VIEW["w"], VIEW["h"]):
        VIEW["w"], VIEW["h"] = w, h
        dpg.configure_item("canvas", width=w, height=h)


def reset():
    state["s"], state["tgt"] = D.any_init(rng)
    state["hist"] = D.Hist()
    state["chist"] = None
    state["hold"] = 0
    state["coll"], state["diff"] = D.HOVER, 0.0
    state["status"] = "reset"


def gust():
    state["s"][2] += rng.uniform(-2.0, 2.0)
    state["s"][3] += rng.uniform(-2.0, 2.0)
    state["status"] = "GUST"


def toggle():
    state["running"] = not state["running"]
    dpg.set_item_label("btn_run", "Pause" if state["running"] else "Run")


def on_click():
    px, py = dpg.get_drawing_mouse_pos()
    x, yw = px_to_world(px, py)
    state["tgt"] = np.array([np.clip(x, -8, 8), np.clip(yw, -6, 6)])
    state["status"] = f"target ({state['tgt'][0]:+.1f}, {state['tgt'][1]:+.1f})"


def human_levels():
    if dpg.is_key_down(dpg.mvKey_W):
        state["coll"] = min(state["coll"] + 0.18, D.TMAX)
    if dpg.is_key_down(dpg.mvKey_S):
        state["coll"] = max(state["coll"] - 0.18, 0.0)
    if dpg.is_key_down(dpg.mvKey_A):
        state["diff"] = max(state["diff"] - 0.12, -2.5)
    elif dpg.is_key_down(dpg.mvKey_D):
        state["diff"] = min(state["diff"] + 0.12, 2.5)
    else:
        state["diff"] *= 0.9
    t1 = np.clip(state["coll"] - state["diff"], 0, D.TMAX)
    t2 = np.clip(state["coll"] + state["diff"], 0, D.TMAX)
    return (int(np.argmin(np.abs(D.LEVELS - t1))),
            int(np.argmin(np.abs(D.LEVELS - t2))))


def step():
    s, tgt = state["s"], state["tgt"]
    ctrl = dpg.get_value("ctrl")
    if ctrl == "pupil" and PUPIL is None:
        ctrl = "oracle"
        state["status"] = "no pupil weights yet — showing oracle"
    elif ctrl == "pupil":
        state["status"] = PUPIL_INFO
    if ctrl == "human":
        lv = human_levels()
    elif ctrl == "oracle":
        lv = D.teacher(s, tgt)
    elif isinstance(PUPIL, dict) and PUPIL.get("cascade"):
        import run_drone_cascade as C
        if state.get("chist") is None:
            state["chist"] = C.CHist()
        h = state["chist"]
        h.see(s, tgt)
        lp, lc, ld = C.cascade_act(PUPIL["m"], h)
        h.did(ld)
        lv = C.levels_from(lp, lc, ld)
    else:
        state["hist"].see(s, tgt)
        lv = D.act(PUPIL, state["hist"].proprio(), state["hist"].target())
        state["hist"].did(lv)
    state["lv"] = lv
    s = D.physics(s, lv)                 # free flight — no walls
    state["s"] = s
    if D.at_goal(s, tgt):
        state["hold"] += 1
        if state["hold"] >= 10:
            state["status"] = "AT TARGET (holding)"
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
    tx, ty = world_to_px(*state["tgt"])
    dpg.draw_circle((tx, ty), 9, color=(90, 220, 120), thickness=2,
                    parent="canvas")
    dpg.draw_line((tx - 13, ty), (tx + 13, ty), color=(90, 220, 120),
                  parent="canvas")
    dpg.draw_line((tx, ty - 13), (tx, ty + 13), color=(90, 220, 120),
                  parent="canvas")
    s = state["s"]
    cx, cy = world_to_px(s[0], s[1])
    ph = D.wrap(s[4])
    dxa, dya = np.cos(ph), np.sin(ph)
    if not (0 < cx < VIEW["w"] and 0 < cy < VIEW["h"]):
        # flown out of view: pin a marker to the edge it left by
        ex = float(np.clip(cx, 14, VIEW["w"] - 14))
        ey = float(np.clip(cy, 14, VIEW["h"] - 14))
        dpg.draw_circle((ex, ey), 9, color=(230, 90, 90), thickness=3,
                        parent="canvas")
    ax = D.ARM_L * SCALE * 2.2
    lx, ly = cx - ax * dxa, cy + ax * dya
    rx, ry = cx + ax * dxa, cy - ax * dya
    t1 = D.LEVELS[state["lv"][0]] / D.TMAX
    t2 = D.LEVELS[state["lv"][1]] / D.TMAX
    fx, fy = np.sin(ph), np.cos(ph)
    for (bx, by), tt in (((lx, ly), t1), ((rx, ry), t2)):
        dpg.draw_line((bx, by), (bx + 34 * tt * fx, by + 34 * tt * fy),
                      color=(90, 130, 240), thickness=7, parent="canvas")
        dpg.draw_line((bx, by), (bx + 22 * tt * fx, by + 22 * tt * fy),
                      color=(190, 220, 255), thickness=3, parent="canvas")
    dpg.draw_line((lx, ly), (rx, ry), color=(230, 200, 90), thickness=6,
                  parent="canvas")
    dpg.draw_circle((cx, cy), 5, fill=(240, 130, 80), parent="canvas")
    dpg.set_value(
        "status",
        f"pos ({s[0]:+5.1f},{s[1]:+5.1f})  vel ({s[2]:+4.1f},{s[3]:+4.1f})  "
        f"tilt {np.rad2deg(ph):+6.1f}  thrust "
        f"[{D.LEVELS[state['lv'][0]]:.1f}, {D.LEVELS[state['lv'][1]]:.1f}]  "
        f"{state['status']}")


def main():
    dpg.create_context()
    with dpg.window(tag="main", label="Planar drone"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset", callback=lambda: reset())
            dpg.add_button(label="Gust", callback=lambda: gust())
            dpg.add_button(label="Run", tag="btn_run",
                           callback=lambda: toggle())
            dpg.add_combo(items=["human", "oracle", "pupil"],
                          default_value="human", tag="ctrl", width=110)
            dpg.add_text("W/S thrust, A/D tilt, click = target")
        dpg.add_text("", tag="status")
        with dpg.drawlist(width=CW, height=CH, tag="canvas"):
            pass
    with dpg.item_handler_registry(tag="canvas_h"):
        dpg.add_item_clicked_handler(callback=lambda: on_click())
    dpg.bind_item_handler_registry("canvas", "canvas_h")
    dpg.create_viewport(title="Zenith drone", width=CW + 16,
                        height=CH + HEADER_PX)
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
