"""Viewer for the single-layer sparse-OR drone (Dear PyGui).

Controllers: human (W/S thrust, A/D tilt), oracle, pupil (the single
hypercolumn: sensory bits in, motor bits empty, winner's motor bits
drive the thrusters). Click the canvas to set a target; Gust kicks it;
Reset drops it somewhere random. Free flight — no walls.

  .venv/bin/python experiments/2026_08_28/single_layer_drone/sl_viewer.py
Env: SL_CKPT = latest (default) | best | <path to .npz>
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dearpygui.dearpygui as dpg          # noqa: E402
import sl_drone as W                       # noqa: E402

RES = HERE / "results"
CW, CH = 900, 660
SCALE = 42.0
HEADER_PX = 78
VIEW = {"w": CW, "h": CH}

PUPIL, PUPIL_INFO = None, "no pupil weights yet"
_pick = os.environ.get("SL_CKPT", "latest")
_c = [(RES / f"{n}.npz", RES / f"{n}.json")
      for n in ("weights_best", "weights_final", "checkpoint")]
if _pick not in ("latest", "best"):
    _p = Path(_pick)
    _c = [(_p, _p.with_suffix(".json"))]
elif _pick == "best":
    _c = [x for x in _c if "weights_best" in x[0].name]
_c = [(a, b) for a, b in _c if a.exists() and b.exists()]
if _c:
    _npz, _json = max(_c, key=lambda x: x[0].stat().st_mtime)
    _w = np.load(_npz)
    _cfg = json.load(open(_json))
    PUPIL = {"W": _w["Wt"], "enc": W.Encoder(int(_cfg["size"]),
                                             seed=int(_cfg["enc_seed"]))}
    PUPIL_INFO = (f"{_npz.stem}: tick={_cfg.get('tick')} "
                  f"score={_cfg.get('score')} k={_cfg.get('k')}")

rng = np.random.default_rng()
_s0, _t0 = W.any_init(rng)
state = {"s": _s0, "tgt": _t0, "running": False, "hold": 0,
         "lv": (W.NLEV // 2, W.NLEV // 2), "coll": W.HOVER, "diff": 0.0,
         "status": "ready — click to set target"}


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


def reset():
    state["s"], state["tgt"] = W.any_init(rng)
    state["hold"] = 0
    state["lv"] = (W.NLEV // 2, W.NLEV // 2)
    state["coll"], state["diff"] = W.HOVER, 0.0
    state["status"] = "reset"


def gust():
    state["s"][2] += rng.uniform(-2.0, 2.0)
    state["s"][3] += rng.uniform(-2.0, 2.0)
    state["status"] = "GUST"


def toggle():
    state["running"] = not state["running"]
    dpg.set_item_label("btn_run", "Pause" if state["running"] else "Run")


def on_click():
    x, y = px_to_world(*dpg.get_drawing_mouse_pos())
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
    s, tgt = state["s"], state["tgt"]
    ctrl = dpg.get_value("ctrl")
    if ctrl == "pupil" and PUPIL is None:
        ctrl = "oracle"
        state["status"] = "no pupil weights — showing oracle"
    if ctrl == "human":
        lv = human_levels()
    elif ctrl == "oracle":
        lv = W.teacher(s, tgt)
    else:
        sens = W.sense(s, tgt, state["lv"])
        q = PUPIL["enc"].encode(sens, None)
        idx = np.nonzero(q)[0]
        raw = PUPIL["W"][:, idx] @ q[idx]
        lv = PUPIL["enc"].read_motors(PUPIL["W"][int(np.argmax(raw))])
        state["status"] = PUPIL_INFO
    state["lv"] = lv
    state["s"] = W.physics(s, lv)
    if W.at_goal(state["s"], tgt):
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
    dpg.set_value("status",
                  f"pos ({s[0]:+5.1f},{s[1]:+5.1f})  vel "
                  f"({s[2]:+4.1f},{s[3]:+4.1f})  tilt "
                  f"{np.rad2deg(ph):+6.1f}  thrust "
                  f"[{W.LEVELS[state['lv'][0]]:.1f},"
                  f"{W.LEVELS[state['lv'][1]]:.1f}]  {state['status']}")


def main():
    dpg.create_context()
    with dpg.window(tag="main", label="Single-layer drone"):
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
    dpg.create_viewport(title="Zenith single-layer drone",
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
