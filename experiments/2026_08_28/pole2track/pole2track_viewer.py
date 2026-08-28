"""Live viewer for the two-track swing-up pupil (Dear PyGui).

Loads the best arm from results/ (weights_best.npz +
config_best.json) and runs it closed-loop: angle in, sensory track up,
top retrieves, motor half reprojects, force out. Buttons: Reset
(uniform-random state, any angle/spin), Shove, Run/Pause.

Run from a terminal WITH a display:
  .venv/bin/python experiments/2026_08_28/pole2track/pole2track_viewer.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_angle"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_swingup"))

import dearpygui.dearpygui as dpg                        # noqa: E402

OUT = HERE / "results"
import os                                                 # noqa: E402
_cfg_early = json.load(open(OUT / "config_best.json"))
os.environ["P2_LAGS"] = ",".join(str(v) for v in _cfg_early["lags"])

from run_pole_ema import LEVELS, NB_F, THERMO_N, physics  # noqa: E402
from run_pole_swingup import any_init, cn, wrap           # noqa: E402
from run_pole2track import (                              # noqa: E402
    AngleHist, K_M, K_S, SDIM)
CW, CH = 760, 340
TRACK_Y = 200
POLE_PX = 90

cfg = json.load(open(OUT / "config_best.json"))
w = np.load(OUT / "weights_best.npz")
Wtop, Ws, Wm = w["Wtop"], w["Ws"], w["Wm"]
ARM, READ = cfg["arm"], cfg["read"]
SD = K_S if ARM == "full" else SDIM
MD = K_M if ARM != "flat" else NB_F

import joblib                                             # noqa: E402
from run_pole2track import make_teacher_from_cfg, make_lqr_expert  # noqa: E402
MLP = joblib.load(OUT / "mlp.joblib") if (OUT / "mlp.joblib").exists() else None
if (OUT / "expert.json").exists():
    ORACLE = make_teacher_from_cfg(json.load(open(OUT / "expert.json")))
else:
    ORACLE = make_lqr_expert(-1.0, 1.0)

rng = np.random.default_rng()
state = {"s": any_init(rng), "running": False, "hold": 0,
         "hist": AngleHist(),
         "status": f"ready — arm={ARM} read={READ} "
                   f"(round {cfg['round']}, {cfg['score']}/100)",
         "level": (NB_F + 1) // 2}


TALLY = w["tally"] if "tally" in w.files else None


def act(s_hat):
    sh = cn(np.maximum(Ws @ s_hat, 0.0)) if ARM == "full" else s_hat
    mh = TALLY[int(np.argmax(Wtop @ sh))]
    if ARM == "flat":
        dirn = mh / (np.linalg.norm(mh) + 1e-9)
    elif READ == "hard":
        u = int(np.argmax(mh))
        dirn = Wm[u] / (np.linalg.norm(Wm[u]) + 1e-9)
    else:
        d = np.maximum(mh, 0.0) @ Wm
        dirn = d / (np.linalg.norm(d) + 1e-9)
    return int(np.argmax(THERMO_N @ dirn))


def reset():
    state["s"] = any_init(rng)
    state["hist"] = AngleHist()
    state["hold"] = 0
    state["status"] = "ready (anywhere)"
    state["level"] = (NB_F + 1) // 2


def shove():
    state["s"][3] += (1 if rng.random() < 0.5 else -1) * rng.uniform(1.5, 3.0)
    state["hold"] = 0
    state["status"] = "SHOVED"


def toggle():
    state["running"] = not state["running"]
    dpg.set_item_label("btn_run", "Pause" if state["running"] else "Run")


def pupil_step():
    s = state["s"]
    ctrl = dpg.get_value("ctrl") if dpg.does_item_exist("ctrl") else "pupil (bank)"
    if ctrl == "pupil (bank)" and TALLY is None:
        ctrl = "oracle"
        state["status"] = "pupil weights regenerating — showing oracle"
    state["hist"].see(s[2])
    if ctrl == "oracle":
        level = ORACLE(s)
    elif ctrl == "mlp" and MLP is not None:
        level = int(MLP.predict(state["hist"].code()[None].astype(
            np.float32))[0])
    else:
        level = act(state["hist"].code())
    state["hist"].did(level)
    state["level"] = level
    s = physics(s, LEVELS[level])
    state["s"] = s
    if abs(wrap(s[2])) < np.deg2rad(1.5) and abs(s[3]) < 0.25:
        state["hold"] += 1
        if state["hold"] >= 10:
            state["status"] = "UPRIGHT (holding)"
    else:
        state["hold"] = 0
        state["status"] = "working on it..."


def redraw():
    dpg.delete_item("canvas", children_only=True)
    dpg.draw_line((0, TRACK_Y), (CW, TRACK_Y), color=(120, 120, 120),
                  thickness=2, parent="canvas")
    s = state["s"]
    cx = CW / 2 + np.clip(s[0], -31, 31) / 31 * (CW / 2 - 60)
    dpg.draw_rectangle((cx - 25, TRACK_Y - 18), (cx + 25, TRACK_Y),
                       fill=(90, 140, 220), parent="canvas")
    th = wrap(s[2])
    tipx = cx + POLE_PX * np.sin(th)
    tipy = TRACK_Y - 18 - POLE_PX * np.cos(th)
    dpg.draw_line((cx, TRACK_Y - 18), (tipx, tipy), color=(240, 200, 80),
                  thickness=5, parent="canvas")
    dpg.draw_circle((tipx, tipy), 6, fill=(240, 120, 80), parent="canvas")
    u = LEVELS[state["level"]]
    dpg.draw_line((cx, TRACK_Y + 14), (cx + u * 60, TRACK_Y + 14),
                  color=(200, 90, 90), thickness=6, parent="canvas")
    dpg.set_value("status",
                  f"theta {np.rad2deg(th):+7.1f} deg   spin {s[3]:+5.2f}   "
                  f"force {u:+.2f}   x {s[0]:+6.2f} m   {state['status']}")


def main():
    dpg.create_context()
    with dpg.window(tag="main", label="Two-track swing-up pupil"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset (anywhere)", callback=lambda: reset())
            dpg.add_button(label="Shove", callback=lambda: shove())
            dpg.add_button(label="Run", tag="btn_run", callback=lambda: toggle())
            dpg.add_combo(items=["pupil (bank)", "mlp", "oracle"],
                          default_value="pupil (bank)", tag="ctrl", width=140)
        dpg.add_text("", tag="status")
        with dpg.drawlist(width=CW, height=CH, tag="canvas"):
            pass
    dpg.create_viewport(title="Zenith two-track pupil", width=CW + 40,
                        height=CH + 130)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)
    while dpg.is_dearpygui_running():
        if state["running"]:
            pupil_step()
        redraw()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
