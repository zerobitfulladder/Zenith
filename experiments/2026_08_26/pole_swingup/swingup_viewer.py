"""Live viewer for the ANY-state swing-up pupil (Dear PyGui).

No fall concept — there is only "not upright yet." Buttons: Reset
(uniform-random state, any angle, any spin), Shove, Run/Pause.

Run from a terminal WITH a display:
  .venv/bin/python experiments/2026_08_26/pole_swingup/swingup_viewer.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_angle"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_swingup"))

import numpy as np
import dearpygui.dearpygui as dpg

from run_pole_ema import LEVELS, NB_F, THERMO_N, physics  # noqa: E402
from run_pole_swingup import (  # noqa: E402
    NB_A,
    OUTPUT_DIR,
    RECOVER_HOLD,
    RECOVER_TDH,
    RECOVER_TH,
    Trails,
    any_init,
    wrap,
)

CW, CH = 760, 340
TRACK_Y = 200
POLE_PX = 90

W = np.load(OUTPUT_DIR / "weights.npz")["W"]
rng = np.random.default_rng()
state = {"s": any_init(rng), "running": False, "hold": 0,
         "status": "ready", "level": (NB_F + 1) // 2, "trails": Trails()}


def reset():
    state["s"] = any_init(rng)
    state["trails"] = Trails()
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
    tr = state["trails"]
    tr.see(s[2])
    q = tr.joint()
    row = W[int(np.argmax(W @ q))]
    nf = row[NB_A * 3 + NB_F:]
    level = int(np.argmax(THERMO_N @ (nf / (np.linalg.norm(nf) + 1e-9))))
    tr.did(level)
    state["level"] = level
    s = physics(s, LEVELS[level])
    state["s"] = s
    if abs(wrap(s[2])) < RECOVER_TH and abs(s[3]) < RECOVER_TDH:
        state["hold"] += 1
        if state["hold"] >= RECOVER_HOLD:
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
                  f"theta {np.rad2deg(th):+7.1f} deg   "
                  f"spin {s[3]:+5.2f}   force {u:+.2f}   "
                  f"x {s[0]:+6.2f} m   {state['status']}")


def main():
    dpg.create_context()
    with dpg.window(tag="main", label="Any-state swing-up pupil"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset (anywhere)", callback=lambda: reset())
            dpg.add_button(label="Shove", callback=lambda: shove())
            dpg.add_button(label="Run", tag="btn_run",
                           callback=lambda: toggle())
        dpg.add_text("", tag="status")
        with dpg.drawlist(width=CW, height=CH, tag="canvas"):
            pass
    dpg.create_viewport(title="Zenith swing-up pupil", width=CW + 40,
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
