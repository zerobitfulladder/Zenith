"""Live viewer for the angle-only pole pupil (Dear PyGui).

Loads pole_angle/results/weights.npz (the trails arm) and runs the pupil
closed-loop at ~50Hz on screen. Buttons: Reset (new random recoverable
start), Run/Pause. Status line shows angle, force level, and
recovered/fallen state.

Run from a terminal WITH a display:
  .venv/bin/python experiments/2026_08_26/pole_angle/pole_viewer.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_angle"))

import numpy as np
import dearpygui.dearpygui as dpg

from run_pole_ema import (  # noqa: E402
    LEVELS,
    NB_A,
    NB_F,
    OUTPUT_DIR,
    RECOVER_HOLD,
    RECOVER_TDH,
    RECOVER_TH,
    TH_FAIL,
    THERMO_N,
    Trails,
    X_WIDE,
    physics,
    recoverable_init,
)

CW, CH = 760, 300                    # canvas size
TRACK_Y = 230
SCALE_X = 30                         # px per meter (track wraps visually)
POLE_PX = 90

wz = np.load(OUTPUT_DIR / "weights.npz")
W = wz["W"]
ALPHA_F = float(wz["alpha_f"])

rng = np.random.default_rng()
state = {"s": recoverable_init(rng), "running": False, "hold": 0,
         "status": "ready", "level": (NB_F + 1) // 2,
         "trails": Trails(ALPHA_F)}


def reset():
    state["s"] = recoverable_init(rng)
    state["trails"] = Trails(ALPHA_F)
    state["hold"] = 0
    state["status"] = "ready"
    state["level"] = (NB_F + 1) // 2


def hard_reset():
    """Worst-case corner: near-max lean + fast same-direction spin."""
    sign = 1 if rng.random() < 0.5 else -1
    state["s"] = np.array([0.0, rng.uniform(-0.5, 0.5),
                           sign * np.deg2rad(rng.uniform(8.5, 10.0)),
                           sign * rng.uniform(1.1, 1.4)])
    state["trails"] = Trails(ALPHA_F)
    state["hold"] = 0
    state["status"] = "HARD start"
    state["level"] = (NB_F + 1) // 2


def shove():
    state["s"][3] += (1 if rng.random() < 0.5 else -1) * rng.uniform(0.9, 1.4)
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
    nf = row[NB_A * 2 + NB_F:]              # bipolar: no relu
    level = int(np.argmax(THERMO_N @ (nf / (np.linalg.norm(nf) + 1e-9))))
    tr.did(level)
    state["level"] = level
    s = physics(s, LEVELS[level])
    state["s"] = s
    if abs(s[2]) > TH_FAIL or abs(s[0]) > X_WIDE:
        state["hold"] = 0
        state["status"] = "FALLEN — still driving (watch it)"
    elif abs(s[2]) < RECOVER_TH and abs(s[3]) < RECOVER_TDH:
        state["hold"] += 1
        if state["hold"] >= RECOVER_HOLD:
            state["status"] = "RECOVERED (still driving)"
    else:
        state["hold"] = 0
        if not state["status"].startswith("FELL"):
            state["status"] = "recovering..."


def redraw():
    dpg.delete_item("canvas", children_only=True)
    dpg.draw_line((0, TRACK_Y), (CW, TRACK_Y), color=(120, 120, 120),
                  thickness=2, parent="canvas")
    s = state["s"]
    cx = CW / 2 + (s[0] % (CW / SCALE_X / 2)) * SCALE_X * 0  # cart centered
    cx = CW / 2 + np.clip(s[0], -11, 11) / 11 * (CW / 2 - 60)
    dpg.draw_rectangle((cx - 25, TRACK_Y - 18), (cx + 25, TRACK_Y),
                       fill=(90, 140, 220), parent="canvas")
    tipx = cx + POLE_PX * np.sin(s[2])
    tipy = TRACK_Y - 18 - POLE_PX * np.cos(s[2])
    dpg.draw_line((cx, TRACK_Y - 18), (tipx, tipy), color=(240, 200, 80),
                  thickness=5, parent="canvas")
    dpg.draw_circle((tipx, tipy), 6, fill=(240, 120, 80), parent="canvas")
    u = LEVELS[state["level"]]
    dpg.draw_line((cx, TRACK_Y + 14), (cx + u * 60, TRACK_Y + 14),
                  color=(200, 90, 90), thickness=6, parent="canvas")
    dpg.set_value("status",
                  f"theta {np.rad2deg(s[2]):+6.2f} deg   "
                  f"force {u:+.2f}   x {s[0]:+5.2f} m   {state['status']}")


def main():
    dpg.create_context()
    with dpg.window(tag="main", label="Angle-only pole pupil"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset", callback=lambda: reset())
            dpg.add_button(label="Hard reset", callback=lambda: hard_reset())
            dpg.add_button(label="Shove", callback=lambda: shove())
            dpg.add_button(label="Run", tag="btn_run",
                           callback=lambda: toggle())
        dpg.add_text("", tag="status")
        with dpg.drawlist(width=CW, height=CH, tag="canvas"):
            pass
    dpg.create_viewport(title="Zenith pole pupil", width=CW + 40,
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
