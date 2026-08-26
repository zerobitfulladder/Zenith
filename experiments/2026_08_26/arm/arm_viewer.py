"""Live viewer for the two-axis reaching arm (Dear PyGui).

CLICK anywhere on the canvas to place the target zone — the arm
reaches for it. Buttons: Reset arm (random pose), Run/Pause.

Run from a terminal WITH a display:
  .venv/bin/python experiments/2026_08_26/arm/arm_viewer.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "arm"))

import numpy as np
import dearpygui.dearpygui as dpg

from run_arm import (  # noqa: E402
    L1,
    L2,
    Motor,
    OUTPUT_DIR,
    STEP,
    TDIM,
    MDIM,
    WS,
    ZONE_R,
    build_z,
    decode_delta,
    rand_angles,
    target_code,
    tip,
    wrap,
)

CW = 560
SCALE = CW / (2 * WS)

W = np.load(OUTPUT_DIR / "weights.npz")["W"]
rng = np.random.default_rng()
state = {"th": rand_angles(rng), "tgt": np.array([1.2, 0.6]),
         "running": False, "motor": None, "status": "click to place a zone"}
state["motor"] = Motor(state["th"])


def to_px(p):
    return (CW / 2 + p[0] * SCALE, CW / 2 - p[1] * SCALE)


def to_ws(px, py):
    return np.array([(px - CW / 2) / SCALE, (CW / 2 - py) / SCALE])


def reset_arm():
    state["th"] = rand_angles(rng)
    state["motor"] = Motor(state["th"])
    state["status"] = "arm scrambled"


def toggle():
    state["running"] = not state["running"]
    dpg.set_item_label("btn_run", "Pause" if state["running"] else "Run")


def on_click():
    if dpg.is_item_hovered("canvas"):
        px, py = dpg.get_drawing_mouse_pos()
        state["tgt"] = to_ws(px, py)
        state["status"] = "new zone placed"


def step():
    m = state["motor"]
    q = build_z(target_code(state["tgt"]), m)
    row = W[int(np.argmax(W @ q))]
    lv = decode_delta(row[TDIM + MDIM:])
    state["th"] = wrap(state["th"] + lv * STEP)
    m.see(state["th"])
    d = np.linalg.norm(tip(state["th"]) - state["tgt"])
    state["status"] = ("IN ZONE" if d < ZONE_R
                       else f"reaching... dist {d:.2f}")


def redraw():
    dpg.delete_item("canvas", children_only=True)
    tx, ty = to_px(state["tgt"])
    dpg.draw_circle((tx, ty), ZONE_R * SCALE, color=(120, 220, 120),
                    thickness=2, parent="canvas")
    dpg.draw_circle((tx, ty), 3, fill=(120, 220, 120), parent="canvas")
    th = state["th"]
    base = to_px(np.zeros(2))
    elbow = to_px(np.array([L1 * np.cos(th[0]), L1 * np.sin(th[0])]))
    hand = to_px(tip(th))
    dpg.draw_line(base, elbow, color=(90, 140, 220), thickness=6,
                  parent="canvas")
    dpg.draw_line(elbow, hand, color=(240, 200, 80), thickness=5,
                  parent="canvas")
    dpg.draw_circle(base, 7, fill=(150, 150, 150), parent="canvas")
    dpg.draw_circle(elbow, 5, fill=(90, 140, 220), parent="canvas")
    dpg.draw_circle(hand, 6, fill=(240, 120, 80), parent="canvas")
    dpg.set_value("status", state["status"])


def main():
    dpg.create_context()
    with dpg.window(tag="main", label="Two-axis reaching arm"):
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset arm", callback=lambda: reset_arm())
            dpg.add_button(label="Run", tag="btn_run",
                           callback=lambda: toggle())
        dpg.add_text("", tag="status")
        with dpg.drawlist(width=CW, height=CW, tag="canvas"):
            pass
    with dpg.handler_registry():
        dpg.add_mouse_click_handler(callback=lambda s, a: on_click())
    dpg.create_viewport(title="Zenith arm", width=CW + 40,
                        height=CW + 130)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)
    while dpg.is_dearpygui_running():
        if state["running"]:
            step()
        redraw()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
