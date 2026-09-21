"""Live sampler for the torus flow.  Pick a class, turn the noise up, watch it denoise.

    theta = chord_y + N(0, sd)  on all 784 angles   ->  f^-1  ->  image

The digit is the STATIC part and the speckle is what varies, so averaging samples is the
operation that recovers it.  Classic background subtraction -- remove what stays still --
would delete the signal, which is why that option is labelled as the control it is.

Two subtractions that do make sense here:
    across-class   mean over all ten chords at the same sd.  Whatever is shared by every
                   class is not class information; what is left is specific to this one.
    noise floor    the same noise around a point that is NOT a chord.  What sd alone looks
                   like, with no digit under it.

    uv run python experiments/2026_09_21/nice_flow/app.py
"""
import time
from pathlib import Path
import numpy as np
import torch
import dearpygui.dearpygui as dpg
import torusflow as TF

ROOT = Path(__file__).resolve().parent
TF.SPAN = np.pi                                   # the checkpoint was trained at span = pi
D, NC, S = 784, 10, 10                            # dims, classes, display upscale
dev = TF.dev

flow = TF.TorusFlow(D, 6, 1024).to(dev)
base = TF.Base(D, D, learn_kappa=True).to(dev)
ck = torch.load(ROOT / "results" / "torusflow_full.pt", map_location=dev)
flow.load_state_dict(ck["flow"]); base.load_state_dict(ck["base"]); flow.eval()
CHORDS = base.chords
torch.manual_seed(0)
OFF = torch.rand(1, D, device=dev) * TF.TAU       # a fixed non-chord point, for the noise floor

with torch.no_grad():
    PROTO = TF.to_x(flow.inverse(CHORDS)).cpu().numpy().reshape(NC, 28, 28)

ema = None


def decode(th):
    with torch.no_grad():
        return TF.to_x(flow.inverse(th % TF.TAU))


def frame(label, sd, n, reduce_median, bg, alpha, normalize):
    """one displayed frame: (single raw sample, processed)"""
    global ema
    need_all = bg == "across-class"
    centres = CHORDS if need_all else CHORDS[label:label+1]
    th = centres.repeat_interleave(n, 0) + torch.randn(len(centres) * n, D, device=dev) * sd
    img = decode(th).reshape(len(centres), n, D)

    own = img[label] if need_all else img[0]
    raw = own[0].cpu().numpy()
    red = own.median(0).values if reduce_median else own.mean(0)

    if bg == "across-class":
        red = red - img.mean(1).mean(0)
    elif bg == "noise floor":
        nf = decode(OFF.repeat(n, 1) + torch.randn(n, D, device=dev) * sd)
        red = red - (nf.median(0).values if reduce_median else nf.mean(0))
    elif bg == "temporal EMA (control)":
        cur = red.cpu().numpy()
        ema = cur if ema is None else alpha * cur + (1 - alpha) * ema
        red = torch.tensor(cur - ema, device=dev)

    out = red.cpu().numpy()
    if normalize:
        out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return np.clip(raw, 0, 1), np.clip(out, 0, 1)


def tex(v):
    """28x784 vector -> flat RGBA float32 at S times scale, for a raw texture."""
    a = np.repeat(np.repeat(v.reshape(28, 28), S, 0), S, 1).astype(np.float32)
    return np.dstack([a, a, a, np.ones_like(a)]).ravel()


def main():
    dpg.create_context()
    W = 28 * S
    blank = tex(np.zeros(784))
    with dpg.texture_registry():
        for t in ("raw", "proc", "proto"):
            dpg.add_raw_texture(W, W, blank, format=dpg.mvFormat_Float_rgba, tag=f"tex_{t}")

    with dpg.window(tag="main"):
        with dpg.group(horizontal=True):
            for t, lab in (("raw", "one sample"), ("proc", "processed"), ("proto", "prototype (sd=0)")):
                with dpg.group():
                    dpg.add_text(lab)
                    dpg.add_image(f"tex_{t}")
        dpg.add_separator()
        dpg.add_radio_button([str(i) for i in range(NC)], tag="label", default_value="0", horizontal=True)
        dpg.add_slider_float(label="sd", tag="sd", default_value=0.15, min_value=0.0, max_value=1.5)
        dpg.add_slider_int(label="samples averaged per frame", tag="n",
                           default_value=1, min_value=1, max_value=128)
        dpg.add_checkbox(label="median instead of mean", tag="median")
        dpg.add_checkbox(label="normalise contrast", tag="norm", default_value=True)
        dpg.add_combo(["none", "across-class", "noise floor", "temporal EMA (control)"],
                      label="background subtraction", tag="bg", default_value="none")
        dpg.add_slider_float(label="EMA alpha", tag="alpha", default_value=0.05,
                             min_value=0.005, max_value=0.5)
        dpg.add_separator()
        dpg.add_text("", tag="fps")
        dpg.add_text("the digit is static, the speckle is not -- averaging is what recovers it",
                     color=(150, 150, 150))

    dpg.create_viewport(title="torus flow - live sampler", width=1000, height=620)
    dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window("main", True)

    TARGET = 1 / 30
    last, smooth = time.perf_counter(), 0.0
    global ema
    while dpg.is_dearpygui_running():
        t0 = time.perf_counter()
        label = int(dpg.get_value("label"))
        raw, proc = frame(label, dpg.get_value("sd"), dpg.get_value("n"),
                          dpg.get_value("median"), dpg.get_value("bg"),
                          dpg.get_value("alpha"), dpg.get_value("norm"))
        dpg.set_value("tex_raw", tex(raw))
        dpg.set_value("tex_proc", tex(proc))
        dpg.set_value("tex_proto", tex(PROTO[label].ravel()))

        compute = time.perf_counter() - t0
        now = time.perf_counter()
        smooth = 0.9 * smooth + 0.1 * (now - last); last = now
        dpg.set_value("fps", f"{1/max(smooth,1e-6):5.1f} fps     compute {compute*1000:5.1f} ms "
                             f"of a {TARGET*1000:.0f} ms budget     {dev}")
        dpg.render_dearpygui_frame()
        slack = TARGET - (time.perf_counter() - t0)
        if slack > 0: time.sleep(slack)

    dpg.destroy_context()


if __name__ == "__main__":
    main()
