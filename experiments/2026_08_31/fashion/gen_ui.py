"""Generation explorer for the Fashion experts.  NOT a fork of viewer.py --
that one flies drones; this one has no simulation in it at all.

    .venv/bin/python experiments/2026_08_31/fashion/gen_ui.py

Pick a class, pick one of the experts that claims it, and draw from it:

    mean          its prototype -- the deterministic centre of the class
    diagonal      each coefficient rattled independently (grainy)
    covariance    coefficients moved together the way they really co-vary

Temperature scales how far from the mean you go: 0 is the prototype, 1 is the
spread the real data has, above 1 leaves the class. The sliders are the twelve
coefficients that carry the most variance for this expert -- drag one and watch
which part of the garment it owns. Beside the drawing you get the nearest real
training image, so realism is judged against something, not asserted.

Run with --check to exercise everything headless.
"""

import sys
from pathlib import Path
import numpy as np
from common import load, center_norm, EPS, CLASSES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
N_IMG, NSLIDER, SCALE = 784, 12, 9
RNG = np.random.default_rng()

z = np.load(OUT / "coef_stats.npz")
W = z["W"].astype(np.float64)
claim, wins = z["claim"], z["wins"]
mu, cov, PROTO = (z["mu"].astype(np.float64), z["cov"].astype(np.float64),
                  z["proto"].astype(np.float64))
K = W.shape[1]
CHOL = {h: np.linalg.cholesky(cov[h] + 1e-9 * np.eye(K))
        for h in range(len(W)) if claim[h] >= 0}

Xtr, ytr, _, _ = load("fashion_mnist")
unit = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), EPS)
REAL_RAW = {c: Xtr[ytr == c][:1500] for c in range(10)}
REAL = {c: unit(center_norm(REAL_RAW[c])) for c in range(10)}
EXPERTS = {c: sorted(np.where(claim == c)[0].tolist(),
                     key=lambda h: -wins[h].sum()) for c in range(10)}

S = {"cls": 0, "h": EXPERTS[0][0], "c": mu[EXPERTS[0][0]].copy(),
     "top": np.argsort(np.diag(cov[EXPERTS[0][0]]))[::-1][:NSLIDER]}


def tile(img):
    """28x28 -> RGBA float32, min-max stretched, magnified."""
    v = img.reshape(28, 28)
    v = (v - v.min()) / max(v.max() - v.min(), 1e-9)
    v = np.repeat(np.repeat(v, SCALE, 0), SCALE, 1)
    out = np.empty((*v.shape, 4), np.float32)
    out[..., 0] = out[..., 1] = out[..., 2] = v
    out[..., 3] = 1.0
    return out.ravel()


def current():
    """The drawn vector, its label completion, and how real it looks."""
    v = S["c"] @ W[S["h"]]
    img, lab = v[:N_IMG], v[N_IMG:]
    c = int(claim[S["h"]])
    g = img / max(np.linalg.norm(img), EPS)
    sims = REAL[c] @ g
    j = int(sims.argmax())
    return img, lab, float(sims[j]), REAL_RAW[c][j], c


def roundtrip(via_label):
    """One pass of the expert's own output back through itself.

    via_label=False  the drawing is re-read as an image with the label blanked,
                     and the coefficients are re-derived from it: keeps this
                     particular garment, drops whatever was off the subspace.
    via_label=True   the figure's version -- keep only the label it just said
                     and draw from that alone, which lands on the prototype.
    """
    h = S["h"]
    v = S["c"] @ W[h]
    q = np.zeros(N_IMG + 10)
    if via_label:
        q[N_IMG:] = v[N_IMG:]
    else:
        q[:N_IMG] = v[:N_IMG]
    q = center_norm(q[None])[0]
    before = v[:N_IMG] / max(np.linalg.norm(v[:N_IMG]), EPS)
    S["c"] = q @ W[h].T
    after = (S["c"] @ W[h])[:N_IMG]
    after = after / max(np.linalg.norm(after), EPS)
    return float(before @ after)


def sample(mode, temp):
    h = S["h"]
    if mode == "mean":
        S["c"] = mu[h].copy()
    elif mode == "diagonal":
        S["c"] = mu[h] + temp * np.sqrt(np.maximum(np.diag(cov[h]), 0)) * \
            RNG.standard_normal(K)
    else:
        S["c"] = mu[h] + temp * (CHOL[h] @ RNG.standard_normal(K))


# --------------------------------------------------------------------------- #
def check():
    for c in range(10):
        S["cls"] = c; S["h"] = EXPERTS[c][0]
        S["top"] = np.argsort(np.diag(cov[S["h"]]))[::-1][:NSLIDER]
        for m in ("mean", "diagonal", "covariance"):
            sample(m, 1.0)
            img, lab, real, near, cc = current()
            assert tile(img).size == (28 * SCALE) ** 2 * 4
            print(f"  {CLASSES[c]:<12} e{S['h']:<3} {m:<11} says "
                  f"{CLASSES[int(lab.argmax())]:<12} realism {real:.3f}")
    print("ok")


def main():
    import dearpygui.dearpygui as dpg
    dpg.create_context()
    with dpg.texture_registry():
        for tag in ("tex_gen", "tex_real", "tex_proto"):
            dpg.add_raw_texture(28 * SCALE, 28 * SCALE, tile(np.zeros(784)),
                                format=dpg.mvFormat_Float_rgba, tag=tag)

    def redraw():
        img, lab, real, near, c = current()
        dpg.set_value("tex_gen", tile(img))
        dpg.set_value("tex_real", tile(near))
        dpg.set_value("tex_proto", tile(PROTO[c]))
        p = np.exp(lab - lab.max()); p /= p.sum()
        order = np.argsort(lab)[::-1]
        dpg.set_value("says", f"it names this: {CLASSES[int(order[0])]}   "
                              f"(then {CLASSES[int(order[1])]})")
        dpg.set_value("realism", f"nearest real image: {real:.3f}      "
                                 f"cos with class mean: "
                                 f"{float(img @ PROTO[c] / max(np.linalg.norm(img), EPS)):+.3f}")

    def push_sliders():
        h = S["h"]
        sd = np.sqrt(np.maximum(np.diag(cov[h]), 0))
        for i, t in enumerate(S["top"]):
            dpg.configure_item(f"sl{i}", label=f"template {t}",
                               min_value=float(mu[h][t] - 4 * sd[t] - 1e-6),
                               max_value=float(mu[h][t] + 4 * sd[t] + 1e-6))
            dpg.set_value(f"sl{i}", float(S["c"][t]))

    def on_slider(sender, value):
        S["c"][S["top"][int(sender[2:])]] = value
        redraw()

    def on_expert(sender, value):
        S["h"] = int(value.split()[0][1:])
        S["top"] = np.argsort(np.diag(cov[S["h"]]))[::-1][:NSLIDER]
        sample("mean", 0.0); push_sliders(); redraw()

    def on_class(sender, value):
        S["cls"] = CLASSES.index(value)
        items = [f"e{h} ({wins[h].sum()} won)" for h in EXPERTS[S["cls"]]]
        dpg.configure_item("expert", items=items, default_value=items[0])
        dpg.set_value("expert", items[0])
        on_expert(None, items[0])

    def on_draw(sender, app, user):
        sample(user, dpg.get_value("temp")); push_sliders(); redraw()

    def on_roundtrip(sender=None, app=None, user=None):
        keep = roundtrip(dpg.get_value("via_label"))
        dpg.set_value("moved", f"kept {keep:.4f} of the previous drawing "
                               f"(1.0 = settled)")
        push_sliders(); redraw()

    with dpg.window(tag="root"):
        dpg.add_text("draw from an expert")
        with dpg.group(horizontal=True):
            dpg.add_combo(CLASSES, default_value=CLASSES[0], width=150,
                          tag="cls", callback=on_class)
            dpg.add_combo([f"e{h} ({wins[h].sum()} won)" for h in EXPERTS[0]],
                          default_value=f"e{EXPERTS[0][0]} ({wins[EXPERTS[0][0]].sum()} won)",
                          width=170, tag="expert", callback=on_expert)
            dpg.add_button(label="mean", user_data="mean", callback=on_draw)
            dpg.add_button(label="diagonal", user_data="diagonal", callback=on_draw)
            dpg.add_button(label="covariance", user_data="covariance", callback=on_draw)
            dpg.add_slider_float(tag="temp", label="temperature", width=170,
                                 default_value=1.0, min_value=0.0, max_value=2.5)
        with dpg.group(horizontal=True):
            dpg.add_button(label="round trip", callback=on_roundtrip)
            dpg.add_checkbox(label="through the label (lands on the prototype)",
                             tag="via_label")
            dpg.add_text("", tag="moved")
        dpg.add_separator()
        with dpg.group(horizontal=True):
            for tag, cap in (("tex_gen", "drawn"), ("tex_real", "nearest real image"),
                             ("tex_proto", "class mean")):
                with dpg.group():
                    dpg.add_text(cap)
                    dpg.add_image(tag)
            with dpg.group():
                dpg.add_text("top coefficients", tag="hdr")
                for i in range(NSLIDER):
                    dpg.add_slider_float(tag=f"sl{i}", width=240, label="",
                                         callback=on_slider)
        dpg.add_separator()
        dpg.add_text("", tag="says")
        dpg.add_text("", tag="realism")

    push_sliders(); redraw()
    dpg.create_viewport(title="Zenith — what the experts can draw",
                        width=1180, height=560)
    dpg.setup_dearpygui(); dpg.show_viewport()
    dpg.set_primary_window("root", True)
    dpg.start_dearpygui(); dpg.destroy_context()


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
