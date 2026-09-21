"""Does a single latent angle encode a style?

Take the class chord, move ONE angle by a multiple of that angle's measured spread across
real images of the class, invert, draw.  Top row block: the angles with the largest spread
(the ones the data varies along).  Bottom block: the smallest-spread angles, as controls.
Second figure: the same traversal as signed differences from the chord image, so a subtle
change is visible.

    uv run python experiments/2026_09_21/trigflow/traverse.py --cls 3
"""
import argparse
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trigflow as T

ap = argparse.ArgumentParser()
ap.add_argument("--cls", type=int, default=3)
ap.add_argument("--tag", default="mnist_L8_span1pi_cap2pi_pull1")
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--bottom", type=int, default=3)
a = ap.parse_args()

ck = torch.load(T.OUT / f"{a.tag}.pt", map_location=T.dev); args = ck["args"]; span = args["span"]
net = T.TrigFlow(784, args["depth"], args["cap"]).to(T.dev); net.load_state_dict(ck["net"]); net.eval()
X, y = T.mnist("test", args["data"])
Xc = T.to_angle(X[y == a.cls], span).to(T.dev)
chord = net.chords[a.cls]

with torch.no_grad():
    Z = net(Xc)
    dev = T.wrap(Z - chord)                                   # deviation of real latents from the chord
    spread = dev.std(0)                                       # per-angle spread across the class
    order = torch.argsort(spread, descending=True)
    picks = torch.cat([order[:a.top], order[-a.bottom:]]).tolist()
    steps = [-2, -1, 0, 1, 2]
    base = T.to_pixel(net.inverse(chord[None]), span)[0]
    rows = []
    for i in picks:
        th = chord[None].repeat(len(steps), 1)
        th[:, i] = chord[i] + torch.tensor(steps, device=T.dev, dtype=torch.float32) * spread[i]
        rows.append(T.to_pixel(net.inverse(T.wrap(th)), span))
    rows = torch.stack(rows).cpu().numpy()                   # picks x steps x 784
base = base.cpu().numpy()

print(f"class {a.cls}: {len(Xc)} test images.  per-angle spread: median {spread.median():.2f} rad, "
      f"max {spread.max():.2f}, min {spread.min():.2f}")
print(f"{'angle':>6} {'pixel':>8} {'spread':>7} {'pixel L1 change per +2 spread':>30}")
for k, i in enumerate(picks):
    print(f"{i:>6} {f'({i//28},{i%28})':>8} {spread[i]:7.2f} {np.abs(rows[k, -1] - base).mean():30.4f}"
          + ("   <- control, smallest spread" if k >= a.top else ""))

for kind in ("image", "diff"):
    fig, axes = plt.subplots(len(picks), len(steps), figsize=(len(steps) * .75, len(picks) * .78))
    for k, i in enumerate(picks):
        for j, s in enumerate(steps):
            ax = axes[k, j]
            if kind == "image":
                ax.imshow(rows[k, j].reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow((rows[k, j] - base).reshape(28, 28), cmap="RdBu_r", vmin=-.5, vmax=.5, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0: ax.set_title(f"{s:+d} sd" if s else "chord", fontsize=7)
        axes[k, 0].set_ylabel(f"θ[{i}]\nsd {spread[i]:.2f}" + ("\nctrl" if k >= a.top else ""),
                              fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle(f"class {a.cls}: one latent angle moved at a time" + (", difference from the chord image" if kind == "diff" else ""),
                 fontsize=8)
    fig.tight_layout(rect=[0.04, 0, 1, 0.96]); fig.savefig(T.OUT / f"traverse_{a.cls}_{kind}.png", dpi=150)
print(f"-> {T.OUT}/traverse_{a.cls}_image.png, traverse_{a.cls}_diff.png")
