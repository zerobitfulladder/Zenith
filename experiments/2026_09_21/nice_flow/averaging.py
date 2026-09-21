"""Does averaging samples around a chord converge to the prototype?

Rerun after the decode fix.  The first version of this measured through a decode that sent
angles just BELOW zero to pure white instead of black, so every number here was taken
through a rendering bug.
"""
import numpy as np, torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import app

dev, D = app.dev, 784
OUT = Path(__file__).resolve().parent / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(1)


def samples(label, sd, n, bs=2048):
    out = []
    for i in range(0, n, bs):
        m = min(bs, n - i)
        th = app.CHORDS[label:label+1] + torch.randn(m, D, device=dev) * sd
        out.append(app.decode(th))
    return torch.cat(out)


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


for sd in (0.15, 0.35):
    print(f"\n=== sd = {sd} ===")
    S = samples(3, sd, 8192)
    proto = torch.tensor(app.PROTO[3].ravel(), device=dev)
    big = S.mean(0)
    print(f"  per-pixel std across samples  mean {S.std(0).mean():.3f}  max {S.std(0).max():.3f}")
    print(f"  corr between two single samples  {np.mean([corr(S[i], S[i+1]) for i in range(0,200,2)]):.3f}")
    print(f"  {'N':>6}  {'corr to prototype':>18}  {'corr to mean-of-8192':>21}")
    for N in (1, 4, 16, 64, 256, 1024, 8192):
        print(f"  {N:6d}  {corr(S[:N].mean(0), proto):18.3f}  {corr(S[:N].mean(0), big):21.3f}")

Ns = (1, 4, 16, 64, 256, 1024, 8192)
fig, axes = plt.subplots(5, len(Ns)+1, figsize=((len(Ns)+1)*.72, 5*.78))
for r, cls in enumerate([0, 3, 4, 7, 8]):
    S = samples(cls, 0.15, 8192)
    for c, N in enumerate(Ns):
        axes[r, c].imshow(S[:N].mean(0).cpu().numpy().reshape(28,28), cmap="gray",
                          vmin=0, vmax=1, interpolation="nearest")
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0: axes[r, c].set_title(f"N={N}", fontsize=6.5, pad=2)
    axes[r, -1].imshow(app.PROTO[cls], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[r, -1].set_xticks([]); axes[r, -1].set_yticks([])
    if r == 0: axes[r, -1].set_title("sd=0", fontsize=6.5, pad=2)
    axes[r, 0].set_ylabel(str(cls), fontsize=8, rotation=0, ha="right", va="center")
fig.suptitle("Averaging N samples at sd=0.15", fontsize=10)
fig.tight_layout(rect=[0.02,0,1,0.94]); fig.savefig(OUT/"averaging.png", dpi=150)
print(f"\n  -> {OUT}/averaging.png")
