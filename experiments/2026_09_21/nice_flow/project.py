"""Is there a point NEAR a class chord that decodes to a clean image?

f^-1(chord_y) is speckled, and averaging cannot fix it because the speckle is
deterministic, not noise (see README §V).  The suspicion is that the chord is simply
badly placed: nothing in training ever required the chord ITSELF to decode to a valid
image, only that real images land near it.

So freeze the network and optimise the POINT instead of the weights:

    minimise over theta      ||wrap(theta - chord_y)||^2          stay in the cluster
                           + lam * sum dist(phi, [0, pi])         phi = f^-1(theta)
                                                                  penalise the empty half

At span = pi the data occupies [0, pi] and the other half of the circle is empty; 41% of
f^-1(chord) lands there and clips to black or white.  That is the speckle.

Reported for each lambda: how far theta had to move, how much of the decode is still out
of band, and -- the control that decides it -- whether theta still classifies as y.
"""
import numpy as np, torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import app                                         # loads the frozen checkpoint

dev, D, NC = app.dev, 784, 10
OUT = Path(__file__).resolve().parent / "results" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
TAU, PI = app.TF.TAU, np.pi
CH = app.CHORDS


def wrap(d):
    return (d + PI) % TAU - PI


def oob(phi):
    """distance from each decoded angle to the valid arc [0, pi].  0 when in band."""
    p = phi % TAU
    return torch.where(p <= PI, torch.zeros_like(p), torch.minimum(p - PI, TAU - p))


def optimise(lam, steps=300, lr=0.05):
    th = CH.clone().requires_grad_(True)            # start at the chords, all 10 at once
    opt = torch.optim.Adam([th], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        phi = app.flow.inverse(th % TAU)
        pull = (wrap(th - CH) ** 2).sum(1)
        pen = (oob(phi) ** 2).sum(1)
        (pull + lam * pen).sum().backward()
        opt.step()
    return th.detach() % TAU


def report(name, th):
    with torch.no_grad():
        phi = app.flow.inverse(th)
        frac = (oob(phi) > 1e-6).float().mean().item()
        moved = wrap(th - CH).abs().mean().item()
        cs = torch.cos(th[:, None, :] - CH[None]).mean(-1)
        pred = cs.argmax(1).cpu().numpy()
        own = cs[torch.arange(NC), torch.arange(NC)].mean().item()
        img = np.clip(app.TF.to_x(phi).cpu().numpy(), 0, 1)
    ok = int((pred == np.arange(NC)).sum())
    print(f"  {name:<22} moved {moved:6.3f} rad   out-of-band {frac:6.1%}   "
          f"cos to own chord {own:.3f}   still classifies {ok}/10")
    return img


print(f"start: theta = the chords themselves\n")
rows = [("chord (lam=0)", report("chord (lam=0)", CH.clone()))]
for lam in (0.3, 1.0, 3.0, 10.0, 30.0):
    rows.append((f"lam={lam:g}", report(f"lam={lam:g}", optimise(lam))))

fig, axes = plt.subplots(len(rows), NC, figsize=(NC * .66, len(rows) * .72))
for r, (name, img) in enumerate(rows):
    for c in range(NC):
        axes[r, c].imshow(img[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0: axes[r, c].set_title(str(c), fontsize=7, pad=2)
    axes[r, 0].set_ylabel(name, fontsize=6.5, rotation=0, ha="right", va="center")
fig.suptitle("Optimising the point instead of the weights", fontsize=10)
fig.tight_layout(rect=[0.06, 0, 1, 0.94]); fig.savefig(OUT / "projected.png", dpi=150)
print(f"\n  -> {OUT}/torusflow_projected.png")
