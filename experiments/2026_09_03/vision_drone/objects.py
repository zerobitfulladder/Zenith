"""What an object template IS: the mean of the frames it wins, its count, and
the command the tally reads for it. 36 most-used objects, from the current
pupil.npz over 8000 training frames."""
import json
from pathlib import Path
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vrig as V

HERE = Path(__file__).resolve().parent; OUT = HERE / "results"
npz = np.load(OUT / "pupil.npz"); cfg = json.load(open(OUT / "pupil.json"))
C = int(cfg["frames"])
rig = V.VRig(48, int(cfg["ps"]), C, int(cfg["grid"]), int(cfg["k1"]))
W1, W2 = cp.asarray(npz["W1"]), cp.asarray(npz["W2"])
mu, sd = cp.asarray(npz["mu"]), cp.asarray(npz["sd"])
TL, TR = npz["TL"], npz["TR"]
d = np.load(OUT / "flight.npz"); F = np.load(OUT / "frames.npy").astype(np.float32) / 255.0
tick = d["tick"]
rng = np.random.default_rng(1); idx = rng.choice(len(F), 8000, replace=False)
Fi = F[idx]
if C == 2:
    prev = np.roll(F, 1, axis=0); prev[tick == 0] = F[tick == 0]
    D = (Fi - prev[idx]) if cfg.get("diff") == "signed" else 0.5 + 0.5 * (Fi - prev[idx])
    X = np.stack([Fi, D], 1).reshape(len(Fi), -1)
else:
    X = Fi.reshape(len(Fi), -1)
Cc = V.standardize(V.encode_contrast(rig, W1, cp.asarray(X)), mu, sd)
win = cp.asnumpy((Cc @ W2.T).argmax(1))
cnt = np.bincount(win, minlength=W2.shape[0])
pick = np.argsort(-cnt)[:36]
fig, axes = plt.subplots(6, 6, figsize=(12, 12.6))
for i, t in enumerate(pick):
    ax = axes[i // 6, i % 6]
    m = Fi[win == t].mean(0)
    ax.imshow(m, cmap="gray", vmin=0, vmax=max(0.2, m.max()))
    ax.set_title(f"obj {t}: {cnt[t]} frames, reads L{int(TL[t].argmax())} R{int(TR[t].argmax())}", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"36 most-used object templates: the MEAN FRAME each one wins (of 8000), and the command it reads. "
             f"Live objects: {(cnt > 0).sum()}/{W2.shape[0]}", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / f"objects_mean_{C}ch.png", dpi=120)
print(f"live objects {(cnt > 0).sum()}/{W2.shape[0]}, top-36 cover {cnt[pick].sum()/len(Fi):.2f} of frames; wrote objects_mean_{C}ch.png")
