"""What does each hypercolumn actually say, and how sure is it?

For every hypercolumn we blank the label, let it complete, and normalise the
completion. Two views:

  said[h, c]   what it says on average over ALL test images -- its default
  conf[h, c]   its confidence in class c when shown images OF class c
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, join, EPS, CLASSES

OUT = Path(__file__).resolve().parent / "results"
N_IMG = 784
z = np.load(OUT / "coef_stats.npz")
W, wins, claim = z["W"].astype(np.float64), z["wins"], z["claim"]
H, K, D = W.shape
_, _, Xte, yte = load("fashion_mnist")

L = []
for s in range(0, len(Xte), 2000):
    Q = join(Xte[s:s + 2000])
    S = (Q @ W.reshape(H * K, D).T).reshape(len(Q), H, K)
    R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
    L.append(R[:, :, N_IMG:])
L = np.concatenate(L)                                   # (n, H, 10)
Ln = L / np.maximum(np.linalg.norm(L, axis=2, keepdims=True), EPS)

said = Ln.mean(0)                                       # (H, 10)
conf = np.stack([Ln[yte == c].mean(0)[:, c] for c in range(10)], 1)   # (H, 10)
own_when_own = np.array([conf[h, claim[h]] if claim[h] >= 0 else np.nan
                         for h in range(H)])
own_when_other = np.array([
    np.nanmean([conf[h, c] for c in range(10) if c != claim[h]])
    if claim[h] >= 0 else np.nan for h in range(H)])
names_own = np.array([float((Ln[:, h].argmax(1) == claim[h]).mean())
                      if claim[h] >= 0 else np.nan for h in range(H)])

alive = wins.sum(1) > 0
use = np.zeros(H)
E = []
for s in range(0, len(Xte), 2000):
    Q = join(Xte[s:s + 2000])
    S = (Q @ W.reshape(H * K, D).T).reshape(len(Q), H, K)
    R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
    E.append(np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG], axis=2) /
             np.maximum(np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS))
E = np.where(alive[None], np.concatenate(E), np.inf)
w = E.argmin(1)
use = np.bincount(w, minlength=H) / len(yte)

print(f"{'expert':<8}{'claims':<13}{'gate use':>9}{'conf in own class':>19}"
      f"{'on other classes':>18}{'names own class':>17}")
for h in np.argsort(use)[::-1]:
    if not alive[h]:
        continue
    print(f"e{h:<7}{CLASSES[claim[h]][:12]:<13}{use[h]*100:8.2f}%"
          f"{own_when_own[h]:>19.3f}{own_when_other[h]:>18.3f}"
          f"{names_own[h]*100:>16.1f}%")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
o = np.argsort(use)[::-1]
o = [h for h in o if alive[h]]
for a, M, t in ((ax[0], said[o], "what it says on ANY image"),
                (ax[1], conf[o], "confidence in class c, on images of class c")):
    im = a.imshow(M, cmap="magma", aspect="auto", vmin=-0.3, vmax=1.0)
    a.set_xticks(range(10)); a.set_xticklabels(CLASSES, rotation=90, fontsize=7)
    a.set_yticks(range(len(o)))
    a.set_yticklabels([f"e{h} · {CLASSES[claim[h]][:9]}" for h in o], fontsize=6)
    a.set_title(t, fontsize=9); fig.colorbar(im, ax=a, fraction=.046)
fig.tight_layout(); fig.savefig(OUT / "90_confidences.png", dpi=140); plt.close(fig)
(OUT / "confidences.json").write_text(json.dumps(
    {"said": said.tolist(), "conf_on_own_class_images": conf.tolist(),
     "claim": claim.tolist(), "gate_use": use.tolist()}, indent=2))
print("\n-> 90_confidences.png")
