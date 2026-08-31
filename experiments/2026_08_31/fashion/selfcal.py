"""Can the gate be calibrated without a teacher?

The reluctance needs someone to say "wrong". But most of what it corrects is
that some hypercolumns are simply better fitters than others for reasons
unrelated to class. That is measurable from the model's own statistics:

    typ[h] = the error h habitually makes            (no labels)
    gate on  e[h] - typ[h]   or   e[h] / typ[h]

Two ways to define "habitually": over all inputs, or only over the inputs h
already wins. Both are unsupervised. Reported against the teacher-trained
reluctance on the same weights.
"""

import json
from pathlib import Path
import numpy as np
from common import load, join, EPS, CLASSES

OUT = Path(__file__).resolve().parent / "results"
N_IMG = 784
MODELS = [("graded λ=4, no punishment", "ws_wake.npz"),
          ("graded λ=4 + sleep", "ws_wake_sleep.npz"),
          ("graded λ=4 + repulsion", "both_w_lam4.0.npz")]


def errs(W, X, alive, chunk=2000):
    h, k, d = W.shape
    E = np.empty((len(X), h))
    for s in range(0, len(X), chunk):
        Q = join(X[s:s + chunk])
        S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
        R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
        E[s:s + chunk] = np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG],
                                        axis=2) / np.maximum(
            np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS)
    return np.where(alive[None], E, np.inf)


Xtr, ytr, Xte, yte = load("fashion_mnist")
out = {}
for name, fn in MODELS:
    p = OUT / fn
    if not p.exists():
        continue
    z = np.load(p)
    W, wins = z["W"].astype(np.float64), z["wins"]
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    Etr, Ete = errs(W, Xtr, alive), errs(W, Xte, alive)
    def acc(S):
        S = np.where(alive[None], S, np.inf)          # dead stay out of argmin
        return float((claim[np.nan_to_num(S, nan=np.inf).argmin(1)] == yte).mean())
    r = {"raw": acc(Ete)}
    # (a) habitual error over everything
    typ_all = np.ones(len(W))
    typ_all[alive] = Etr[:, alive].mean(0)
    r["minus habitual (all inputs)"] = acc(Ete - typ_all[None])
    r["divided by habitual (all inputs)"] = acc(Ete / typ_all[None])
    # (b) habitual error over the inputs it already wins, iterated to a fixed point
    w = Etr.argmin(1)
    typ_w = np.ones(len(W))
    for _ in range(5):
        typ_w = np.array([Etr[w == h, h].mean() if (w == h).any() and alive[h] else 1.0
                          for h in range(len(W))])
        w = np.where(alive[None], Etr - typ_w[None], np.inf).argmin(1)
    r["minus habitual (own wins, iterated)"] = acc(Ete - typ_w[None])
    if "b" in z:
        r["teacher-trained reluctance"] = acc(Ete + z["b"][None])
    out[name] = r
    print(f"[{name}]")
    for k, v in r.items():
        print(f"    {k:<38} {v:.4f}")
(OUT / "selfcal.json").write_text(json.dumps(out, indent=2))
