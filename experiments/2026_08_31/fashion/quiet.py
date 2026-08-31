"""Dim whoever shouts too much -- at READ time, with no labels anywhere.

Conscience does this during training (who may learn). The reluctance does
something that looks similar at read time, but it needs the teacher to say
"wrong". This is the third thing: adjust each hypercolumn's gate offset until
no one wins far more of the inputs than its share, and never look at a label.

    b[h] += rate * (share[h] - target)     share = fraction of inputs h wins

target = 1/(number that are alive), i.e. perfectly equal airtime, and a
softer version at 1/2 strength for comparison. Fitted on the TRAIN inputs only
(their labels are never touched), read on test.
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


def equalise(E, alive, strength, iters=300, rate=0.02):
    """No labels: push offsets until airtime is even."""
    H = E.shape[1]
    n_alive = int(alive.sum())
    target = strength / n_alive
    b = np.zeros(H)
    for _ in range(iters):
        w = np.where(alive[None], E + b[None], np.inf).argmin(1)
        share = np.bincount(w, minlength=H) / len(E)
        b += rate * np.where(alive, share - target, 0.0)
    return b


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

    def acc(b):
        s = np.where(alive[None], Ete + b[None], np.inf)
        return float((claim[s.argmin(1)] == yte).mean())

    def busiest(b):
        w = np.where(alive[None], Ete + b[None], np.inf).argmin(1)
        return float((np.bincount(w, minlength=len(W)) / len(yte)).max())

    r = {"raw": acc(np.zeros(len(W))), "raw_busiest": busiest(np.zeros(len(W)))}
    for st, tag in ((1.0, "equal airtime"), (0.5, "half-strength")):
        b = equalise(Etr, alive, st)
        r[tag] = acc(b); r[tag + "_busiest"] = busiest(b)
    typ = np.ones(len(W)); typ[alive] = Etr[:, alive].mean(0)
    r["habitual error"] = acc(-typ)
    if "b" in z:
        r["teacher reluctance"] = acc(z["b"])
        r["teacher reluctance_busiest"] = busiest(z["b"])
    out[name] = r
    print(f"[{name}]")
    for k in ("raw", "equal airtime", "half-strength", "habitual error",
              "teacher reluctance"):
        if k in r:
            bz = f"   busiest {r[k+'_busiest']*100:5.1f}%" if k + "_busiest" in r else ""
            print(f"    {k:<22} {r[k]:.4f}{bz}")
(OUT / "quiet.json").write_text(json.dumps(out, indent=2))
