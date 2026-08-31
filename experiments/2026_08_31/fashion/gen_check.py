"""Does sleep damage the generative half? Same metrics as sample_quality.py,
across the three ways of correcting: none, repulsion on real data, sleep.

  own-class rebuild error   lower = the model still fits its own class
  imagination               cos(drawn from the label alone, class mean)
  realism                   cos to the NEAREST REAL image of the class
  self-similarity           mean pairwise cos among samples (lower = varied)
"""

import json, sys
from pathlib import Path
import numpy as np
from common import load, join, center_norm, EPS, CLASSES

OUT = Path(__file__).resolve().parent / "results"
N_IMG, NS = 784, 40
MODELS = [("wake (no correction)", "ws_wake.npz"),
          ("wake + sleep", "ws_wake_sleep.npz"),
          ("wake + repulsion", "both_w_lam4.0.npz")]


def rebuild(W, Q):
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
    R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
    e = np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG], axis=2) / np.maximum(
        np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS)
    return e, R


unit = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), EPS)
Xtr, ytr, Xte, yte = load("fashion_mnist")
proto = unit(center_norm(np.stack([Xtr[ytr == c].mean(0) for c in range(10)])))
out = {}
for name, fn in MODELS:
    p = OUT / fn
    if not p.exists():
        print(f"  {name}: {fn} not written yet"); continue
    z = np.load(p)
    W = z["W"].astype(np.float64); wins = z["wins"]
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.concatenate([np.where(alive[None], rebuild(W, join(Xte[s:s + 2000]))[0],
                                 np.inf) for s in range(0, len(Xte), 2000)])
    acc = float((claim[E.argmin(1)] == yte).mean())
    own = (claim[None, :] == yte[:, None]) & alive[None]
    with np.errstate(all="ignore"):
        oe = float(np.nanmean(np.nanmin(np.where(own, E, np.nan), 1)))
    picks = [int(np.where(claim == c)[0][wins[np.where(claim == c)[0]].sum(1).argmax()])
             for c in range(10) if (claim == c).any()]
    # imagination: label in, image out
    V = np.zeros((len(picks), N_IMG + 10))
    for i, h in enumerate(picks):
        V[i, N_IMG + claim[h]] = 1.0
    _, R = rebuild(W, center_norm(V))
    G = unit(np.stack([R[i, h, :N_IMG] for i, h in enumerate(picks)]))
    imag = float(np.mean([G[i] @ proto[claim[h]] for i, h in enumerate(picks)]))
    # realism / diversity of covariance samples
    rng = np.random.default_rng(0); rea, div = [], []
    for h in picks:
        c = claim[h]
        S = join(Xtr[ytr == c][:1500]) @ W[h].T
        mu, cov = S.mean(0), np.cov(S.T)
        w_, Vv = np.linalg.eigh(cov + 1e-8 * np.eye(W.shape[1]))
        A = Vv * np.sqrt(np.maximum(w_, 0))
        C = mu[None] + rng.standard_normal((NS, W.shape[1])) @ A.T
        Gs = unit((C @ W[h])[:, :N_IMG])
        Rr = unit(center_norm(Xtr[ytr == c][:1500]))
        rea.append(float((Gs @ Rr.T).max(1).mean()))
        P = Gs @ Gs.T
        div.append(float((P.sum() - np.trace(P)) / (NS * (NS - 1))))
    out[name] = {"acc": acc, "own_class_err": oe, "imagination": imag,
                 "realism": float(np.mean(rea)), "self_similarity": float(np.mean(div))}
    print(f"  {name:<22} acc {acc:.4f}   own-err {oe:.4f}   imagination {imag:+.4f}"
          f"   realism {np.mean(rea):.4f}   self-sim {np.mean(div):.4f}")
print(f"  {'real images':<22} {'':>29}   realism 0.8961   self-sim 0.6194")
(OUT / "gen_check.json").write_text(json.dumps(out, indent=2))
