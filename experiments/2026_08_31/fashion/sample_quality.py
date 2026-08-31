"""Cosine to the class mean rewards being average, so it cannot rank samplers.

Two honest numbers instead, per sampling mode:
  realism    cosine to the NEAREST REAL image of that class (higher = plausible)
  diversity  mean pairwise cosine BETWEEN samples (lower = more varied)

A prototype scores high realism and terrible diversity by construction; the
question is whether the covariance buys diversity without losing realism.
Real images of the class are the reference: their own realism/diversity is the
ceiling and the target.
"""

import json
from pathlib import Path
import numpy as np
from common import load, join, center_norm, EPS, CLASSES

OUT = Path(__file__).resolve().parent / "results"
N_IMG, NS = 784, 40
RNG = np.random.default_rng(0)

z = np.load(OUT / "coef_stats.npz")
W, claim, mu, cov = (z["W"].astype(np.float64), z["claim"],
                     z["mu"].astype(np.float64), z["cov"].astype(np.float64))
wins = z["wins"]
K = W.shape[1]
Xtr, ytr, _, _ = load("fashion_mnist")

unit = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), EPS)
picks = [int(np.where(claim == c)[0][wins[np.where(claim == c)[0]].sum(1).argmax()])
         for c in range(10) if (claim == c).any()]
res = {}
for mode in ("mean", "diagonal", "full", "real"):
    rea, div = [], []
    for h in picks:
        c = claim[h]
        R = unit(center_norm(Xtr[ytr == c][:1500]))
        if mode == "real":
            G = unit(center_norm(Xtr[ytr == c][1500:1500 + NS]))
        else:
            if mode == "mean":
                C = np.repeat(mu[h][None], NS, 0)
            elif mode == "diagonal":
                C = mu[h] + np.sqrt(np.maximum(np.diag(cov[h]), 0)) * \
                    RNG.standard_normal((NS, K))
            else:
                L = np.linalg.cholesky(cov[h] + 1e-9 * np.eye(K))
                C = mu[h] + RNG.standard_normal((NS, K)) @ L.T
            G = unit((C @ W[h])[:, :N_IMG])
        rea.append(float((G @ R.T).max(1).mean()))
        P = G @ G.T
        div.append(float((P.sum() - np.trace(P)) / (NS * (NS - 1))))
    res[mode] = {"realism": float(np.mean(rea)), "diversity": float(np.mean(div))}
    print(f"  {mode:<9} realism {res[mode]['realism']:.4f}   "
          f"self-similarity {res[mode]['diversity']:.4f}")
(OUT / "sample_quality.json").write_text(json.dumps(res, indent=2))
