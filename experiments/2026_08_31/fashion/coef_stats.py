"""The other half of the generative model: how an expert's coefficients co-vary.

Training only ever learned WHICH directions an expert spans. It never learned
how they are normally combined, so sampling them independently gives grain.
Here we measure, per expert, the mean and the full covariance of its
coefficients on its own class, and compare three ways of drawing:

    mean       the deterministic prototype
    diagonal   each coefficient sampled independently (what dream.py did)
    full       sampled from the covariance -- directions move together

Writes coef_stats.npz, which is also what the UI loads.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, join, center_norm, EPS, CLASSES

OUT = Path(__file__).resolve().parent / "results"
N_IMG, LAM, NPC = 784, "4.0", 2000
RNG = np.random.default_rng(0)


def main():
    z = np.load(OUT / f"both_w_lam{LAM}.npz")
    W, wins = z["W"].astype(np.float64), z["wins"]
    H, K, D = W.shape
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    Xtr, ytr, _, _ = load("fashion_mnist")
    proto = center_norm(np.stack([Xtr[ytr == c].mean(0) for c in range(10)]))
    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), EPS)

    mu = np.zeros((H, K)); cov = np.zeros((H, K, K)); n = np.zeros(H, int)
    for h in range(H):
        if claim[h] < 0:
            continue
        X = Xtr[ytr == claim[h]][:NPC]
        S = join(X) @ W[h].T
        mu[h], cov[h], n[h] = S.mean(0), np.cov(S.T), len(S)
    np.savez_compressed(OUT / "coef_stats.npz", W=W.astype(np.float32),
                        wins=wins, claim=claim, mu=mu.astype(np.float32),
                        cov=cov.astype(np.float32), proto=proto.astype(np.float32),
                        n=n)

    def draw(h, mode, rng):
        if mode == "mean":
            c = mu[h]
        elif mode == "diagonal":
            c = mu[h] + np.sqrt(np.maximum(np.diag(cov[h]), 0)) * rng.standard_normal(K)
        else:
            L = np.linalg.cholesky(cov[h] + 1e-9 * np.eye(K))
            c = mu[h] + L @ rng.standard_normal(K)
        return (c @ W[h])[:N_IMG]

    picks = [int(np.where(claim == c)[0][wins[np.where(claim == c)[0]].sum(1).argmax()])
             for c in range(10) if (claim == c).any()]
    modes = ["mean"] + ["diagonal"] * 3 + ["full"] * 4
    fig, axes = plt.subplots(len(picks), len(modes) + 1,
                             figsize=(1.05 * (len(modes) + 1), 1.15 * len(picks)))
    score = {"mean": [], "diagonal": [], "full": []}
    for r, h in enumerate(picks):
        c = claim[h]
        a = axes[r, 0]
        a.imshow(proto[c].reshape(28, 28), cmap="gray")
        a.set_xticks([]); a.set_yticks([]); a.set_title("class mean", fontsize=5.5, pad=1.5)
        a.set_ylabel(f"e{h}\n{CLASSES[c][:9]}", fontsize=5.5, rotation=0,
                     ha="right", va="center")
        for j, m in enumerate(modes):
            g = draw(h, m, RNG)
            cs = float(g @ proto[c] / (np.linalg.norm(g) + EPS))
            score[m].append(cs)
            a = axes[r, 1 + j]
            a.imshow(g.reshape(28, 28), cmap="gray")
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{m[:4]} {cs:+.2f}", fontsize=5.5, pad=1.5)
    fig.suptitle("the coefficient distribution is the missing half of the model",
                 fontsize=9)
    fig.subplots_adjust(left=.085, right=.995, top=.91, bottom=.005,
                        wspace=.06, hspace=.30)
    fig.savefig(OUT / "73_covariance.png", dpi=140); plt.close(fig)

    res = {m: float(np.mean(v)) for m, v in score.items()}
    res["prior_independent_all36"] = 0.460
    res["prior_independent_top8"] = 0.593
    res["prior_pure_noise"] = -0.024
    print(json.dumps(res, indent=2))
    (OUT / "coef_stats.json").write_text(json.dumps(
        {"cosine_with_class_mean": res,
         "experts": [[int(h), int(claim[h])] for h in picks],
         "samples_per_expert": int(NPC)}, indent=2))
    print(f"-> 73_covariance.png, coef_stats.npz")


if __name__ == "__main__":
    main()
