"""Don't average the hypercolumns -- pick the one that's most sure, and use it.

Loads the trained weights, so no training happens here. Six ways of asking a
hypercolumn "how confident are you about this image", plus two bounds:

  oracle      -- if ANY of the 30 is right, count it right. The ceiling any
                 gate could reach, and the honest measure of whether the
                 information is present at all.
  anti-oracle -- if any is wrong, count it wrong. The floor.

If the oracle is high and every real gate is low, the answers are in there
and the SELECTION is what fails -- which is the thing the whole
one-winner-hypercolumn architecture depends on.
"""

import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
EPS = 1e-12
from ensemble import load, join, center_norm, H, K          # noqa: E402


def main():
    Xtr, ytr, Xte, yte = load()
    n_img = Xtr.shape[1]
    W = np.load(OUT / "weights.npz")["W"].astype(np.float64)
    print(f"loaded {W.shape[0]} hypercolumns x {W.shape[1]} templates")
    Q = join(Xte)                                     # image written, label blank
    n = len(Q)

    P = np.zeros((len(W), n, 10))                     # completed label block
    rec = np.zeros((len(W), n))                       # image-half rebuild error
    for s in range(0, n, 500):                        # chunked: (H,500,794)
        b = Q[s:s + 500]
        S = (b @ W.reshape(len(W) * K, -1).T).reshape(len(b), len(W), K).transpose(1, 0, 2)
        R = np.matmul(S, W)
        P[:, s:s + 500] = R[:, :, n_img:]
        d = b[None, :, :n_img] - R[:, :, :n_img]
        rec[:, s:s + 500] = np.linalg.norm(d, axis=2) / np.maximum(
            np.linalg.norm(b[None, :, :n_img], axis=2), EPS)

    pred = P.argmax(2)                                # (H, n)
    correct = (pred == yte[None])
    srt = np.sort(P, axis=2)
    res = {}

    gates = {
        "margin (top1 - top2)":            srt[:, :, -1] - srt[:, :, -2],
        "margin / label-block norm":       (srt[:, :, -1] - srt[:, :, -2]) /
                                           np.maximum(np.linalg.norm(P, axis=2), EPS),
        "top-1 score":                     srt[:, :, -1],
        "label-block norm (commitment)":   np.linalg.norm(P, axis=2),
        "negative softmax entropy":        -(lambda p: -(p * np.log(p + 1e-12)).sum(2))(
                                               np.exp(P - P.max(2, keepdims=True)) /
                                               np.exp(P - P.max(2, keepdims=True)).sum(2, keepdims=True)),
        "best image reconstruction":       -rec,
    }
    for name, conf in gates.items():
        pick = conf.argmax(0)                          # (n,) which hypercolumn
        acc = float((pred[pick, np.arange(n)] == yte).mean())
        share = np.bincount(pick, minlength=len(W)) / n
        res[name] = {"accuracy": acc, "busiest_hypercolumn_share": float(share.max()),
                     "hypercolumns_ever_picked": int((share > 0).sum())}
        print(f"  {name:<32} {acc:.4f}   busiest column picked "
              f"{share.max()*100:4.1f}% of the time, {int((share>0).sum())}/{len(W)} used")

    orc = float(correct.any(0).mean())
    anti = float(correct.all(0).mean())
    res["oracle_any_right"] = orc
    res["anti_oracle_all_right"] = anti
    res["single_mean"] = float(correct.mean(1).mean())
    prev = json.loads((OUT / "metrics.json").read_text())
    res["soft_vote_30"] = prev["curve"]["30"]["soft_mean"]
    res["majority_30"] = prev["curve"]["30"]["hard_mean"]
    print(f"\n  {'ORACLE (any of the 30 is right)':<32} {orc:.4f}")
    print(f"  {'all 30 right':<32} {anti:.4f}")
    print(f"  {'one hypercolumn (mean)':<32} {res['single_mean']:.4f}")
    print(f"  {'soft vote, all 30':<32} {res['soft_vote_30']:.4f}")
    print(f"  {'majority vote, all 30':<32} {res['majority_30']:.4f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = list(gates) + ["one column (mean)", "majority vote", "soft vote", "ORACLE"]
    val = [res[g]["accuracy"] for g in gates] + [
        res["single_mean"], res["majority_30"], res["soft_vote_30"], orc]
    col = ["#c1462d"] * len(gates) + ["#888", "#888", "#1b6ca8", "#2f8f4e"]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.barh(range(len(lab)), val, color=col)
    for i, v in enumerate(val):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=7)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlim(0, 1.0); ax.grid(alpha=.25, axis="x")
    ax.set_title("pick the most confident hypercolumn, vs averaging them", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "02_gating.png", dpi=130); plt.close(fig)
    (OUT / "gating.json").write_text(json.dumps(res, indent=2))
    print(f"\n-> {OUT/'02_gating.png'}")


if __name__ == "__main__":
    main()
