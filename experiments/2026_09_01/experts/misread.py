"""The 6% the counted table gets wrong, and what it calls them instead."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experts as E, vote as V, settle as S

OUT = Path(__file__).resolve().parent / "results"
NL = 10


def main():
    W = np.load(OUT / "weights_lam0.0.npz")["W"].astype(np.float64)
    Xtr, ytr, Xte, yte = E.load()
    w = lambda X: np.concatenate([np.where(kp, f.argmax(1), -1) for f, kp in
                                  [S.fits(W, X[a:a + 1000]) for a in range(0, len(X), 1000)]])
    T = V.llr_table(w(Xtr), ytr, len(W), True)
    sc = V.llr_score(w(Xte), T, True)
    pred = sc.argmax(1)
    bad = np.nonzero(pred != yte)[0]
    print(f"accuracy {float((pred == yte).mean()):.4f}   {len(bad)} wrong of {len(yte)}")

    cm = np.zeros((NL, NL), int); np.add.at(cm, (yte, pred), 1)
    off = cm - np.diag(np.diag(cm))
    top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:8], off.shape))[0]
    print("worst confusions: " + "  ".join(f"{a}->{b} ({off[a,b]})" for a, b in top))

    # margin: how close was the right answer?
    z = (sc - sc.mean(1, keepdims=True)) / (sc.std(1, keepdims=True) + 1e-9)
    margin = z[bad, pred[bad]] - z[bad, yte[bad]]
    order = bad[np.argsort(margin)]          # near-misses first
    show = np.concatenate([order[:16], order[-16:]])
    fig, ax = plt.subplots(4, 8, figsize=(13, 7.4))
    for k, (a, i) in enumerate(zip(ax.ravel(), show)):
        a.imshow(Xte[i], cmap="gray_r"); a.axis("off")
        m = z[i, pred[i]] - z[i, yte[i]]
        a.set_title(f"{yte[i]} -> {pred[i]}   m={m:.2f}", fontsize=8,
                    color="tab:orange" if k < 16 else "tab:red")
    plt.suptitle("MNIST misclassified by the counted table (0.9423).  "
                 "top two rows = narrowest margin, bottom two = most confident errors",
                 fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "misread.png", dpi=125); plt.close()
    (OUT / "misread.json").write_text(json.dumps(
        {"accuracy": float((pred == yte).mean()), "n_wrong": int(len(bad)),
         "confusion": cm.tolist(),
         "worst_pairs": [[int(a), int(b), int(off[a, b])] for a, b in top],
         "median_margin": float(np.median(margin))}, indent=2))
    print(f"median margin of the errors {np.median(margin):.2f}  -> results/misread.png")


if __name__ == "__main__":
    main()
