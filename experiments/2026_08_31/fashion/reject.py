"""Practice, not learning: the teacher says only "no", and only the selector moves.

Content is frozen. We take the trained competitive weights and add ONE scalar
per expert, a reluctance b_h, so the gate scores  -err_h(x) - b_h.

Training the reluctances is the parent's "no": show an image, let the current
gate pick, and if the winner's claimed class is wrong, make it slightly more
reluctant; make the best expert that does claim the true class slightly less
reluctant. No template ever rotates.

Three reference points, all read from the same frozen errors:
  raw fit gate      what we have now
  reluctance gate   the above (40 numbers learned)
  logistic on errs  a full classifier on the 40 error values -- the ceiling on
                    how much class information the fit vector holds at all
"""

import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, join, EPS, CLASSES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ARM = "competitive"
EPOCHS, LR = 12, 0.002


def errors(W, X, n_img=784, chunk=2000):
    """(n, H) relative reconstruction error of the image half, per expert."""
    h, k, d = W.shape
    Wf = W.reshape(h * k, d)
    out = np.empty((len(X), h))
    for s in range(0, len(X), chunk):
        Q = join(X[s:s + chunk])
        S = (Q @ Wf.T).reshape(len(Q), h, k)
        R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)   # (n, h, d)
        num = np.linalg.norm(Q[:, None, :n_img] - R[:, :, :n_img], axis=2)
        out[s:s + chunk] = num / np.maximum(
            np.linalg.norm(Q[:, :n_img], axis=1)[:, None], EPS)
    return out


def main():
    z = np.load(OUT / "l1_weights.npz")
    W = z[ARM].astype(np.float64)
    wins = z[ARM + "_wins"]
    live = wins.sum(1) > 0
    claim = np.where(live, wins.argmax(1), -1)
    H = len(claim)

    Xtr, ytr, Xte, yte = load("fashion_mnist")
    Etr, Ete = errors(W, Xtr), errors(W, Xte)
    Etr[:, ~live] = np.inf; Ete[:, ~live] = np.inf

    def acc(E, y, b):
        pick = (E + b[None]).argmin(1)
        return float((claim[pick] == y) .mean()), pick

    b = np.zeros(H)
    raw_tr, _ = acc(Etr, ytr, b)
    raw_te, _ = acc(Ete, yte, b)
    print(f"raw fit gate        train {raw_tr:.4f}   test {raw_te:.4f}")

    rng = np.random.default_rng(0)
    curve = []
    for ep in range(EPOCHS):
        for i in rng.permutation(len(ytr)):
            s = Etr[i] + b
            w = s.argmin()
            if claim[w] == ytr[i]:
                continue
            same = np.where(claim == ytr[i])[0]
            if not len(same):
                continue
            c = same[s[same].argmin()]
            b[w] += LR
            b[c] -= LR
        a_tr, _ = acc(Etr, ytr, b)
        a_te, pick = acc(Ete, yte, b)
        curve.append((a_tr, a_te))
        print(f"  epoch {ep+1:>2}  train {a_tr:.4f}   test {a_te:.4f}   "
              f"|b| max {np.abs(b[live]).max():.3f}", flush=True)

    _, pick = acc(Ete, yte, b)
    got = claim[pick]
    per_cal = {CLASSES[c]: float((got[yte == c] == c).mean()) for c in range(10)}
    cm = np.zeros((10, 10), int); np.add.at(cm, (yte, np.maximum(got, 0)), 1)

    # ceiling: how much class information is in the 40 error values at all
    from sklearn.linear_model import LogisticRegression
    F = lambda E: np.where(np.isfinite(E), E, 0.0)[:, live]
    lr = LogisticRegression(max_iter=1000).fit(F(Etr), ytr)
    ceil = float((lr.predict(F(Ete)) == yte).mean())
    print(f"\nlogistic on the {int(live.sum())} error values: {ceil:.4f}   "
          f"(ceiling for any read of the fit vector)")

    res = {"arm": ARM, "raw_fit_gate": raw_te, "reluctance_gate": curve[-1][1],
           "reluctance_gate_train": curve[-1][0], "logistic_on_errors": ceil,
           "curve": curve, "b": b.tolist(), "claim": claim.tolist(),
           "per_class_reluctance": per_cal, "epochs": EPOCHS, "lr": LR}
    (OUT / "reject_metrics.json").write_text(json.dumps(res, indent=2))

    base = json.loads((OUT / "diagnostics.json").read_text())["autopsy_competitive"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax[0].plot([c[0] for c in curve], label="train", color="#2f8f4e")
    ax[0].plot([c[1] for c in curve], label="test", color="#1b6ca8")
    ax[0].axhline(raw_te, color="#999", ls="--", label=f"raw fit gate {raw_te:.3f}")
    ax[0].axhline(ceil, color="#c1462d", ls=":", label=f"logistic on errors {ceil:.3f}")
    ax[0].set_xlabel("practice epoch"); ax[0].set_ylabel("accuracy")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.25)
    ax[0].set_title("content frozen, 40 reluctances learned", fontsize=9)

    o = np.argsort([base["per_class"][c] for c in CLASSES])
    ax[1].barh(range(10), [base["per_class"][CLASSES[c]] for c in o],
               height=.38, color="#999", label="raw fit gate")
    ax[1].barh(np.arange(10) + .4, [per_cal[CLASSES[c]] for c in o],
               height=.38, color="#2f8f4e", label="after practice")
    ax[1].set_yticks(np.arange(10) + .2)
    ax[1].set_yticklabels([CLASSES[c] for c in o], fontsize=7.5)
    ax[1].set_xlim(0, 1); ax[1].legend(fontsize=7); ax[1].grid(alpha=.25, axis="x")
    ax[1].set_title("per class", fontsize=9)

    ax[2].bar(range(H), b, color=["#2f8f4e" if l else "#ddd" for l in live])
    ax[2].set_xlabel("expert"); ax[2].set_ylabel("reluctance b")
    ax[2].set_title("who learned to keep quiet", fontsize=9)
    ax[2].grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "30_practice.png", dpi=140); plt.close(fig)
    print(f"-> {OUT}/30_practice.png")


if __name__ == "__main__":
    main()
