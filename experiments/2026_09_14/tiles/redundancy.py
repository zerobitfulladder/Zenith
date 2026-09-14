"""Is the gap between counting and fitting caused by pixels repeating each other?

Each pixel is read exactly once -- nothing is added twice.  The claim is about the ASSUMPTION:
the model takes the pixels to be independent once the digit is known, so it treats a pixel and its
neighbour as two separate witnesses when they are near-copies of one.

Test: thin the picture out.  Keep every pixel, then every 2nd, 3rd, 4th -- the survivors sit further
and further apart, so they repeat each other less.  Both models are then built on exactly the same
survivors:
  * counted: P(where ink lands | digit) over the kept pixels, filled by counting, read by Bayes
  * fitted:  the same scorecard form, weights fitted together instead of counted
If the claim is right, the gap must shrink as the pixels spread apart.
Also reported: how correlated neighbouring survivors actually are once the digit is known.

    python redundancy.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import run as R

OUT = Path(__file__).parent / "results"
FLOOR = 0.002


def neighbour_corr(X, y, step):
    """mean correlation between a kept pixel and the next kept one along, WITHIN a digit."""
    cs = []
    for c in range(10):
        Z = X[y == c][:, ::step, ::step].reshape((y == c).sum(), -1)
        Z = Z.reshape(len(Z), 28 // step if 28 % step == 0 else Z.shape[1], -1) if False else Z
        w = int(np.ceil(28 / step))
        Z = Z.reshape(len(Z), w, w)
        a, b = Z[:, :, :-1].reshape(len(Z), -1), Z[:, :, 1:].reshape(len(Z), -1)
        a = a - a.mean(0); b = b - b.mean(0)
        num = (a * b).sum(0); den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
        keep = den > 1e-6
        cs.append(float((num[keep] / den[keep]).mean()))
    return float(np.mean(cs))


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = R.load()
    n_c = np.bincount(ytr, minlength=10).astype(np.float64)
    logprior = np.log(n_c / n_c.sum())
    rows = []
    for step in [1, 2, 3, 4]:
        A = Xtr[:, ::step, ::step].reshape(len(Xtr), -1).astype(np.float64)
        B = Xte[:, ::step, ::step].reshape(len(Xte), -1).astype(np.float64)
        d = A.shape[1]

        oh = np.zeros((10, len(A))); oh[ytr, np.arange(len(A))] = 1.0
        M = oh @ A + n_c[:, None] * FLOOR
        logT = np.log(M / M.sum(1, keepdims=True))
        counted = float(((B @ logT.T + logprior).argmax(1) == yte).mean())

        m = LogisticRegression(max_iter=3000, C=0.05, n_jobs=-1).fit(A, ytr)
        fitted = float((m.predict(B) == yte).mean())

        cc = neighbour_corr(Xtr, ytr, step)
        rows.append(dict(step=step, pixels_kept=int(d), counted=round(counted, 4), fitted=round(fitted, 4),
                         gap=round(fitted - counted, 4), neighbour_corr_within_digit=round(cc, 3)))
        print(f"every {step} pixel  ({d:>3} kept)  neighbour corr {cc:.3f}  |  counted {counted:.4f}  "
              f"fitted {fitted:.4f}  |  gap {fitted-counted:+.4f}")

    res = dict(rows=rows, seconds=round(time.time() - t0, 1))
    json.dump(res, open(OUT / "redundancy.json", "w"), indent=1)


if __name__ == "__main__":
    main()
