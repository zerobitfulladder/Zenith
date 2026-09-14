"""One count per pixel per digit -- and is that the ceiling?

Two things called "1x1".

(a) 1x1 windows under the rules we have been using.  A window's grid is normalised so its cells
    sum to 1; one cell means the grid is 1.0 for every digit and log 1 = 0.  Every digit scores
    zero.  Degenerate -- a grid only speaks through comparisons BETWEEN its cells.

(b) What was meant: for each pixel, count how often it is inked for each digit,
    p_c[j] = P(pixel j is inked | digit c), and read with BOTH halves:
        score(c) = sum_j [ x_j log p_c[j] + (1 - x_j) log(1 - p_c[j]) ] + log P(c)
    A blank pixel now contributes log(1 - p), which differs by digit, so absence of ink is evidence.

And the reason (b) settles the ceiling question exactly.  Rearranged,
    score(c) = sum_j x_j * [log p - log(1-p)]_c[j]   +   sum_j log(1 - p_c[j])   +   log P(c)
which is one weight picture per digit plus one number per digit -- the SAME functional form, with
the same number of free numbers, as the fitted scorecard that reads 0.9195.  So the two differ in
nothing but how the numbers are set: counted one pixel at a time, or fitted all together.

    python pixels.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import run as R

OUT = Path(__file__).parent / "results"
ALPHA = 1.0                      # one imaginary inked and one imaginary blank picture per pixel


def calib(tot, yte):
    pred = tot.argmax(1)
    z = tot - tot.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    s = np.sort(tot, 1); gap = s[:, -1] - s[:, -2]
    out = dict(acc=round(float((pred == yte).mean()), 4),
               mean_conf=round(float(p.max(1).mean()), 4),
               frac_conf_over_99=round(float((p.max(1) > 0.99).mean()), 4),
               median_gap=round(float(np.median(gap)), 1),
               frac_gap_under_5=round(float((gap < 5).mean()), 4), by_gap={})
    for a, b in [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]:
        m = (gap >= a) & (gap < b)
        if m.sum() > 20:
            out["by_gap"][f"{a}-{b if b < 1e8 else 'inf'}"] = [round(float((pred[m] == yte[m]).mean()), 3), int(m.sum())]
    return out


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = R.load()
    A = Xtr.reshape(len(Xtr), -1); B = Xte.reshape(len(Xte), -1)
    n_c = np.bincount(ytr, minlength=10).astype(np.float64)
    logprior = np.log(n_c / n_c.sum())
    res = {}

    # (a) 1x1 windows under the current rules -- degenerate by construction, measured anyway
    res["1x1_windows_current_rules"] = dict(
        acc=round(float((np.full(len(yte), int(np.argmax(n_c))) == yte).mean()), 4),
        note="every grid is 1.0, every score 0, so the read falls back on the prior alone")

    # (b) one count per pixel per digit, ink AND no-ink
    bt, bv = (A > 0.5).astype(np.float64), (B > 0.5).astype(np.float64)
    oh = np.zeros((10, len(bt))); oh[ytr, np.arange(len(bt))] = 1.0
    p = (oh @ bt + ALPHA) / (n_c[:, None] + 2 * ALPHA)                    # (10,784)
    W = (np.log(p) - np.log1p(-p)).T                                      # (784,10) the weight picture
    bias = np.log1p(-p).sum(1) + logprior                                 # one number per digit
    res["per_pixel_counted_ink_and_blank"] = calib(bv @ W + bias, yte)

    # the same, ink only (blank pixels muted) -- what the blank half is worth
    res["per_pixel_counted_ink_only"] = calib(bv @ np.log(p).T + logprior, yte)

    # (c) the ceiling of that exact form: same weights + bias, fitted together instead of counted
    m = LogisticRegression(max_iter=3000, C=0.05, n_jobs=-1).fit(bt, ytr)
    res["same_form_fitted"] = calib(m.decision_function(bv), yte)

    res["cost_of_counting_not_fitting"] = round(res["same_form_fitted"]["acc"] - res["per_pixel_counted_ink_and_blank"]["acc"], 4)
    res["blank_half_worth"] = round(res["per_pixel_counted_ink_and_blank"]["acc"] - res["per_pixel_counted_ink_only"]["acc"], 4)
    res["tiles_4x4_for_reference"] = json.load(open(OUT / "tiles.json"))["acc_ink"]
    res["whole_pattern_3x3_for_reference"] = 0.9251
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "pixels.json", "w"), indent=1)
    for k in ["per_pixel_counted_ink_and_blank", "per_pixel_counted_ink_only", "same_form_fitted"]:
        v = res[k]
        print(f"{k:<34} acc {v['acc']:.4f}  conf {v['mean_conf']:.4f}  median gap {v['median_gap']:>6.1f}  "
              f"within 5 nats {v['frac_gap_under_5']*100:.0f}%")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
