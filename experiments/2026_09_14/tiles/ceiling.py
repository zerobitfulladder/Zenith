"""How much of the gap is the independence assumption, and how much is the shape?

Our read collapses to one scorecard picture per digit: score(c) = sum over pixels of x * W_c[pixel].
So the BEST POSSIBLE scorecard of that form is the hard ceiling of anything this read can reach --
no model built on per-pixel evidence can ever beat it.  Our scorecard is COUNTED (each pixel
estimated on its own, then multiplied).  Fitting the same scorecard directly instead measures
exactly what the independence assumption costs.

Three numbers on the same split:
  1. counted scorecard      -- what we built
  2. best-fit scorecard     -- the ceiling of this read's FORM
  3. same, on the binarised picture -- sanity, in case greyscale is doing the work

    python ceiling.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import run as R

OUT = Path(__file__).parent / "results"


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = R.load()
    A, B = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    res = {}

    # 1. what we built
    res["counted_scorecard"] = json.load(open(OUT / "tiles.json"))["acc_ink"]

    # 2. the ceiling of the form
    for name, tr, te in [("best_fit_scorecard", A, B),
                         ("best_fit_scorecard_binarised", (A > 0.5).astype(np.float32), (B > 0.5).astype(np.float32))]:
        m = LogisticRegression(max_iter=2000, C=0.05, n_jobs=-1)
        m.fit(tr, ytr)
        p = m.predict_proba(te)
        pred = p.argmax(1)
        s = np.sort(np.log(np.maximum(p, 1e-30)), 1)
        res[name] = round(float((pred == yte).mean()), 4)
        res[name + "_mean_conf"] = round(float(p.max(1).mean()), 4)
        res[name + "_median_gap"] = round(float(np.median(s[:, -1] - s[:, -2])), 1)
        print(f"{name}: {res[name]:.4f}  (mean top guess {res[name+'_mean_conf']:.3f}, median gap {res[name+'_median_gap']:.1f})")

    res["cost_of_counting_independently"] = round(res["best_fit_scorecard"] - res["counted_scorecard"], 4)
    res["whole_pattern_3x3_for_reference"] = 0.9251
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT / "ceiling.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
