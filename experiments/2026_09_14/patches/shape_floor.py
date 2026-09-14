"""Does the floor let a BLANK patch speak?  It cannot under the ink read (the floor's contribution
there is one constant per digit, the same for every picture).  Under the shape read each patch is
divided by its own ink total, so a floored blank patch becomes a flat patch with full voting weight,
and a flat patch should match whichever digit leaves that window blank.  Tested directly:

  * shape read, floor E in training and reading
  * the same, with every genuinely-blank patch MUTED (its contribution forced to zero)
If blank patches are really saying something useful, muting them must cost accuracy.

    python shape_floor.py [tag]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

import run as R

HERE = Path(__file__).parent
OUT = HERE / "results"
EPS = [0.0, 0.0005, 0.002, 0.01, 0.05]
BATCH = 1000


def run_eps(Xte, yte, M0, n_c, logprior, E):
    M = M0 + n_c[:, None, None] * E
    T = M + (1e-3 if E == 0 else 0.0)
    T = T / T.sum(2, keepdims=True)
    logT = np.log(T)
    lt = np.ascontiguousarray(logT.transpose(1, 2, 0))             # (576,25,10)

    tot = np.zeros((len(Xte), R.C)); tot_mute = np.zeros((len(Xte), R.C))
    blank_frac = 0.0
    for b in range(0, len(Xte), BATCH):
        Xw = R.windows(Xte[b:b + BATCH]).astype(np.float64)        # (n,576,25)
        real = Xw.sum(2)                                            # ink actually in the patch
        Xf = Xw + E
        s = Xf.sum(2)
        Xn = Xf / np.maximum(s, 1e-12)[:, :, None]                  # every patch now sums to 1
        sc = np.matmul(np.ascontiguousarray(Xn.transpose(1, 0, 2)), lt).transpose(1, 0, 2)
        blank = real < 1e-6
        if E == 0.0:
            sc = np.where(blank[:, :, None], 0.0, sc)                # with no floor a blank patch has nothing to divide by
        blank_frac += blank.mean() * len(Xw)
        tot[b:b + BATCH] = sc.sum(1)
        tot_mute[b:b + BATCH] = np.where(blank[:, :, None], 0.0, sc).sum(1)
    tot += logprior; tot_mute += logprior
    p = tot.argmax(1)
    return (float((p == yte).mean()),
            float((tot_mute.argmax(1) == yte).mean()),
            blank_frac / len(Xte),
            np.bincount(p, minlength=R.C) / len(p))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "tables"
    t0 = time.time()
    M0 = np.load(OUT / f"{tag}.npz")["mass_raw"].astype(np.float64)
    Xtr, ytr, Xte, yte = R.load()
    n_c = np.bincount(ytr, minlength=R.C).astype(np.float64)
    logprior = np.log(n_c / n_c.sum())

    rows = []
    for E in EPS:
        a, am, bf, ph = run_eps(Xte, yte, M0, n_c, logprior, E)
        rows.append(dict(eps=E, acc=round(a, 4), acc_blank_muted=round(am, 4),
                         blank_patches_per_image=round(float(bf), 3), gain_from_blanks=round(a - am, 4),
                         guess_share=[round(float(v), 3) for v in ph], guessed_1_share=round(float(ph[1]), 3)))
        print(f"E={E:<7} shape read: acc {a:.4f}  |  blanks muted {am:.4f}  |  blanks {bf*100:.0f}% of patches  "
              f"|  blanks worth {a-am:+.4f}  |  guessed '1' on {ph[1]*100:.0f}% of images (true 11%)")
    res = dict(tag=tag, read="shape (each patch divided by its own ink total)",
               n_test=len(Xte), seconds=round(time.time() - t0, 2), rows=rows)
    json.dump(res, open(OUT / f"{tag}_shapefloor.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
