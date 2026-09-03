"""Position as a dial: pool the winners into g x g cells, g = 1 (OR) .. 6 (champion).

Same rig, trained per grid (the belief bias reads the table, so training sees
the grid too), read with the per-cell table, test set shifted 0-3 px.

Usage:  uv run python grid_sweep.py
"""
import json, sys
from pathlib import Path
import numpy as np
import cupy as cp
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "plasticity_rf"))
import rf_sweep as R
from or_readout import shift, SHIFTS

SIZES = [5, 9]
GRIDS = [1, 2, 3, 4, 6]


def main():
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    res = {}
    print(f"{'size':>5}{'grid':>6}{'cells':>7}" + "".join(f"{f'shift {s[0]}':>10}" for s in SHIFTS))
    for ps in SIZES:
        for g in GRIDS:
            R.GRID = g
            rig = R.Rig(ps)
            _, W = R.run(rig, True, Xtr, ytr_g, Xte, yte_g, 7)
            W = cp.asarray(W)
            itr, ktr = rig.code(W, Xtr)
            T = rig.table_from(rig.build_table(itr, ytr_g, ktr))
            row = [rig.accuracy(T, rig.code(W, shift(Xte, dx, dy)), yte_g) for dx, dy in SHIFTS]
            res[f"{ps}|{g}"] = row
            print(f"{ps:>5}{g:>6}{rig.gg:>7}" + "".join(f"{a:>10.4f}" for a in row), flush=True)
    (HERE / "results" / "grid_sweep.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
