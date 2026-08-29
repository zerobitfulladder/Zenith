"""Two layers: does a second layer carry the wave, and what must it hear?

Layer one answers y from x, one pair at a time, and cannot see the wave
— a single pair carries no information about which way the curve is
going. Layer two is given a WINDOW of layer one's answers, so a
template is a piece of curve, and the hole is filled by completing the
window from its rims.

The window is written down four ways, which is the real question: what
should one layer send to the next?

Run:  .venv/bin/python experiments/2026_08_29/two_layer/run_two_layer.py
Env:  TL_TAG TL_K1 TL_K2 TL_WINDOWS TL_EPOCHS TL_TAPS
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "regress"))
import pop_regress as P                      # noqa: E402
import two_layer as T                        # noqa: E402
import two_layer_viz as V                    # noqa: E402

from sklearn.neural_network import MLPRegressor         # noqa: E402

OUT = HERE / "results" / os.environ.get('TL_TAG', '').lstrip("_")
K1 = int(os.environ.get("TL_K1", "1024"))
K2 = int(os.environ.get("TL_K2", "384"))
WINDOWS = int(os.environ.get("TL_WINDOWS", "12000"))
EPOCHS = int(os.environ.get("TL_EPOCHS", "25"))
TAPS = int(os.environ.get("TL_TAPS", "21"))
DELTA = 0.2
GAP = (1.2, 2.8)
NTRAIN = 1600


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_function(name, out_prefix):
    P.configure(x_range=(-12, 12), y_range=(-6, 6), gap=GAP,
                target_name=name)
    xtr, ytr, grid, sup, gapx = P.make_dataset(n_train=NTRAIN, seed=0)

    # ---- layer one --------------------------------------------------
    enc = P.Encoder(size=16384, nb=192, halfw=20, seed=0)
    t0 = time.time()
    hc = P.train(enc, xtr, ytr, k=K1, eta=0.05, epochs=EPOCHS, seed=1)
    l1_sup = rmse(P.predict(hc, enc, sup, mode="masked", read="graded",
                            topm=16)["peak"], P.target(sup))
    print(f"[{name}] layer 1: {hc.n_boot} minicolumns, {time.time()-t0:.0f}s, "
          f"in-support RMSE {l1_sup:.4f}", flush=True)

    g = T.tap_grid(DELTA)
    field = T.l1_field(hc, enc, g)
    known = T.coverage(g, xtr, DELTA)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                       max_iter=20000, random_state=0).fit(
                           xtr.reshape(-1, 1), ytr)

    # ---- layer two, four interfaces ---------------------------------
    codes = [T.ValueTaps(relative=True),
             T.ValueTaps(relative=False),
             T.ActivationTaps(hc, enc, g, onehot=True),
             T.ActivationTaps(hc, enc, g, topm=16)]
    res, rows, wins, shapes = [], [], [], None
    for code in codes:
        t0 = time.time()
        l2 = T.WindowLayer(code, k=K2, taps=TAPS, delta=DELTA, seed=0)
        T.train_l2(l2, g, field, known, n_windows=WINDOWS, seed=0)
        br = T.bridge(l2, g, field, known, GAP)
        unk = ~br["known"]
        err = rmse(br["y"][unk], P.target(br["x"][unk]))
        ref = code.reference(br["field"], br["known"])
        fit = rmse([code.decode(l2.tap_profile(l2.hc.row(br["winner"]), t),
                                ref) for t in np.nonzero(br["known"])[0]],
                   br["field"][br["known"]])
        ctrl = T.blank_control(l2, g, field, known, int(unk.sum()), seed=3)
        res.append((code.name, br, err))
        rows.append({"interface": code.name, "rmse_hole": err,
                     "fit_to_known_taps": round(fit, 3),
                     "blank_control_rmse": round(ctrl, 3),
                     "templates": int(l2.hc.n_boot),
                     "cells_per_tap": int(code.dim),
                     "seconds": round(time.time() - t0, 1)})
        print(f"[{name}] {code.name:<18} hole {err:.3f}  fit-to-known {fit:.3f}"
              f"  blank-control {ctrl:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
        # the same window, drawn in this interface's own code
        ti = T.taps_at(int(np.argmin(np.abs(g - 0.5 * sum(GAP)))), 2,
                       (TAPS - 1) // 2, TAPS)
        img = np.zeros((code.dim, TAPS), dtype=np.float32)
        r0 = code.reference(field[ti], np.ones(TAPS, bool))
        for t in range(TAPS):
            c, w = code.payload(int(ti[t]), field[ti[t]], r0)
            img[c, t] = w
        note = (f"{code.dim} cells per tap, "
                f"{int((img > 0).sum())} on in the window")
        wins.append((code.name, img, note))
        if code.name == "value-relative":
            shapes = l2

    unk = ~res[0][1]["known"]
    ux = res[0][1]["x"][unk]
    base = [{"label": "layer 1 alone (no second layer)", "kind": "base",
             "rmse_hole": rmse(res[0][1]["field"][unk], P.target(ux))},
            {"label": "hold the value at the nearer rim", "kind": "base",
             "rmse_hole": rmse(P.edge_hold(ux, GAP), P.target(ux))},
            {"label": "MLP (64-64, tanh)", "kind": "base",
             "rmse_hole": rmse(mlp.predict(ux.reshape(-1, 1)), P.target(ux))}]
    bars = base + [{"label": f"layer 2 — {r['interface']}", "kind": "l2",
                    "rmse_hole": r["rmse_hole"]} for r in rows]

    V.save_bridge(res, xtr, ytr, lambda z: mlp.predict(z.reshape(-1, 1)),
                  GAP, OUT / f"{out_prefix}_bridge.png")
    V.save_window_codes(wins, OUT / f"{out_prefix}_window_codes.png")
    V.save_interface_bars(
        sorted(bars, key=lambda r: r["rmse_hole"]),
        OUT / f"{out_prefix}_interfaces.png",
        f"What layer two can bridge depends on what layer one may say "
        f"({name.replace('_', ' ')})")
    if shapes is not None:
        V.save_shapes(shapes, OUT / f"{out_prefix}_shapes.png")

    return {"function": name, "layer1_support_rmse": round(l1_sup, 4),
            "unknown_taps": int(unk.sum()),
            "hole": list(GAP), "interfaces": rows,
            "baselines": {b["label"]: round(b["rmse_hole"], 4) for b in base}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out = {"config": {"k1": K1, "k2": K2, "taps": TAPS, "delta": DELTA,
                      "windows": WINDOWS, "n_train": NTRAIN,
                      "x_range": [-12, 12], "gap": list(GAP)},
           "runs": [run_function("wave_plus_trend", "trend"),
                    run_function("pure_wave", "periodic")]}
    out["total_seconds"] = round(time.time() - t0, 1)
    with open(OUT / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"done in {out['total_seconds']}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
