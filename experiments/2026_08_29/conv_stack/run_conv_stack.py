"""The stack the architecture actually wants.

  level 0   the raw pairs, read through a small window
  layer 1   ONE shared hypercolumn over a 5-sample patch (0.8 wide),
            each patch written relative to its own mean, applied at
            every x. Its templates are a vocabulary of local shapes
            with no location attached, so the same shape anywhere
            produces the same code.
  layer 2   a window of layer one's CODES. Because the vocabulary is
            shared, two windows overlap by content, not by place.
  decode    completed patches say how their samples differ from one
            another; one least-squares solve turns all of those
            differences into absolute values at once.

Run:  .venv/bin/python experiments/2026_08_29/conv_stack/run_conv_stack.py
Env:  CS_K1 CS_K2 CS_TAPS CS_WINDOWS CS_TAG
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
sys.path.insert(0, str(HERE.parent / "two_layer"))
import pop_regress as P                      # noqa: E402
import two_layer as T                        # noqa: E402
import conv_stack as C                       # noqa: E402
import conv_viz as V                         # noqa: E402

from sklearn.neural_network import MLPRegressor        # noqa: E402

OUT = HERE / "results" / os.environ.get('CS_TAG', '').lstrip("_")
K1 = int(os.environ.get("CS_K1", "64"))
K2 = int(os.environ.get("CS_K2", "96"))
TAPS2 = int(os.environ.get("CS_TAPS", "51"))
WINDOWS = int(os.environ.get("CS_WINDOWS", "12000"))
GAP = (1.2, 2.8)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P.configure(x_range=(-12, 12), y_range=(-6, 6), gap=GAP,
                target_name="wave_plus_trend")
    xtr, ytr, *_ = P.make_dataset(n_train=1600, seed=0)
    g = T.tap_grid(0.2)
    val, known = C.sample_raw(g, xtr, ytr)
    miss = np.nonzero(~known)[0]
    print(f"level 0: {known.sum()} of {len(g)} grid points have data; "
          f"{len(miss)} missing (x {g[miss[0]]:.1f}..{g[miss[-1]]:.1f})",
          flush=True)

    # ---- layer one: shared shape vocabulary -------------------------
    l1 = C.PatchLayer(k=K1, seed=0)
    l1.train(g, val, known, n=20000, seed=0)
    ok = np.zeros(len(g), bool)
    ok[l1.valid(g, known)] = True
    codes = [l1.code(g, val, i, topm=8) if ok[i] else None
             for i in range(len(g))]
    codes = [(c[0], c[1]) if c is not None else None for c in codes]
    blank = np.nonzero(~ok)[0]
    print(f"layer 1: {l1.hc.n_boot} shared templates over a "
          f"{l1.offsets(g)[-1]-l1.offsets(g)[0]:.1f}-wide patch; "
          f"codes exist at {ok.sum()} positions, absent at {len(blank)} "
          f"(x {g[blank[0]]:.1f}..{g[blank[-1]]:.1f})", flush=True)

    # ---- layer two: windows of those codes --------------------------
    l2 = C.CodeWindowLayer(dim=l1.hc.n_boot, taps=TAPS2, step=1, k=K2,
                           size=8192, seed=0)
    anchors = l2.train(codes, ok, len(g), n=WINDOWS, seed=0)
    print(f"layer 2: {l2.hc.n_boot} templates over a "
          f"{(TAPS2-1)*0.1:.1f}-wide window of codes, "
          f"{len(anchors)} training positions", flush=True)

    centre = int(np.argmin(np.abs(g - 0.5*(GAP[0]+GAP[1]))))
    filled_ids, l2_winner, score = l2.complete(codes, ok, centre)
    print(f"        completed {len(filled_ids)} missing codes from "
          f"template #{l2_winner} (match {score:.3f})", flush=True)

    shapes = {p: l1.template_patch(j) for p, j in filled_ids.items()}
    sol = C.stitch(shapes, l1, len(g),
                   anchor_vals=val[known], anchor_idx=np.nonzero(known)[0])
    pred = val.copy()
    for q, v in sol.items():
        if not known[q]:
            pred[q] = v

    # ---- score, against everything else -----------------------------
    truth = P.target(g[miss])
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                       max_iter=20000, random_state=0).fit(
                           xtr.reshape(-1, 1), ytr)
    l1w = C.PatchLayer(k=K1, seed=0)
    RM = np.array([True]*4 + [False])
    l1w.train(g, val, known, n=20000, seed=0, ref_mask=RM)
    walk, _ = l1w.walk(g, val, known, ref_mask=RM)

    rows = [("conv L1 + L2 over its codes", rmse(pred[miss], truth)),
            ("conv L1 alone, walking", rmse(walk[miss], truth)),
            ("MLP (64-64, tanh)", rmse(mlp.predict(g[miss].reshape(-1,1)), truth)),
            ("hold the value at the nearer rim",
             rmse(P.edge_hold(g[miss], GAP), truth)),
            ("old place-bound layer 1 alone", 0.817)]
    print()
    for n, v in rows:
        print(f"  {n:<34} {v:.3f}", flush=True)

    # ---- pictures ---------------------------------------------------
    V.save_vocabulary(l1, OUT / "l1_vocabulary.png")
    V.save_code_field(l1, g, codes, ok, GAP, OUT / "l1_code_field.png")
    V.save_l2_templates(l2, l1, g, OUT / "l2_templates.png")
    V.save_regeneration(g, val, known, pred, walk, xtr, ytr, GAP,
                        lambda z: mlp.predict(z.reshape(-1, 1)),
                        OUT / "regeneration.png")
    V.save_completed_codes(l2, l1, g, codes, ok, filled_ids, l2_winner,
                           centre, OUT / "completed_codes.png")

    metrics = {"config": {"k1": K1, "k2": K2, "l1_taps": 5,
                          "l1_patch_width": 0.8, "l2_taps": TAPS2,
                          "l2_window_width": (TAPS2-1)*0.1,
                          "windows": WINDOWS, "gap": list(GAP)},
               "level0": {"grid": len(g), "with_data": int(known.sum()),
                          "missing": int(len(miss))},
               "layer1": {"templates": int(l1.hc.n_boot),
                          "codes_present": int(ok.sum()),
                          "codes_absent": int(len(blank))},
               "layer2": {"templates": int(l2.hc.n_boot),
                          "training_positions": int(len(anchors)),
                          "winner": l2_winner, "match": round(score, 3),
                          "codes_completed": len(filled_ids)},
               "gap_rmse": {n: round(v, 4) for n, v in rows},
               "per_point": [{"x": round(float(g[q]), 1),
                              "predicted": round(float(pred[q]), 3),
                              "truth": round(float(P.target(g[q])), 3)}
                             for q in miss],
               "seconds": round(time.time() - t0, 1)}
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\ndone in {metrics['seconds']}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
