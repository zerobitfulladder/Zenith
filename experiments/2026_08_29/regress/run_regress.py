"""Population-coded regression — the run.

Trains one hypercolumn on OR'ed (x, y) codes, asks it for y at x values
it has never seen — including a whole slice of x cut out of training —
compares it against three off-the-shelf regressors, and then probes the
three things this rig exists to measure:

  * how much overlap the code needs (bump-width sweep),
  * whether more minicolumns help (selection-rule sweep),
  * how wide a hole it can bridge (gap sweep).

Run:  .venv/bin/python experiments/2026_08_29/regress/run_regress.py
Env:  PR_K PR_EPOCHS PR_HALFW PR_NB PR_SIZE PR_ETA PR_N PR_TAG PR_QUICK
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pop_regress as P                      # noqa: E402
import regress_viz as V                      # noqa: E402

from sklearn.ensemble import RandomForestRegressor      # noqa: E402
from sklearn.neighbors import KNeighborsRegressor       # noqa: E402
from sklearn.neural_network import MLPRegressor         # noqa: E402

OUT = HERE / "results" / os.environ.get('PR_TAG', '').lstrip("_")
K = int(os.environ.get("PR_K", "256"))
EPOCHS = int(os.environ.get("PR_EPOCHS", "40"))
HALFW = int(os.environ.get("PR_HALFW", "10"))
NB = int(os.environ.get("PR_NB", "96"))
SIZE = int(os.environ.get("PR_SIZE", "8192"))
ETA = float(os.environ.get("PR_ETA", "0.05"))
NTRAIN = int(os.environ.get("PR_N", "800"))
QUICK = os.environ.get("PR_QUICK", "0") == "1"
K_BEST = 1024


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def score(name, probe, xs, yhat):
    return {"probe": probe, "model": name,
            "rmse": round(rmse(yhat, P.target(xs)), 4),
            "mae": round(float(np.mean(np.abs(yhat - P.target(xs)))), 4)}


def baselines(xtr, ytr):
    return {"k-nearest neighbours (k=5)": KNeighborsRegressor(n_neighbors=5),
            "random forest (300 trees)": RandomForestRegressor(
                n_estimators=300, random_state=0),
            "MLP (64-64, tanh)": MLPRegressor(
                hidden_layer_sizes=(64, 64), activation="tanh",
                max_iter=20000, random_state=0)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    xtr, ytr, grid, sup, gap = P.make_dataset(n_train=NTRAIN, seed=0)
    Xtr = xtr.reshape(-1, 1)

    # ---- the rig exactly as specified -------------------------------
    enc = P.Encoder(size=SIZE, nb=NB, halfw=HALFW, seed=0)
    t0 = time.time()
    hc = P.train(enc, xtr, ytr, k=K, eta=ETA, epochs=EPOCHS, seed=1)
    train_s = time.time() - t0
    g = P.predict(hc, enc, grid)
    ps, pg = P.predict(hc, enc, sup), P.predict(hc, enc, gap)
    print(f"trained {hc.n_boot} minicolumns in {train_s:.1f}s", flush=True)

    table = [score("hypercolumn — top-1 cell", "unseen x in range", sup,
                   ps["peak"]),
             score("hypercolumn — sub-cell read", "unseen x in range", sup,
                   ps["centroid"])]
    base_grid, fitted = {}, {}
    for name, m in baselines(xtr, ytr).items():
        m.fit(Xtr, ytr)
        fitted[name] = m
        base_grid[name] = m.predict(grid.reshape(-1, 1))
        table.append(score(name, "unseen x in range", sup,
                           m.predict(sup.reshape(-1, 1))))
    table += [score("hypercolumn — top-1 cell", "held-out slice", gap,
                    pg["peak"]),
              score("hypercolumn — sub-cell read", "held-out slice", gap,
                    pg["centroid"])]
    for name, m in fitted.items():
        table.append(score(name, "held-out slice", gap,
                           m.predict(gap.reshape(-1, 1))))
    table.append(score("(hold the value at the nearer rim)", "held-out slice",
                       gap, P.edge_hold(gap)))

    pairs = [(float(xtr[i]), float(ytr[i]))
             for i in np.linspace(20, len(xtr) - 20, 4).astype(int)]
    V.save_dataset(xtr, ytr, grid, OUT / "dataset.png")
    V.save_encoding(enc, pairs, OUT / "encoding.png")
    V.save_templates(hc, enc, OUT / "templates.png")
    V.save_template_map(hc, enc, xtr, ytr, grid, OUT / "template_map.png")
    V.save_predictions(grid, g, base_grid, xtr, ytr, OUT / "predictions.png")
    V.save_readout_field(grid, g, enc, OUT / "readout_field.png")

    metrics = {
        "config": {"k": K, "epochs": EPOCHS, "halfw_cells": HALFW,
                   "cells_per_channel": NB, "array_size": SIZE, "eta": ETA,
                   "n_train": NTRAIN, "gap": list(P.GAP),
                   "selection": "dot (the drone's rule)", "read": "top-1",
                   "train_seconds": round(train_s, 1)},
        "layer": {"minicolumns_adopted": int(hc.n_boot),
                  "distinct_winners_on_grid": int(len(np.unique(g["winner"]))),
                  "mean_winning_bid_on_grid": round(float(np.mean(g["match"])), 3),
                  "active_cells_per_code": int(
                      len(enc.encode_sparse(0.0, 0.0)[0])),
                  "code_density": round(
                      len(enc.encode_sparse(0.0, 0.0)[0]) / SIZE, 4)},
        "headline": table,
    }
    for r in table:
        print(f"  {r['probe']:<20} {r['model']:<36} rmse {r['rmse']:.4f}",
              flush=True)

    # ---- position collisions: what the disjoint scatter is worth ----
    e_col = P.Encoder(size=SIZE, nb=NB, halfw=HALFW, seed=0, disjoint=False)
    h_col = P.train(e_col, xtr, ytr, k=K, eta=ETA, epochs=EPOCHS, seed=1)
    p_col = P.predict(h_col, e_col, sup)
    n_col = len(np.intersect1d(e_col.pos["x"], e_col.pos["y"]))
    metrics["collisions"] = {
        "shared_positions": int(n_col),
        "rmse_with_collisions": round(rmse(p_col["peak"], P.target(sup)), 4),
        "rmse_disjoint": round(rmse(ps["peak"], P.target(sup)), 4)}
    print(f"collisions: {n_col} shared positions -> rmse "
          f"{metrics['collisions']['rmse_with_collisions']} vs "
          f"{metrics['collisions']['rmse_disjoint']} disjoint", flush=True)

    # ---- selection rule x capacity ----------------------------------
    sel_rows, best = [], None
    for k in (128, 256, 512, 1024):
        h = P.train(enc, xtr, ytr, k=k, eta=ETA, epochs=EPOCHS, seed=1)
        for mode in ("dot", "masked"):
            s_ = P.predict(h, enc, sup, mode=mode)
            gr = P.predict(h, enc, grid, mode=mode)
            sel_rows.append({"k": k, "mode": mode,
                             "rmse_support": round(rmse(s_["peak"],
                                                        P.target(sup)), 4),
                             "distinct_winners":
                                 int(len(np.unique(gr["winner"])))})
            print(f"  k={k:<5} {mode:<7} rmse {sel_rows[-1]['rmse_support']:.4f}"
                  f"  winners {sel_rows[-1]['distinct_winners']}", flush=True)
        if k == K_BEST:
            best = h
    V.save_selection(sel_rows, OUT / "selection_rule.png")
    metrics["selection_sweep"] = sel_rows

    # ---- read rule, on the best-selecting layer ---------------------
    read_rows = []
    for name, kw in (("top-1 cell (as specified)", dict(read="top1")),
                     ("graded, top-4 matches", dict(read="graded", topm=4)),
                     ("graded, top-16 matches", dict(read="graded", topm=16)),
                     ("graded, top-64 matches", dict(read="graded", topm=64)),
                     ("graded, every positive match",
                      dict(read="graded", topm=0))):
        s_ = P.predict(best, enc, sup, mode="masked", **kw)
        g_ = P.predict(best, enc, gap, mode="masked", **kw)
        read_rows.append({"read": name,
                          "rmse_support": round(rmse(s_["peak"],
                                                     P.target(sup)), 4),
                          "rmse_support_subcell": round(
                              rmse(s_["centroid"], P.target(sup)), 4),
                          "rmse_gap": round(rmse(g_["peak"],
                                                 P.target(gap)), 4)})
        print(f"  {name:<32} sup {read_rows[-1]['rmse_support']:.4f}  "
              f"gap {read_rows[-1]['rmse_gap']:.4f}", flush=True)
    metrics["read_rules"] = read_rows

    gb = P.predict(best, enc, grid, mode="masked", read="graded", topm=16)
    V.save_predictions(grid, gb, base_grid, xtr, ytr,
                       OUT / "predictions_best.png",
                       left_title=f"Same rig, three fixes: {K_BEST} "
                                  f"minicolumns, norm-corrected selection, "
                                  f"graded read")
    V.save_readout_field(grid, gb, enc, OUT / "readout_field_best.png",
                         note="  (graded read: every positive match "
                              "contributes, magnitudes intact)")

    # ---- how much overlap does the code need? -----------------------
    width_rows = []
    for hw in (1, 2, 4, 8, 12, 16, 24, 32, 48):
        e2 = P.Encoder(size=SIZE, nb=NB, halfw=hw, seed=0)
        h2 = P.train(e2, xtr, ytr, k=K, eta=ETA, epochs=EPOCHS, seed=1)
        s2, g2 = P.predict(h2, e2, sup), P.predict(h2, e2, gap)
        gr = P.predict(h2, e2, grid)
        width_rows.append({"halfw": hw,
                           "rmse_support": round(rmse(s2["peak"],
                                                      P.target(sup)), 4),
                           "rmse_gap": round(rmse(g2["peak"],
                                                  P.target(gap)), 4),
                           "distinct_winners":
                               int(len(np.unique(gr["winner"])))})
        print(f"  halfw {hw:3d}  sup {width_rows[-1]['rmse_support']:.4f}"
              f"  gap {width_rows[-1]['rmse_gap']:.4f}", flush=True)
    # the same sweep on a SPARSE training set, where an unseen x really
    # has no example next to it — this is where overlap has to earn its
    # keep, and with 800 dense pairs it never gets asked to.
    xs_, ys_, _, sup_, _ = P.make_dataset(n_train=60, seed=0)
    sparse_rows = []
    for hw in (1, 2, 4, 8, 12, 16, 24, 32, 48):
        e3 = P.Encoder(size=SIZE, nb=NB, halfw=hw, seed=0)
        h4 = P.train(e3, xs_, ys_, k=64, eta=ETA, epochs=EPOCHS, seed=1)
        s4 = P.predict(h4, e3, sup_)
        sparse_rows.append({"halfw": hw,
                            "rmse_support": round(rmse(s4["peak"],
                                                       P.target(sup_)), 4)})
        print(f"  sparse halfw {hw:3d}  sup "
              f"{sparse_rows[-1]['rmse_support']:.4f}", flush=True)
    V.save_width_sweep(width_rows, sparse_rows, OUT / "width_sweep.png")
    metrics["width_sweep"] = width_rows
    metrics["width_sweep_sparse_60_pairs"] = sparse_rows

    # ---- how wide a hole can it bridge? -----------------------------
    gap_rows = []
    for w in (0.2, 0.4, 0.8, 1.2, 1.6, 2.2, 3.0):
        gp = (2.0 - w / 2, 2.0 + w / 2)
        xa, ya, _, _, gx = P.make_dataset(n_train=NTRAIN, seed=0, gap=gp)
        h3 = P.train(enc, xa, ya, k=K, eta=ETA, epochs=EPOCHS, seed=1)
        pm = P.predict(h3, enc, gx)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh",
                           max_iter=20000, random_state=0).fit(
                               xa.reshape(-1, 1), ya)
        gap_rows.append({"width": w,
                         "rmse_model": round(rmse(pm["peak"],
                                                  P.target(gx)), 4),
                         "rmse_edge": round(rmse(P.edge_hold(gx, gp),
                                                 P.target(gx)), 4),
                         "rmse_mlp": round(rmse(mlp.predict(gx.reshape(-1, 1)),
                                                P.target(gx)), 4)})
        print(f"  hole {w:.1f}  model {gap_rows[-1]['rmse_model']:.3f}"
              f"  rim-hold {gap_rows[-1]['rmse_edge']:.3f}"
              f"  mlp {gap_rows[-1]['rmse_mlp']:.3f}", flush=True)
    V.save_gap_sweep(gap_rows, OUT / "gap_sweep.png")
    metrics["gap_sweep"] = gap_rows

    metrics["total_seconds"] = round(time.time() - t_all, 1)
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(OUT / "weights.npz", W=hc.W, wins=hc.wins, n_boot=hc.n_boot)
    print(f"done in {metrics['total_seconds']}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
