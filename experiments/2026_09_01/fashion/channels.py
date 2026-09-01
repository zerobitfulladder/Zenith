"""Fashion is material: three brightness channels beside L1, plus the factor
the binned table drops.

The champion Fashion number (0.8447, magnitude.py binned4) comes from a pipeline
that centres every patch, L2-normalises it, and drops the flat ones. Each of
those steps discards brightness -- and the pixel-counting control says brightness
is worth +9.5 points on Fashion (2 bins 0.6377 -> 16 bins 0.7330) while it
HURTS on MNIST. So: keep the L1 champion untouched and add counted evidence
channels beside it, at small validated weights (the shouting law).

    flat     brightness of the DROPPED patches, binned, per cell
             "flat and bright" (fabric fill) vs "flat and dark" (background)
    int      brightness of EVERY patch (its raw mean), binned, per cell
    magbin   the factor binned-N throws away: T[t,cell,MAGBIN,class] conditions
             the template on the magnitude bin but never counts the bin itself
             as evidence; N[mbin, cell, class] is that missing term
    silh7/4  the whole image average-pooled to 7x7 / 4x4, each coarse pixel
             binned by brightness, counted per position -- the global scale
             nothing in the stack looks at

Protocol: base = L1 binned table (nb swept). Channel weights and nb chosen on a
2,000-image carve of the train split; tables rebuilt on the full 12,000 before
the test set is touched; binned4 with seed 7 reproduced as the anchor. Fashion
runs 3 seeds (GPU atomics are worth 0.3-0.4 points); MNIST runs 1 seed as the
asymmetry control -- the prediction is that every one of these channels is
neutral-to-negative on ink.
"""

import json, os, sys, time
from pathlib import Path
import numpy as np
import cupy as cp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experts"))
sys.path.insert(0, str(HERE.parent / "stack"))
import experts as E
import gpu_stack as G

OUT = HERE / "results"
SMOKE = "--smoke" in sys.argv
K1, NL, ALPHA = G.K1, 10, 1.0
SIDE, GRID = G.SIDE, G.GRID1
P = SIDE * SIDE
NB_SWEEP = [1, 3] if SMOKE else [1, 2, 3, 4, 6, 8, 12, 16]   # nb=1 = pure count
if SMOKE:
    G.EPOCHS1 = 1
LAMS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
NB_FLAT, NB_INT, NB_SILH = 4, 6, 8
N_VAL = 2000
DATASETS = [("fashion_mnist", [7] if SMOKE else [7, 8, 9]),
            ("mnist", [7])]


def table_of(N, K):
    """Identical to magnitude.py: log P(event | ..., class) / P(event | ...)."""
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * K)
    pm = ((N.sum(-1, keepdims=True) + ALPHA * NL)
          / (N.sum((0, -1), keepdims=True) + ALPHA * K * NL))
    return (cp.log(pc) - cp.log(pm)).astype(cp.float32)


def mag_edges(live, nb):
    if nb == 1:
        return cp.zeros(0, cp.float32)
    q = cp.asarray(np.linspace(0, 100, nb + 1)[1:-1])
    return cp.percentile(live, q)


def bright_edges(vals, nb):
    """Bin 0 is 'off' (< 0.02); the rest are quantiles of the positive mass."""
    lo = cp.asarray([0.02], cp.float32)
    if nb == 2:
        return lo
    pos = vals[vals > 0.02]
    q = cp.asarray(np.linspace(0, 100, nb)[1:-1])
    return cp.concatenate([lo, cp.percentile(pos, q).astype(cp.float32)])


def bins_of(vals, edges, nb):
    return cp.clip(cp.searchsorted(edges, vals.ravel()).reshape(vals.shape),
                   0, nb - 1).astype(cp.int32)


def count_chan(bins, valid, cells_row, y, nb, ncell):
    """N[bin, cell, class] from one event per valid position."""
    n, p = bins.shape
    C = cp.tile(cells_row, n).reshape(n, p)
    lab = cp.repeat(y, p).reshape(n, p)
    flat = (bins[valid] * ncell + C[valid]) * NL + lab[valid]
    N = cp.bincount(flat, minlength=nb * ncell * NL)
    return N.reshape(nb, ncell, NL).astype(cp.float64)


def ev_chan(T, bins, valid, cells_row):
    n, p = bins.shape
    C = cp.tile(cells_row, n).reshape(n, p)
    return (T[bins, C] * valid[..., None]).sum(1)


def count_l1(idx, keep, mbins, y, nb):
    n = len(idx)
    C = cp.tile(G.C1, n).reshape(n, P)
    lab = cp.repeat(y, P).reshape(n, P)
    flat = (((idx[keep] * (GRID * GRID) + C[keep]) * nb + mbins[keep]) * NL
            + lab[keep])
    N = cp.bincount(flat, minlength=K1 * GRID * GRID * nb * NL)
    return N.reshape(K1, GRID * GRID, nb, NL).astype(cp.float64)


def ev_l1(T, idx, keep, mbins):
    n = len(idx)
    C = cp.tile(G.C1, n).reshape(n, P)
    return (T[idx, C, mbins] * keep[..., None]).sum(1)


def brightness(X, chunk=1024):
    """Raw mean of every 5x5 patch -- computed BEFORE centring throws it away."""
    out = cp.zeros((len(X), P), cp.float32)
    for a in range(0, len(X), chunk):
        out[a:a + chunk] = X[a:a + chunk][:, G.PIDX].mean(-1)
    return out


def silhouette(X, g):
    b = 28 // g
    return X.reshape(len(X), g, b, g, b).mean(axis=(2, 4)).reshape(len(X), g * g)


def acc(sc, y):
    return float((sc.argmax(1) == y).mean())


def build_all(codes, y, nb_l1, edges=None):
    """codes = (idx, mag, keep, bright, s7, s4). Returns (tables, edges)."""
    idx, mag, keep, bright, s7, s4 = codes
    if edges is None:
        edges = {
            "mag": mag_edges(mag[keep], nb_l1),
            "flat": bright_edges(bright[~keep], NB_FLAT),
            "int": bright_edges(bright, NB_INT),
            "s7": bright_edges(s7, NB_SILH),
            "s4": bright_edges(s4, NB_SILH),
        }
    mb = bins_of(mag, edges["mag"], nb_l1)
    ones7 = cp.ones(s7.shape, bool)
    ones4 = cp.ones(s4.shape, bool)
    onesP = cp.ones(bright.shape, bool)
    cells7 = cp.arange(49)
    cells4 = cp.arange(16)
    T = {
        "l1": table_of(count_l1(idx, keep, mb, y, nb_l1), K1),
        "flat": table_of(count_chan(bins_of(bright, edges["flat"], NB_FLAT),
                                    ~keep, G.C1, y, NB_FLAT, GRID * GRID), NB_FLAT),
        "int": table_of(count_chan(bins_of(bright, edges["int"], NB_INT),
                                   onesP, G.C1, y, NB_INT, GRID * GRID), NB_INT),
        "magbin": table_of(count_chan(mb, keep, G.C1, y, nb_l1, GRID * GRID), nb_l1),
        "silh7": table_of(count_chan(bins_of(s7, edges["s7"], NB_SILH),
                                     ones7, cells7, y, NB_SILH, 49), NB_SILH),
        "silh4": table_of(count_chan(bins_of(s4, edges["s4"], NB_SILH),
                                     ones4, cells4, y, NB_SILH, 16), NB_SILH),
    }
    return T, edges


def all_ev(T, edges, codes):
    idx, mag, keep, bright, s7, s4 = codes
    mb = bins_of(mag, edges["mag"], T["magbin"].shape[0])
    ones7 = cp.ones(s7.shape, bool)
    ones4 = cp.ones(s4.shape, bool)
    onesP = cp.ones(bright.shape, bool)
    return {
        "l1": ev_l1(T["l1"], idx, keep, mb),
        "flat": ev_chan(T["flat"], bins_of(bright, edges["flat"], NB_FLAT),
                        ~keep, G.C1),
        "int": ev_chan(T["int"], bins_of(bright, edges["int"], NB_INT),
                       onesP, G.C1),
        "magbin": ev_chan(T["magbin"], mb, keep, G.C1),
        "silh7": ev_chan(T["silh7"], bins_of(s7, edges["s7"], NB_SILH),
                         ones7, cp.arange(49)),
        "silh4": ev_chan(T["silh4"], bins_of(s4, edges["s4"], NB_SILH),
                         ones4, cp.arange(16)),
    }


CHANS = ["flat", "int", "magbin", "silh7", "silh4"]


def best(cand):
    """Highest accuracy; ties go to the SMALLEST weight, so a channel with no
    validation gain is dropped rather than kept at an arbitrary large lambda."""
    return max(cand, key=lambda t: (t[0], -t[1]))


def combo_score(ev, lams):
    sc = ev["l1"].copy()
    for c, l in lams.items():
        if l > 0:
            sc += l * ev[c]
    return sc


def select_config(codes_tr, ytr, log):
    """Choose nb and channel weights on a train/val carve. Test never touched."""
    n = len(ytr)
    cut = n - N_VAL
    fit = tuple(a[:cut] for a in codes_tr)
    val = tuple(a[cut:] for a in codes_tr)
    yfit, yval = ytr[:cut], ytr[cut:]

    pick = (None, -1.0)
    for nb in NB_SWEEP:
        T, edges = build_all(fit, yfit, nb)
        a = acc(all_ev(T, edges, val)["l1"], yval)
        log(f"    val  nb={nb:<3d} base {a:.4f}")
        if a > pick[1]:
            pick = (nb, a)
    nb = pick[0]

    T, edges = build_all(fit, yfit, nb)
    ev = all_ev(T, edges, val)
    base = acc(ev["l1"], yval)
    log(f"    val  chosen nb={nb}  base {base:.4f}")

    solo = {}
    for c in CHANS:
        a, l = best([(acc(ev["l1"] + l * ev[c], yval), l) for l in LAMS])
        solo[c] = (l, a - base)
        log(f"    val  +{c:<7s} lam {l:<5.2f} {a:.4f}  ({a - base:+.4f})")

    lams = {c: 0.0 for c in CHANS}
    cur = base
    for c in sorted(CHANS, key=lambda c: -solo[c][1]):
        a, l = best([(acc(combo_score(ev, {**lams, c: l}), yval), l)
                     for l in LAMS])
        if a > cur:
            lams[c], cur = l, a
    for c in CHANS:                       # one coordinate round, 0 allowed
        a, l = best([(acc(combo_score(ev, {**lams, c: l}), yval), l)
                     for l in LAMS])
        if a >= cur:
            lams[c], cur = l, a
    log(f"    val  combo {cur:.4f}  lams " +
        " ".join(f"{c}={lams[c]:.2f}" for c in CHANS if lams[c] > 0))
    return nb, lams, {"nb": nb, "val_base": base, "val_combo": cur,
                      "val_solo": {c: solo[c] for c in CHANS}}


def run_dataset(DS, seeds, log):
    Xtr, ytr, Xte, yte = E.load(DS)
    if SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:4000], ytr[:4000], Xte[:800], yte[:800]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)

    config, per_seed = None, []
    for s in seeds:
        t0 = time.time()
        W1, _ = G.l1_train(Xtr, ytr_g, np.random.default_rng(s))
        itr, mtr, ktr = G.l1_code(W1, Xtr)
        ite, mte, kte = G.l1_code(W1, Xte)
        codes_tr = (itr, mtr, ktr, brightness(Xtr), silhouette(Xtr, 7),
                    silhouette(Xtr, 4))
        codes_te = (ite, mte, kte, brightness(Xte), silhouette(Xte, 7),
                    silhouette(Xte, 4))
        if s == seeds[0]:
            frac = float(ktr.mean())
            log(f"  {DS} seed {s}: live patches {frac:.3f} "
                f"({time.time() - t0:.0f}s to code)")
            nb, lams, val_info = select_config(codes_tr, ytr_g, log)
            config = (nb, lams, val_info)
        nb, lams, _ = config

        row = {"seed": s}
        T4, e4 = build_all(codes_tr, ytr_g, 4)          # the anchor
        row["binned4"] = acc(all_ev(T4, e4, codes_te)["l1"], yte_g)
        T, edges = build_all(codes_tr, ytr_g, nb)
        ev = all_ev(T, edges, codes_te)
        row["base"] = acc(ev["l1"], yte_g)
        for c in CHANS:
            l = lams[c] if lams[c] > 0 else config[2]["val_solo"][c][0]
            row[f"+{c}"] = acc(ev["l1"] + l * ev[c], yte_g)
        row["combo"] = acc(combo_score(ev, lams), yte_g)
        if s == seeds[0]:
            row["test_sweep"] = {c: {str(l): acc(ev["l1"] + l * ev[c], yte_g)
                                     for l in LAMS} for c in CHANS}
        per_seed.append(row)
        log(f"  seed {s}:  binned4 {row['binned4']:.4f}   base(nb={nb}) "
            f"{row['base']:.4f}   combo {row['combo']:.4f}   "
            f"({time.time() - t0:.0f}s)")
        for c in CHANS:
            log(f"           +{c:<7s} {row[f'+{c}']:.4f}")

    agg = {}
    for k, v0 in per_seed[0].items():
        if not isinstance(v0, float):
            continue
        v = [r[k] for r in per_seed]
        agg[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    return {"dataset": DS, "seeds": seeds, "config": {
                "nb": config[0], "lams": config[1], **config[2]},
            "per_seed": per_seed, "agg": agg}


def main():
    OUT.mkdir(exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    results = {}
    for DS, seeds in DATASETS:
        log(f"\n=== {DS} ===")
        results[DS] = run_dataset(DS, seeds, log)
        a = results[DS]["agg"]
        log(f"\n  {DS} over {len(seeds)} seed(s):  "
            f"binned4 {a['binned4']['mean']:.4f}+-{a['binned4']['std']:.4f}   "
            f"combo {a['combo']['mean']:.4f}+-{a['combo']['std']:.4f}")

    tag = "_smoke" if SMOKE else ""
    (OUT / f"channels{tag}.json").write_text(json.dumps(results, indent=2))
    (OUT / f"channels{tag}.log").write_text("\n".join(lines))
    log(f"\nwrote {OUT / f'channels{tag}.json'}")


if __name__ == "__main__":
    main()
