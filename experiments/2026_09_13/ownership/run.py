"""One arm, one seed:  python run.py <rule> <price> <seed>
Writes results/<rule>_l<price>_s<seed>.json and, for seed 0, weights_<rule>_l<price>.npz."""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import engine as E  # noqa: E402

N_TRAIN, N_TEST = 12000, 2000
BATCH, EPOCHS = 128, 1
TALLY_N = 6000          # train images read for the tally (reading is the expensive part)


def load(seed):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32).reshape(-1, 784)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(int)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return E.unit(X[:N_TRAIN]), y[:N_TRAIN], E.unit(X[N_TRAIN:N_TRAIN + N_TEST]), y[N_TRAIN:N_TRAIN + N_TEST]


def support(Wp, frac=0.9):
    """Pixels carrying `frac` of a template's (positive) energy, and the mask of them."""
    w2 = Wp ** 2
    o = np.argsort(-w2, 1)
    cs = np.cumsum(np.take_along_axis(w2, o, 1), 1) / np.maximum(w2.sum(1, keepdims=True), 1e-12)
    k = (cs < frac).sum(1) + 1
    mask = np.zeros_like(Wp, bool)
    for t in range(len(Wp)):
        mask[t, o[t, :k[t]]] = True
    return k, mask


def tally_fit(F, y, alpha=1.0):
    N = np.stack([F[y == c].sum(0) for c in range(10)], 1).astype(np.float64)
    py = np.bincount(y, minlength=10) / len(y)
    return np.log(((N + alpha) / (N.sum(1, keepdims=True) + 10 * alpha)) / py)


def main():
    rule, lam, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    tag = f"{rule}_l{lam:g}_s{seed}"
    if rule == "owned-fit":
        rule, E.LEARN_ASSIGN = "owned", "fit"
    Xtr, ytr, Xte, yte = load(seed)
    net = E.Ownership(rule, lam, np.random.default_rng(seed))
    t0 = time.time()
    net.fit(Xtr, np.random.default_rng(seed + 100), EPOCHS, BATCH,
            log=lambda step, st: print(f"  {tag} step {step:3d}  on/img {st['count'].mean():.2f}  "
                                       f"cost {st['cost'].mean():.3f}  hired {net.hired}", flush=True))
    t_train = time.time() - t0
    train_rounds, train_caps = float(np.mean(net.rounds)), net.cap_hits
    net.rounds, net.cap_hits = [], 0
    Cte, ste = net.codes(Xte)
    Ctr, _ = net.codes(Xtr[:TALLY_N])
    live = net.n > 0
    supp, smask = support(net.Wp)
    # identity of the templates on together: overlap of their supports, per image
    ov = []
    for i in range(len(Xte)):
        on = ste["idx"][i][ste["idx"][i] >= 0]
        if len(on) < 2:
            continue
        m = smask[on].astype(np.float32)
        inter = m @ m.T
        uni = m.sum(1)[:, None] + m.sum(1)[None, :] - inter
        iu = np.triu_indices(len(on), 1)
        ov.append(float((inter[iu] / np.maximum(uni[iu], 1)).mean()))
    F_te, F_tr = Cte > 0, Ctr > 0
    L = tally_fit(F_tr, ytr[:TALLY_N])
    tally = float(((F_te.astype(np.float64) @ L).argmax(1) == yte).mean())
    fires = F_te.sum(0)
    Nc = np.stack([F_te[yte == c].sum(0) for c in range(10)], 1)
    sel = Nc[fires > 0].max(1) / fires[fires > 0]
    res = dict(rule=rule, lam=lam, seed=seed, t_train=t_train,
               train_rounds=train_rounds, train_cap_hits=train_caps,
               test_rounds=float(np.mean(net.rounds)), test_cap_hits=net.cap_hits,
               hired=int(net.hired), live=int(live.sum()), dead=float(1 - live.mean()),
               on_per_image=float(ste["count"].mean()), on_max=int(ste["count"].max()),
               slot_cap_hits=int((ste["count"] >= E.S).sum()),
               unexplained=float((ste["cost"] - lam * ste["count"]).mean()),
               cost=float(ste["cost"].mean()),
               support_median=float(np.median(supp[live])), support_mean=float(supp[live].mean()),
               support=[int(k) for k in supp[live]],
               overlap=float(np.mean(ov)) if ov else 0.0,
               tally=tally, sel_mean=float(sel.mean()), sel_gt50=float((sel > 0.5).mean()),
               on_hist=np.bincount(ste["count"], minlength=E.S + 1).tolist())
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(res, f)
    if seed == 0:
        k = 8
        np.savez(OUT / f"weights_{tag.rsplit('_s', 1)[0]}.npz", W=net.W, Wp=net.Wp, n=net.n, fires=fires,
                 X=Xte[:k], y=yte[:k], idx=ste["idx"][:k], owner=ste["owner"][:k], xhat=ste["xhat"][:k],
                 cost=ste["cost"][:k])
    print(f"{tag}: train {t_train:.0f}s  rounds {train_rounds:.1f} (cap hits {train_caps})  live {live.sum()}  "
          f"hired {net.hired} | test on/img {res['on_per_image']:.2f} (max {res['on_max']})  unexplained {res['unexplained']:.3f}  "
          f"support median {res['support_median']:.0f}px  overlap {res['overlap']:.3f}  tally {tally:.3f}  sel {res['sel_mean']:.2f}",
          flush=True)


if __name__ == "__main__":
    main()
