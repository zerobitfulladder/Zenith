"""Self-priced: no hand-set price. A name costs -log2 of how often it is used,
the noise level is the leftover the current explanations leave, and the two
are combined by the description-length exchange rate:

    price_t = 2 * sigma^2 * ln(1 / p_t)        sigma^2 = mean unexplained energy per pixel
                                               p_t     = usage rate of template t

Usage is a decayed count. A new hire is priced as a typical template until it
has a record (its count starts at the mean), so Huffman pricing cannot starve
newcomers. Otherwise identical to run.py: same search, same rules.

    python selfprice.py <rule> <seed>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE))
import engine as E  # noqa: E402
from run import load, support, tally_fit, N_TEST, BATCH, TALLY_N  # noqa: E402

DECAY = 0.98          # per batch, on usage counts and the noise estimate (~50 batches = 6,400 images)


MODE = "free"         # "free": price = 2 sigma^2 ln(1/p); "fixed": Huffman shape, mean price held at 0.02


class SelfPriced(E.Ownership):
    def __init__(self, rule, rng):
        super().__init__(rule, 0.02, rng)
        self.uses = np.full(E.H, 1.0, np.float32)     # decayed use counts, start uniform
        self.images = float(E.H)                      # decayed image count (so uses/images = rate)
        self.sigma2 = 0.27 / E.D                      # a first guess; replaced after the first batch
        self.reprice()

    def reprice(self):
        p = np.clip(self.uses / self.images, 1e-4, 1.0)
        bits = np.log(1.0 / p)
        if MODE == "fixed":
            live = self.n > 0
            ref = bits[live].mean() if live.any() else bits.mean()
            self.price = (self.lam * bits / ref).astype(np.float32)
        else:
            self.price = (2.0 * self.sigma2 * bits).astype(np.float32)

    def learn(self, st, X, step):
        hired_before = self.hired
        n_before = self.n.copy()
        super().learn(st, X, step)
        # usage: decay, then count who was on
        idx = st["idx"]
        on = np.bincount(idx[idx >= 0], minlength=E.H).astype(np.float32)
        self.uses = DECAY * self.uses + on
        self.images = DECAY * self.images + len(X)
        # newcomers are priced as typical until they have a record
        new = (n_before == 0) & (self.n > 0)
        if new.any():
            self.uses[new] = self.uses[self.n > 0].mean()
        # noise level: the leftover per pixel the current explanations leave
        U = float(((X - st["xhat"]) ** 2).sum(1).mean())
        self.sigma2 = DECAY * self.sigma2 + (1 - DECAY) * U / E.D if step > 0 else U / E.D
        self.reprice()


def main():
    rule, seed = sys.argv[1], int(sys.argv[2])
    global MODE
    MODE = sys.argv[3] if len(sys.argv) > 3 else "free"
    tag = f"selfprice_{rule}_s{seed}" if MODE == "free" else f"selfprice-fixed_{rule}_s{seed}"
    Xtr, ytr, Xte, yte = load(seed)
    net = SelfPriced(rule, np.random.default_rng(seed))
    t0 = time.time()
    traj = []

    def log(step, st):
        live = net.n > 0
        names = np.where(st["idx"] >= 0, net.price[np.maximum(st["idx"], 0)], 0.0).sum(1)
        traj.append(dict(step=step, on=float(st["count"].mean()), unexplained=float((st["cost"] - names).mean()),
                         price_mean=float(net.price[live].mean()), price_min=float(net.price[live].min()),
                         price_max=float(net.price[live].max()), sigma2=float(net.sigma2)))
        print(f"  {tag} step {step:3d}  on/img {st['count'].mean():.2f}  price mean {net.price[live].mean():.4f} "
              f"[{net.price[live].min():.4f}, {net.price[live].max():.4f}]  sigma2*784 {net.sigma2*E.D:.3f}  hired {net.hired}", flush=True)

    net.fit(Xtr, np.random.default_rng(seed + 100), 1, BATCH, log=log)
    t_train = time.time() - t0
    Cte, ste = net.codes(Xte)
    Ctr, _ = net.codes(Xtr[:TALLY_N])
    live = net.n > 0
    supp, _ = support(net.Wp)
    F_te, F_tr = Cte > 0, Ctr > 0
    L = tally_fit(F_tr, ytr[:TALLY_N])
    tally = float(((F_te.astype(np.float64) @ L).argmax(1) == yte).mean())
    names = np.where(ste["idx"] >= 0, net.price[np.maximum(ste["idx"], 0)], 0.0).sum(1)
    bits_names = np.where(ste["idx"] >= 0, np.log2(1.0 / np.clip(net.uses / net.images, 1e-4, 1))[np.maximum(ste["idx"], 0)], 0.0).sum(1)
    unex = ste["cost"] - names
    res = dict(rule=rule, seed=seed, selfpriced=True, t_train=t_train, traj=traj,
               live=int(live.sum()), hired=int(net.hired), dead=float(1 - live.mean()),
               on_per_image=float(ste["count"].mean()), on_max=int(ste["count"].max()),
               unexplained=float(unex.mean()), cost=float(ste["cost"].mean()),
               name_bits_per_image=float(bits_names.mean()),
               price_mean=float(net.price[live].mean()), price_min=float(net.price[live].min()),
               price_max=float(net.price[live].max()), sigma2=float(net.sigma2),
               support_median=float(np.median(supp[live])), support=[int(k) for k in supp[live]],
               wholes=int((supp[live] > 60).sum()), strokes=int(((supp[live] > 20) & (supp[live] <= 60)).sum()),
               dots=int((supp[live] <= 20).sum()), tally=tally,
               on_hist=np.bincount(ste["count"], minlength=E.S + 1).tolist(),
               train_cap_hits=net.cap_hits)
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(res, f)
    if seed == 0:
        np.savez(OUT / f"weights_{tag}.npz", W=net.W, Wp=net.Wp, n=net.n, fires=F_te.sum(0), price=net.price,
                 uses=net.uses / net.images)
    print(f"{tag}: train {t_train:.0f}s | on/img {res['on_per_image']:.2f}  unexplained {res['unexplained']:.3f}  "
          f"price mean {res['price_mean']:.4f} [{res['price_min']:.4f},{res['price_max']:.4f}]  name bits/img {res['name_bits_per_image']:.1f}  "
          f"support {res['support_median']:.0f}px  wholes/strokes/dots {res['wholes']}/{res['strokes']}/{res['dots']}  tally {tally:.3f}", flush=True)


if __name__ == "__main__":
    main()
