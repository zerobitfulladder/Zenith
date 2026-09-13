"""Search in place of backprop, same MLP shape. Second pass: a beam over the
order, and the label as a second stream into the top layer only.

784 -> 256 -> 128 -> 64, ReLU, three hidden layers, same random init, same data
order, same two passes over 12k MNIST images. All layers learn online at once.

  backprop    softmax head, cross-entropy, Adam. The label shapes every layer.
  search-b1   sequential explaining, greedy: pick the unit with the largest
              projection onto what is LEFT, keep the projection as its
              activation (max(0, w.r), one unit at a time on the residual),
              subtract, pick again, stop when the next pick would explain less
              than STOP of the input's energy. Only chosen units learn, toward
              the residual they were shown, by a running count. Units that never
              win, or stop winning, are hired from the largest unexplained
              residual. The label never touches a weight.
  search-b4   the same with a beam of 4 over the ORDER of picks: each partial
              explanation expands into its 4 best next picks, the 4 best totals
              survive, two orders reaching the same set are merged, the best
              total at the end is the explanation and the only one that learns.
  label-b4    search-b4, and the top layer's input is [layer-2 code ; LAM * one-hot
              label]. The label is one more thing the top templates explain. At
              read time the label is absent: templates are scored by their norm
              over the code part only, and the label is read from the label
              part of the templates that fired.

Per layer, per arm, on held-out images: tally over which units fired, linear
probe on the graded code, units on per input, dead units, selectivity
max_c P(c | unit fires). Plus backprop's head and the label-stream read.

    python run.py <arm> <seed>      writes results/<arm>_s<seed>.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

N_TRAIN, N_TEST = 12000, 2000
WIDTHS = [256, 128, 64]
KMAX = [16, 12, 8]        # most picks a layer may make per input
STOP = 0.02               # a pick must explain this fraction of the input's energy
HIRE = 0.5                # a residual still holding this much energy hires a free template
HIRE_MAX = 8              # per batch
DUP = 0.7                 # cosine above this to an earlier hire this batch = duplicate, skip
N_MAX = 200               # count window: after this many wins a template tracks at 1/N_MAX
STALE = 60                # batches without a win -> the template is free to be re-hired
LAM = 0.7                 # label stream weight: label energy 0.49 against code energy 1
BATCH, EPOCHS = 128, 2
PROBE_N = 6000


# ---------------------------------------------------------------- data ----
def load(seed):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32)
    X = X.reshape(len(X), -1)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(int)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def unit(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-8)


# ------------------------------------------------------- the search net ----
class SearchLayer:
    def __init__(self, d, h, kmax, beam, rng, d_code=None):
        self.M = (rng.normal(size=(h, d)) / np.sqrt(d)).astype(np.float32)   # same draw as the MLP
        self.W = unit(self.M)
        self.n = np.zeros(h, np.float32)          # wins so far
        self.last = np.full(h, -10**9)            # batch index of the last win
        self.h, self.d, self.kmax, self.beam = h, d, kmax, beam
        self.dc = d if d_code is None else d_code  # the dims present at read time
        self.hv = np.random.default_rng(12345).random(h)   # set signature, for merging orders
        self.hired = 0

    def read_W(self, full):
        """Partial cue: score a template by its norm over the queried cells."""
        if full or self.dc == self.d:
            return self.W
        Wr = np.zeros_like(self.W)
        Wr[:, :self.dc] = unit(self.W[:, :self.dc])
        return Wr

    def read(self, X, full=True, keep=False):
        """Beam over the order of picks. Returns the best explanation's code,
        its residual, and (if keep) what each chosen template was shown."""
        W = self.read_W(full)
        B, d, h, K = len(X), self.d, self.h, self.beam
        R = np.repeat(X[:, None, :], K, 1).astype(np.float32)
        tot = np.full((B, K), -np.inf)
        tot[:, 0] = 0.0
        used = np.zeros((B, K, h), bool)
        C = np.zeros((B, K, h), np.float32)
        alive = np.zeros((B, K), bool)
        alive[:, 0] = True
        valid = alive.copy()
        Ht = np.full((B, K, self.kmax), -1, int)
        Hr = np.zeros((B, K, self.kmax, d), np.float32) if keep else None
        bi = np.arange(B)[:, None]
        parent_all = np.concatenate([np.repeat(np.arange(K), K), np.arange(K)])[None].repeat(B, 0)
        for step in range(self.kmax):
            if not alive.any():
                break
            s = (R.reshape(B * K, d) @ W.T).reshape(B, K, h)
            s[used] = -np.inf
            s[~alive] = -np.inf
            top = np.argpartition(-s, K - 1, axis=2)[:, :, :K]          # K best next picks per slot
            a = np.take_along_axis(s, top, 2)
            g = np.where(np.isfinite(a), a * a, 0.0)
            ok = (a > 0) & (g > STOP)
            cand = np.where(ok, tot[:, :, None] + g, -np.inf)             # expansions
            stay = np.where(valid & ~ok.any(2), tot, -np.inf)              # finished slots persist
            all_tot = np.concatenate([cand.reshape(B, -1), stay], 1)
            all_t = np.concatenate([top.reshape(B, -1), np.full((B, K), -1)], 1)
            all_a = np.concatenate([a.reshape(B, -1), np.zeros((B, K), np.float32)], 1)
            # merge two orders that reach the same set: keep the better total
            sig = np.take_along_axis((used * self.hv).sum(2), parent_all, 1)
            sig = sig + np.where(all_t >= 0, self.hv[np.maximum(all_t, 0)], 0.0)
            sig[all_tot == -np.inf] = np.inf
            order = np.lexsort((-all_tot, sig), axis=1)
            sig_s = np.take_along_axis(sig, order, 1)
            tot_s = np.take_along_axis(all_tot, order, 1)
            dup = np.zeros_like(tot_s, bool)
            dup[:, 1:] = sig_s[:, 1:] == sig_s[:, :-1]
            tot_s[dup] = -np.inf
            pick = np.argsort(-tot_s, axis=1)[:, :K]
            chosen = np.take_along_axis(order, pick, 1)
            tot = np.take_along_axis(tot_s, pick, 1)
            par = np.take_along_axis(parent_all, chosen, 1)
            t = np.take_along_axis(all_t, chosen, 1)
            av = np.take_along_axis(all_a, chosen, 1)
            valid = tot > -np.inf
            expand = valid & (t >= 0)
            R, used, C, Ht = R[bi, par], used[bi, par], C[bi, par], Ht[bi, par]
            if keep:
                Hr = Hr[bi, par]
            b_idx, k_idx = np.nonzero(expand)
            tt, aa = t[b_idx, k_idx], av[b_idx, k_idx]
            if keep:
                Hr[b_idx, k_idx, step] = R[b_idx, k_idx]
            Ht[b_idx, k_idx, step] = tt
            C[b_idx, k_idx, tt] = aa
            used[b_idx, k_idx, tt] = True
            R[b_idx, k_idx] -= aa[:, None] * W[tt]
            alive = expand
        best = tot.argmax(1)
        ar = np.arange(B)
        shown = []
        if keep:
            Htb, Hrb = Ht[ar, best], Hr[ar, best]
            for step in range(self.kmax):
                m = Htb[:, step] >= 0
                if not m.any():
                    break
                shown.append((Htb[m, step], Hrb[m, step]))
        return C[ar, best], R[ar, best], shown

    def learn(self, shown, R, step):
        h, d = self.h, self.d
        for tt, r in shown:
            S = np.zeros((h, d), np.float32)
            np.add.at(S, tt, r)
            m = np.bincount(tt, minlength=h).astype(np.float32)
            p = m > 0
            mean_r = S[p] / m[p, None]
            self.n[p] += m[p]
            self.last[p] = step
            eta = np.minimum(1.0, m[p] / np.minimum(self.n[p], N_MAX))     # a running count
            self.M[p] += eta[:, None] * (mean_r - self.M[p])
        free = np.flatnonzero((self.n == 0) | (step - self.last > STALE))
        if len(free):
            e = (R ** 2).sum(1)
            cand = np.argsort(-e)
            cand = cand[e[cand] > HIRE]
            taken = []
            for i in cand:
                if len(taken) >= min(len(free), HIRE_MAX):
                    break
                r = R[i]
                if taken and (unit(r) @ unit(np.stack(taken)).T).max() > DUP:
                    continue
                taken.append(r)
            for k, r in enumerate(taken):
                t = free[k]
                self.M[t], self.n[t], self.last[t] = r, 1, step
                self.hired += 1
        self.W = unit(self.M)


class SearchNet:
    def __init__(self, rng, beam, label_top=False):
        dims = [784] + WIDTHS
        self.label_top = label_top
        self.layers = []
        for i in range(3):
            d = dims[i] + (10 if (label_top and i == 2) else 0)
            self.layers.append(SearchLayer(d, dims[i + 1], KMAX[i], beam, rng, d_code=dims[i]))

    def forward(self, X, y=None, learn=False, step=0):
        inp = unit(X)
        codes = []
        for i, L in enumerate(self.layers):
            top = self.label_top and i == 2
            if top:
                lab = np.zeros((len(inp), 10), np.float32)
                if y is not None:
                    lab[np.arange(len(y)), y] = LAM
                inp = np.concatenate([inp, lab], 1)
            C, R, shown = L.read(inp, full=(not top) or (y is not None), keep=learn)
            if learn:
                L.learn(shown, R, step)
            codes.append(C)
            inp = unit(C)
        return codes

    def train(self, X, y, rng):
        step = 0
        for _ in range(EPOCHS):
            order = rng.permutation(len(X))
            for b in range(0, len(X), BATCH):
                idx = order[b:b + BATCH]
                self.forward(X[idx], y[idx], learn=True, step=step)
                step += 1
        return self

    def codes(self, X):
        out = [[] for _ in WIDTHS]
        for b in range(0, len(X), 256):
            for i, C in enumerate(self.forward(X[b:b + 256])):
                out[i].append(C)
        return [np.concatenate(c) for c in out]

    def label_read(self, C3):
        L = self.layers[2]
        return (C3 @ L.W[:, L.dc:]).argmax(1)


# ---------------------------------------------------------- the MLP ----
class MLP:
    def __init__(self, rng, lr=1e-3):
        dims = [784] + WIDTHS + [10]
        self.W = [(rng.normal(size=(dims[i + 1], dims[i])) * np.sqrt(2.0 / dims[i])).astype(np.float32)
                  for i in range(4)]
        self.b = [np.zeros(dims[i + 1], np.float32) for i in range(4)]
        self.lr = lr
        self.m = [np.zeros_like(w) for w in self.W + self.b]
        self.v = [np.zeros_like(w) for w in self.W + self.b]
        self.t = 0

    def forward(self, X):
        H, a = [], X
        for i in range(3):
            a = np.maximum(0.0, a @ self.W[i].T + self.b[i])
            H.append(a)
        return H, a @ self.W[3].T + self.b[3]

    def step(self, X, y):
        H, logits = self.forward(X)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1.0
        g = p / len(y)
        acts = [X] + H
        gW, gb = [None] * 4, [None] * 4
        for i in range(3, -1, -1):
            gW[i] = g.T @ acts[i]
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i]) * (acts[i] > 0)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, (P, G) in enumerate(zip(self.W + self.b, gW + gb)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * G
            self.v[k] = b2 * self.v[k] + (1 - b2) * G * G
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            P -= self.lr * mh / (np.sqrt(vh) + eps)

    def train(self, X, y, rng):
        for _ in range(EPOCHS):
            order = rng.permutation(len(X))
            for b in range(0, len(X), BATCH):
                idx = order[b:b + BATCH]
                self.step(unit(X[idx]), y[idx])
        return self

    def codes(self, X):
        H, _ = self.forward(unit(X))
        return H

    def head(self, X):
        _, logits = self.forward(unit(X))
        return logits.argmax(1)


# ------------------------------------------------------------ readouts ----
def tally_fit(F, y, alpha=1.0):
    """log P(y | unit fired) / P(y), one row per unit. Absolute values, no rank."""
    N = np.stack([F[y == c].sum(0) for c in range(10)], 1).astype(np.float64)
    py = np.bincount(y, minlength=10) / len(y)
    P = (N + alpha) / (N.sum(1, keepdims=True) + 10 * alpha)
    return np.log(P / py)


def tally_acc(Ftr, ytr, Fte, yte):
    L = tally_fit(Ftr, ytr)
    return float(((Fte.astype(np.float64) @ L).argmax(1) == yte).mean())


def probe_acc(Atr, ytr, Ate, yte):
    if Atr[:PROBE_N].std() == 0:
        return 0.1
    clf = LogisticRegression(max_iter=500)
    clf.fit(Atr[:PROBE_N], ytr[:PROBE_N])
    return float(clf.score(Ate, yte))


def identity(A, y):
    F = A > 0
    fires = F.sum(0)
    live = fires > 0
    out = dict(active=float(F.mean()), dead=float(1 - live.mean()), units=int(A.shape[1]))
    if live.sum() == 0:
        out.update(sel_mean=0.1, sel_gt50=0.0, kurt_median=0.0, sel=[])
        return out
    Nc = np.stack([F[y == c].sum(0) for c in range(10)], 1)[live]
    sel = Nc.max(1) / fires[live]
    Al = A[:, live]
    z = (Al - Al.mean(0)) / (Al.std(0) + 1e-8)
    kurt = (z ** 4).mean(0) - 3
    out.update(sel_mean=float(sel.mean()), sel_gt50=float((sel > 0.5).mean()),
               kurt_median=float(np.median(kurt)), sel=[float(s) for s in sel])
    return out


# ----------------------------------------------------------------- run ----
def run_arm(arm, seed, Xtr, ytr, Xte, yte):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if arm == "backprop":
        net = MLP(rng).train(Xtr, ytr, np.random.default_rng(seed + 100))
    else:
        beam = int(arm.split("-b")[1])
        net = SearchNet(rng, beam, label_top=arm.startswith("label"))
        net.train(Xtr, ytr, np.random.default_rng(seed + 100))
    t_train = time.time() - t0
    Ctr, Cte = net.codes(Xtr), net.codes(Xte)
    res = dict(arm=arm, seed=seed, t_train=t_train, layers=[])
    if arm == "backprop":
        res["head"] = float((net.head(Xte) == yte).mean())
    else:
        res["hired"] = [int(L.hired) for L in net.layers]
        if net.label_top:
            res["label_read"] = float((net.label_read(Cte[2]) == yte).mean())
    F_all_tr = np.concatenate([c > 0 for c in Ctr], 1)
    F_all_te = np.concatenate([c > 0 for c in Cte], 1)
    res["tally_all"] = tally_acc(F_all_tr, ytr, F_all_te, yte)
    for i in range(3):
        row = identity(Cte[i], yte)
        row["tally"] = tally_acc(Ctr[i] > 0, ytr, Cte[i] > 0, yte)
        row["probe"] = probe_acc(Ctr[i], ytr, Cte[i], yte)
        row["picks"] = float((Cte[i] > 0).sum(1).mean())
        res["layers"].append(row)
    if seed == 0:
        Ws = [L.W for L in net.layers] if arm != "backprop" else net.W[:3]
        # what each unit sees: the activation-weighted mean of the test images it fired on
        FM = [(c.T @ Xte) / np.maximum(c.sum(0), 1e-8)[:, None] for c in Cte]
        fires = [(c > 0).sum(0) for c in Cte]
        np.savez(OUT / f"weights_{arm}.npz", W1=Ws[0], W2=Ws[1], W3=Ws[2], FM1=FM[0], FM2=FM[1], FM3=FM[2],
                 N1=fires[0], N2=fires[1], N3=fires[2])
    with open(OUT / f"{arm}_s{seed}.json", "w") as f:
        json.dump(res, f)
    print(f"{arm:10s} seed {seed}  train {t_train:5.1f}s  " +
          "  ".join(f"L{i+1} tally {r['tally']:.3f} probe {r['probe']:.3f} on {r['picks']:.1f} "
                    f"dead {r['dead']:.2f} sel {r['sel_mean']:.2f}"
                    for i, r in enumerate(res["layers"])) +
          (f"  head {res['head']:.3f}" if "head" in res else f"  hired {res['hired']}") +
          (f"  label-read {res['label_read']:.3f}" if "label_read" in res else "") +
          f"  tally-all {res['tally_all']:.3f}", flush=True)
    return res


if __name__ == "__main__":
    arm, seed = sys.argv[1], int(sys.argv[2])
    Xtr, ytr, Xte, yte = load(seed)
    run_arm(arm, seed, Xtr, ytr, Xte, yte)
