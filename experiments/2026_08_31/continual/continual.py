"""Split-MNIST: learn 0-4, then learn 5-9, and see what survives.

Experts are hired, not pre-allocated. Competition is on FIT of the joined
[image ; label] vector, so an expert can only claim inputs it explains --
and a digit it has never seen has an orthogonal label block, so nothing
explains it, so a new expert gets hired. Existing experts are only ever
updated by inputs they win, so learning 5-9 cannot overwrite 0-4.

    err = fit of every existing expert to the joined vector
    if min(err) > theta:  buffer it; hire once the buffer holds 2K samples
    else:                 the winner learns

At test time the label is blank, so the gate scores fit on the image half
alone and reads the winner's label block.

Baselines are trained on exactly the same two-phase stream with partial_fit.
"""

import json, time, warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
EPS = 1e-12
K, RHO, ETA = 36, 1.0, 0.5
THETAS = [0.20, 0.25, 0.30, 0.35]
MAX_EXPERTS = 200
N_TRAIN, N_TEST, EPOCHS, BATCH = 20000, 5000, 3, 128
SEED = 0
A, B = list(range(5)), list(range(5, 10))


def center_norm(V):
    Vc = V - V.mean(axis=1, keepdims=True)
    return Vc / np.maximum(np.linalg.norm(Vc, axis=1, keepdims=True), EPS)


def load():
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    X = X.reshape(len(X), -1)
    if X.max() > 1.5:
        X = X / 255.0
    p = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[p], y[p]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def join(X, y=None, n_lab=10):
    V = np.zeros((len(X), X.shape[1] + n_lab))
    V[:, :X.shape[1]] = X
    if y is not None:
        L = np.zeros((len(X), n_lab)); L[np.arange(len(X)), y] = 1.0
        g = RHO * np.linalg.norm(X, axis=1) / np.maximum(np.linalg.norm(L, axis=1), EPS)
        V[:, X.shape[1]:] = L * g[:, None]
    return center_norm(V)


def geo_step(Wh, Bt, eta):
    S = Bt @ Wh.T
    E = Bt - S @ Wh
    M = (S.T @ E) / len(Bt)
    tau = M - (M * Wh).sum(1, keepdims=True) * Wh
    tn = np.linalg.norm(tau, axis=1)
    th = np.clip(eta * tn, 0.0, np.pi / 4)
    hat = np.zeros_like(tau); live = tn > EPS
    hat[live] = tau[live] / tn[live, None]
    Wh = Wh * np.cos(th)[:, None] + hat * np.sin(th)[:, None]
    return Wh / (np.linalg.norm(Wh, axis=1, keepdims=True) + EPS)


class Hiring:
    """Experts appear when nothing explains the input."""

    def __init__(self, dim, theta):
        self.W = np.zeros((0, K, dim)); self.dim, self.theta = dim, theta
        self.buf = []; self.tag = []                    # tag = digit each claimed
        self.wins = np.zeros((0, 10), dtype=np.int64)
        self.hired_at = []                              # which phase hired it
        self.trace = []                                 # expert count over the stream
        self.phase = 1

    def _err(self, J, half=None):
        if len(self.W) == 0:
            return np.ones((0, len(J)))
        h, k, d = self.W.shape
        S = (J @ self.W.reshape(h * k, d).T).reshape(len(J), h, k).transpose(1, 0, 2)
        R = np.matmul(S, self.W)
        sl = slice(None) if half is None else slice(0, half)
        return (np.linalg.norm(J[None, :, sl] - R[:, :, sl], axis=2) /
                np.maximum(np.linalg.norm(J[None, :, sl], axis=2), EPS))

    def _hire(self):
        Bf = np.array(self.buf)
        Vt = np.linalg.svd(Bf, full_matrices=False)[2][:K]
        if len(Vt) < K:
            Vt = np.vstack([Vt, np.random.default_rng(len(self.W)).standard_normal(
                (K - len(Vt), self.dim))])
        Vt = Vt / (np.linalg.norm(Vt, axis=1, keepdims=True) + EPS)
        self.W = np.concatenate([self.W, Vt[None]], 0)
        self.wins = np.concatenate([self.wins, np.zeros((1, 10), np.int64)], 0)
        self.hired_at.append(self.phase)
        self.buf = []

    def learn(self, J, y):
        e = self._err(J)
        if len(self.W) == 0:
            best = np.full(len(J), -1); mn = np.full(len(J), np.inf)
        else:
            best = e.argmin(0); mn = e[best, np.arange(len(J))]
        for i in range(len(J)):
            if mn[i] > self.theta and len(self.W) < MAX_EXPERTS:
                self.buf.append(J[i]); self.tag.append(y[i])
                if len(self.buf) >= 2 * K:
                    self._hire()
            if best[i] >= 0:
                self.wins[best[i], y[i]] += 1
        for h in range(len(self.W)):
            m = (best == h)                 # the winner learns regardless of theta:
            if m.sum() >= 2:                # best-available and good are different
                self.W[h] = geo_step(self.W[h], J[m], ETA)
        self.trace.append(len(self.W))

    def assign(self, X, n_img):
        e = self._err(join(X), half=n_img)
        return e.argmin(0)

    def predict(self, X, n_img):
        Q = join(X)
        e = self._err(Q, half=n_img)
        pick = e.argmin(0)
        lab = self.wins.argmax(1)                       # each expert's claimed digit
        return lab[pick]


def score(pred, y, mask):
    return float((pred[mask] == y[mask]).mean()) if mask.sum() else float("nan")


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    n_img = Xtr.shape[1]
    inA, inB = np.isin(ytr, A), np.isin(ytr, B)
    teA, teB = np.isin(yte, A), np.isin(yte, B)
    print(f"phase 1: digits {A}  ({inA.sum()} train)   "
          f"phase 2: digits {B}  ({inB.sum()} train)\n")
    res, models = {}, {}

    # ---------------- baselines: same stream, gradient training ------------
    for name, mk in (("logistic (SGD)", lambda: SGDClassifier(
                          loss="log_loss", random_state=0)),
                     ("MLP 256 (SGD)", lambda: MLPClassifier(
                          hidden_layer_sizes=(256,), random_state=0,
                          learning_rate_init=0.01))):
        clf = mk(); rng = np.random.default_rng(1)
        for ph, m in (("1", inA), ("2", inB)):
            idx = np.nonzero(m)[0]
            for _ in range(EPOCHS):
                for s in range(0, len(idx), BATCH):
                    b = rng.permutation(idx)[s:s + BATCH] if s == 0 else idx[s:s + BATCH]
                    clf.partial_fit(Xtr[b], ytr[b], classes=np.arange(10))
            p = clf.predict(Xte)
            res.setdefault(name, {})[f"after phase {ph}"] = {
                "0-4": score(p, yte, teA), "5-9": score(p, yte, teB),
                "all": float((p == yte).mean())}
        r = res[name]
        print(f"[{name}]")
        print(f"  after phase 1 (0-4 only):  0-4 {r['after phase 1']['0-4']:.4f}")
        print(f"  after phase 2 (5-9 added): 0-4 {r['after phase 2']['0-4']:.4f}"
              f"   5-9 {r['after phase 2']['5-9']:.4f}"
              f"   all {r['after phase 2']['all']:.4f}"
              f"   FORGOT {r['after phase 1']['0-4']-r['after phase 2']['0-4']:+.4f}\n")

    # ---------------- hiring experts --------------------------------------
    for th in THETAS:
        mdl = Hiring(n_img + 10, th); rng = np.random.default_rng(1)
        row = {}
        for ph, m in (("1", inA), ("2", inB)):
            mdl.phase = int(ph); idx = np.nonzero(m)[0]
            for _ in range(EPOCHS):
                o = rng.permutation(idx)
                for s in range(0, len(o), BATCH):
                    b = o[s:s + BATCH]
                    mdl.learn(join(Xtr[b], ytr[b]), ytr[b])
            p = mdl.predict(Xte, n_img)
            row[f"after phase {ph}"] = {
                "0-4": score(p, yte, teA), "5-9": score(p, yte, teB),
                "all": float((p == yte).mean()), "experts": int(len(mdl.W))}
        row["hired_at"] = list(mdl.hired_at); row["trace"] = list(mdl.trace)
        row["claimed"] = mdl.wins.argmax(1).tolist()
        models[th] = mdl
        res[f"hiring experts (theta={th})"] = row
        print(f"[hiring experts, theta={th}]")
        print(f"  after phase 1 (0-4 only):  0-4 {row['after phase 1']['0-4']:.4f}"
              f"   ({row['after phase 1']['experts']} experts hired)")
        print(f"  after phase 2 (5-9 added): 0-4 {row['after phase 2']['0-4']:.4f}"
              f"   5-9 {row['after phase 2']['5-9']:.4f}"
              f"   all {row['after phase 2']['all']:.4f}"
              f"   ({row['after phase 2']['experts']} experts)"
              f"   FORGOT {row['after phase 1']['0-4']-row['after phase 2']['0-4']:+.4f}\n")

    np.savez_compressed(OUT / "models.npz", **{
        f"W_{t}": models[t].W.astype(np.float32) for t in models} , **{
        f"wins_{t}": models[t].wins for t in models}, **{
        f"hired_{t}": np.array(models[t].hired_at) for t in models})
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(res)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(names))
    ax.bar(x - .27, [res[n]["after phase 1"]["0-4"] for n in names], .25,
           label="0-4, after learning 0-4", color="#1b6ca8")
    ax.bar(x, [res[n]["after phase 2"]["0-4"] for n in names], .25,
           label="0-4, after also learning 5-9", color="#c1462d")
    ax.bar(x + .27, [res[n]["after phase 2"]["5-9"] for n in names], .25,
           label="5-9, after learning 5-9", color="#2f8f4e")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7, rotation=12, ha="right")
    ax.set_ylabel("accuracy"); ax.grid(alpha=.25, axis="y"); ax.legend(fontsize=7)
    ax.set_title("split-MNIST: does learning 5-9 destroy 0-4?", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "01_forgetting.png", dpi=130); plt.close(fig)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
