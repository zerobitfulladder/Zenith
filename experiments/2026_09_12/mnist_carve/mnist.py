"""The carve engine on MNIST. No labels used as supervision, no patches learned,
no convolution: the label is just ten more lights in the same vector.

    codes     k-means on 4x4 patches (inked ones), k = 20, one dictionary
              shared by all 49 positions. A patch becomes ONE light:
              slot = position * 20 + cluster. Blank patches emit nothing.
    label     ten more lights, one-hot. Same vector.
    engine    ../carve/engine.py, counting every round rather than only the
              final one (the inflation bug found there), plus a fired-count
              so "the most typical template with label 3" is answerable.

Then:
    1  image -> engine -> read the label lights        classification
    2  label -> most-fired template with it -> decode   generation
    3  linear probe on the raw codes                    the baseline
    4  image -> engine -> which templates fired -> probe the explainers
    5  image -> engine -> expand -> decode               reconstruction
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "carve"))
from engine import Carve  # noqa: E402

N_TRAIN, N_TEST, K, SEED = 6000, 2000, 20, 0
P, G = 4, 7                                   # patch size, grid (7x7 = 49)
INK = 0.5                                     # a patch must sum to this to count
NCODE = G * G * K                             # 980 code lights
NL = NCODE + 10                               # + 10 label lights


# ---------------------------------------------------------------- data ----
def load(n_train, n_test, seed=0):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float32)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(int)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm], y[perm]
    return X[:n_train], y[:n_train], X[n_train:n_train + n_test], y[n_train:n_train + n_test]


def patches(X):
    """(n, 49, 16): every 4x4 non-overlapping patch of every image."""
    n = len(X)
    return X.reshape(n, G, P, G, P).transpose(0, 1, 3, 2, 4).reshape(n, G * G, P * P)


def encode(Xp, km):
    """Light vectors: one code light per inked patch. Labels added by caller."""
    n = len(Xp)
    V = np.zeros((n, NL), bool)
    inked = Xp.sum(2) > INK
    for i in range(n):
        pos = np.flatnonzero(inked[i])
        if len(pos):
            c = km.predict(Xp[i, pos])
            V[i, pos * K + c] = True
    return V


def decode(lights, cent):
    """Code lights back to a 28x28 picture: paint each patch's centroid."""
    img = np.zeros((28, 28), np.float32)
    for s in np.flatnonzero(lights[:NCODE]):
        pos, c = divmod(int(s), K)
        r, q = divmod(pos, G)
        img[r * P:(r + 1) * P, q * P:(q + 1) * P] = cent[c].reshape(P, P)
    return img


# -------------------------------------------------------------- engine ----
class CarveMNIST(Carve):
    """Count every round, not only the final vector -- the absorbed-as-absent
    bug from ../carve -- and keep a fired count per template."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fired = np.zeros(self.M)
        self.round_of = np.full(self.M, -1)

    def depth(self):
        """1 for a template of lights; 1 + the deepest template member
        otherwise. Members are always older than the template, so one forward
        pass is exact."""
        d = np.ones(self.V, int)
        for t in range(self.V):
            ms = [m - self.NL for m in self.members[t] if m >= self.NL]
            if ms:
                d[t] = 1 + max(d[m] for m in ms)
        return d

    def fit(self, X, log_every=1000):
        t0 = time.time()
        for i, x in enumerate(X):
            v = np.zeros(self.nT, bool)
            v[:self.NL] = x
            if i < self.warmup:
                self.observe(v)
                continue
            for r in range(self.rounds):
                self.observe(v)
                seq, resid = self.search(v)
                for t in seq:
                    self.fired[t] += 1
                t = self.carve(resid)
                if t is not None:
                    self.round_of[t] = r
                    seq = seq + [t]
                if not seq:
                    break
                nv = self.substitute(v, seq)
                if nv.sum() >= v.sum():
                    break
                v = nv
            if log_every and (i + 1) % log_every == 0:
                print(f"    {i+1} inputs, {self.V} templates, {time.time()-t0:.0f}s", flush=True)
        return self

    def read(self, lights):
        """Explain an input. Returns which templates fired (all rounds), the
        top vector, and the expansion of the templates ALONE (no input OR'd
        in), so reconstruction shows what the templates captured."""
        v = np.zeros(self.nT, bool)
        v[:self.NL] = lights
        fired = []
        for _ in range(self.rounds):
            seq, _ = self.search(v)
            if not seq:
                break
            fired += seq
            nv = self.substitute(v, seq)
            if nv.sum() >= v.sum():
                break
            v = nv
        p = self.propose(v)
        if p is not None:
            fired.append(p)
            v = v.copy()
            v[self.NL + p] = True
        only = np.zeros(self.nT, bool)
        only[self.NL:] = v[self.NL:]
        return fired, v, self.expand(only)

    def label_votes(self, fired):
        """Expand each fired template all the way down and count the label
        lights it reaches. Direct membership alone misses every label that
        sits inside a member of a member."""
        votes = np.zeros(10)
        for t in fired:
            v = np.zeros(self.nT, bool)
            v[self.NL + t] = True
            votes += self.expand(v)[NCODE:NL]
        return votes


# ---------------------------------------------------------------- main ----
def main():
    Xtr, ytr, Xte, yte = load(N_TRAIN, N_TEST, SEED)
    Ptr, Pte = patches(Xtr), patches(Xte)

    # ---- codebook
    t0 = time.time()
    ink = Ptr[Ptr.sum(2) > INK]
    km = KMeans(K, n_init=3, random_state=SEED).fit(ink)
    cent = km.cluster_centers_.astype(np.float32)
    print(f"k-means on {len(ink)} inked patches: {time.time()-t0:.1f}s")

    Vtr, Vte = encode(Ptr, km), encode(Pte, km)
    print(f"code lights per image: {Vtr[:, :NCODE].sum(1).mean():.1f} of {NCODE}")
    Vtr_lab = Vtr.copy()
    Vtr_lab[np.arange(N_TRAIN), NCODE + ytr] = True

    # ---- engine
    t0 = time.time()
    eng = CarveMNIST(NL, M=3000, min_frac=0.6, warmup=1000, min_age=300).fit(Vtr_lab)
    print(f"engine: {eng.V} templates in {time.time()-t0:.0f}s")
    dep = eng.depth()
    np.savez_compressed(HERE / "results/engine.npz", members=np.array(eng.members, dtype=object),
                        fired=eng.fired[:eng.V], depth=dep, round_of=eng.round_of[:eng.V],
                        E=np.packbits(eng.E, axis=1), V=eng.V, cent=cent)
    # labelled = reaches a label light by expansion, at any depth
    def reaches_label(t):
        v = np.zeros(eng.nT, bool); v[eng.NL + t] = True
        return eng.expand(v)[NCODE:NL].any()
    labelled = [t for t in range(eng.V) if reaches_label(t)]
    lab_mask = np.zeros(eng.V, bool); lab_mask[labelled] = True
    print("  templates by depth      :", {int(k): int((dep == k).sum()) for k in np.unique(dep)})
    print("  labelled ones by depth  :", {int(k): int(((dep == k) & lab_mask).sum()) for k in np.unique(dep)})
    print("  carved at round         :", {int(k): int((eng.round_of[:eng.V] == k).sum())
                                          for k in np.unique(eng.round_of[:eng.V])})
    print("  members per template    :", {int(k): int(v) for k, v in zip(*np.unique(
        [len(m) for m in eng.members], return_counts=True))})

    # ---- 1. classification by reading the label lights
    t0 = time.time()
    fired_te, recon_te, pred = [], [], np.full(N_TEST, -1)
    for i in range(N_TEST):
        f, _, ex = eng.read(Vte[i])
        fired_te.append(f)
        recon_te.append(ex)
        votes = eng.label_votes(f)
        if votes.any():
            pred[i] = int(np.argmax(votes))
    covered = pred >= 0
    acc = (pred == yte).mean()
    acc_cov = (pred[covered] == yte[covered]).mean() if covered.any() else 0.0
    print(f"\n1. read the label:   acc {acc:.3f}   "
          f"({covered.mean():.0%} of images got any label; {acc_cov:.3f} on those)"
          f"   [{time.time()-t0:.0f}s]")

    # ---- 2. generation from a label: the most-fired, and the deepest
    def expand_t(t):
        v = np.zeros(eng.nT, bool); v[eng.NL + t] = True
        return decode(eng.expand(v), cent)
    gen, gen_deep, gen_depth = [], [], []
    for d in range(10):
        cands = [t for t in labelled if eng.E[t, NCODE + d]]
        if not cands:
            gen.append(np.zeros((28, 28), np.float32)); gen_deep.append(gen[-1]); gen_depth.append(0)
            continue
        gen.append(expand_t(max(cands, key=lambda t: eng.fired[t])))
        td = max(cands, key=lambda t: (dep[t], eng.fired[t]))
        gen_deep.append(expand_t(td)); gen_depth.append(int(dep[td]))
    print(f"2. generation:       deepest labelled template per digit has depth {gen_depth}")

    # ---- 3. linear probe on the raw codes
    t0 = time.time()
    lr = LogisticRegression(max_iter=2000).fit(Vtr[:, :NCODE], ytr)
    acc_codes = lr.score(Vte[:, :NCODE], yte)
    print(f"3. probe on codes:   acc {acc_codes:.3f}   [{time.time()-t0:.0f}s]")

    # ---- 4. linear probe on which templates fired
    t0 = time.time()
    def fired_matrix(V):
        Fm = np.zeros((len(V), eng.V), bool)
        for i in range(len(V)):
            f, _, _ = eng.read(V[i])
            Fm[i, f] = True
        return Fm
    NP = 3000
    Ftr = fired_matrix(Vtr[:NP])
    Fte = np.zeros((N_TEST, eng.V), bool)
    for i, f in enumerate(fired_te):
        Fte[i, f] = True
    lr2 = LogisticRegression(max_iter=2000).fit(Ftr, ytr[:NP])
    acc_expl = lr2.score(Fte, yte)
    print(f"4. probe on fired:   acc {acc_expl:.3f}   "
          f"({Fte.sum(1).mean():.1f} templates fire per image)   [{time.time()-t0:.0f}s]")

    # ---- 5. reconstruction quality
    rec_err = np.mean([np.abs(decode(recon_te[i], cent) - decode(Vte[i], cent)).mean()
                       for i in range(200)])
    base_err = np.mean([np.abs(decode(Vte[i], cent)).mean() for i in range(200)])
    print(f"5. reconstruction:   mean |templates - codes| {rec_err:.3f}  "
          f"(all-black would be {base_err:.3f})")

    np.savez(HERE / "results/summary.npz", acc=acc, acc_cov=acc_cov, coverage=covered.mean(),
             acc_codes=acc_codes, acc_expl=acc_expl, V=eng.V, labelled=len(labelled), depth=dep)

    # ---- board
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 17))
    fig.suptitle(f"carve on MNIST -- {N_TRAIN} train, k={K} on 4x4 patches, {eng.V} templates "
                 f"(depths {dict(zip(*[list(map(int, a)) for a in np.unique(dep, return_counts=True)]))}), "
                 f"no supervision\n"
                 f"read the label: {acc:.3f}   |   probe on codes: {acc_codes:.3f}   |   "
                 f"probe on fired templates: {acc_expl:.3f}", fontsize=12)
    gs = fig.add_gridspec(8, 20, hspace=0.35, wspace=0.08)

    for c in range(K):                                       # codebook
        ax = fig.add_subplot(gs[0, c]); ax.imshow(cent[c].reshape(P, P), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0: ax.set_title("the 20 patch codes", fontsize=9, loc="left")

    def tile(row, ts, ylabel):
        for j, t in enumerate(ts[:20]):
            ax = fig.add_subplot(gs[row, j])
            ax.imshow(expand_t(t), cmap="gray", vmin=0, vmax=1)
            lab = np.flatnonzero(eng.E[t, NCODE:NL])
            ax.set_title((f"{lab[0]}" if len(lab) else "-") + f"d{dep[t]}", fontsize=7, pad=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0: ax.set_ylabel(ylabel, fontsize=8)
    tile(1, np.argsort(-eng.fired[:eng.V]), "most-fired")
    tile(2, np.lexsort((-eng.fired[:eng.V], -dep)), "deepest")
    tile(3, [t for t in np.lexsort((-eng.fired[:eng.V], -dep)) if lab_mask[t]], "deepest\nlabelled")

    for d in range(10):                                      # generation
        ax = fig.add_subplot(gs[4, d * 2:d * 2 + 2])
        ax.imshow(gen[d], cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"label {d}, most-fired", fontsize=8)
        ax = fig.add_subplot(gs[5, d * 2:d * 2 + 2])
        ax.imshow(gen_deep[d], cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"label {d}, deepest (d{gen_depth[d]})", fontsize=8)

    for i in range(10):                                      # reconstruction
        ax = fig.add_subplot(gs[6, i * 2:i * 2 + 2])
        ax.imshow(decode(Vte[i], cent), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"codes ({yte[i]})", fontsize=8)
        ax = fig.add_subplot(gs[7, i * 2:i * 2 + 2])
        ax.imshow(decode(recon_te[i], cent), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"templates say {pred[i] if pred[i] >= 0 else '?'}", fontsize=8)
    fig.savefig(HERE / "results/board.png", dpi=110, bbox_inches="tight")
    print("wrote results/board.png")


if __name__ == "__main__":
    main()
