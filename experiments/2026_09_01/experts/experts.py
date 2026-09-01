"""Competing experts on 5x5 patches, trained jointly with the label.

Yesterday's competitive rule (`2026_08_31/patch_competitive`), on the
same 5x5 patches the k-means vocabulary used, with two changes:

    the vector is [patch ; label]        the label rides along as a second stream
    the competition is judged by         ||patch - rebuilt||^2
                                       + LAMBDA * ||label - rebuilt||^2

So an expert wins a patch by drawing the WHOLE thing well -- the ink and the
class it came with -- and only the winner learns. Inside an expert the K
templates rebuild densely, as a subspace; across experts exactly one wins.
That is the shape the user asked for: dense description inside, sparse identity
outside, so the description is good enough to generate from and the winner is
sharp enough to name.

Reading it back, the label is blank and only the ink chooses:

    patch in -> winner expert -> its rebuild of the label block = its OPINION
    image    -> sum those opinions over every patch -> the class

LAMBDA = 0 is the unsupervised control, and it is the number every other row has
to beat before joint training has earned anything.

Note on K: the joint vector is 35 numbers, so an expert holding K templates spans
K of 35 dimensions. Large K rebuilds everything and competition stops meaning
anything -- hence K well under 35, and a sweep to show the shape.
"""

import json, sys, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
EPS = 1e-12
PS, FLOOR = 5, 0.05                       # the k-means patch and its flatness gate
H, K, LAM, RHO, ETA, GAMMA = 30, 8, 1.0, 1.0, 0.5, 1.0
N_TRAIN, N_TEST, EPOCHS, BATCH, CHUNK, MIN_S = 12000, 3000, 3, 2048, 96, 4
PER_IMG, SEED = 60, 0
NL = 10


def load(ds="mnist"):
    d = ROOT / "data"
    X = np.load(d / f"mnist/{'fashion' if 'fashion' in ds else 'digits'}/train_images.npy").astype(np.float64).reshape(-1, 28, 28)
    y = np.load(d / f"mnist/{'fashion' if 'fashion' in ds else 'digits'}/train_labels.npy").astype(np.int64)
    if X.max() > 1.5:
        X /= 255.0
    p = np.random.default_rng(SEED).permutation(len(X))
    X, y = X[p], y[p]
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:N_TRAIN + N_TEST], y[N_TRAIN:N_TRAIN + N_TEST]


def patches(imgs):
    """Centred, unit-length 5x5 patches; flat ones dropped. Same as km.prep."""
    P = sliding_window_view(imgs, (PS, PS), axis=(1, 2)).reshape(len(imgs), -1, PS * PS)
    C = P - P.mean(-1, keepdims=True)
    n = np.linalg.norm(C, axis=-1)
    return C / np.maximum(n, EPS)[..., None], n > FLOOR


def join(U, lab=None):
    """[patch ; label * RHO], unit length. Blank label at read time."""
    V = np.zeros((len(U), PS * PS + NL))
    V[:, :PS * PS] = U
    if lab is not None:
        V[np.arange(len(U)), PS * PS + lab] = RHO
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)


def sample(X, y, rng, per_img):
    """A spread of patches per image, each carrying its image's label."""
    U, L = [], []
    for a in range(0, len(X), 256):
        Q, keep = patches(X[a:a + 256])
        for i in range(len(Q)):
            idx = np.nonzero(keep[i])[0]
            if len(idx):
                pick = rng.choice(idx, min(per_img, len(idx)), False)
                U.append(Q[i, pick]); L.append(np.full(len(pick), y[a + i]))
    return np.concatenate(U), np.concatenate(L)


def split_err(W, B, lam):
    """Per-expert error, ink and label weighted separately. (H, b)"""
    S = np.einsum('bd,hkd->hbk', B, W, optimize=True)
    R = np.einsum('hbk,hkd->hbd', S, W, optimize=True)
    D = B[None] - R
    ink = (D[:, :, :PS * PS] ** 2).sum(-1)
    lab = (D[:, :, PS * PS:] ** 2).sum(-1)
    return ink + lam * lab, ink, lab


def geo_step(Wh, B, eta):
    """The 08-30 geodesic step: rotate the subspace toward what it left over."""
    S = B @ Wh.T
    E = B - S @ Wh
    M = (S.T @ E) / len(B)
    tau = M - (M * Wh).sum(1, keepdims=True) * Wh
    tn = np.linalg.norm(tau, axis=1)
    th = np.clip(eta * tn, 0.0, np.pi / 4)
    hat = np.zeros_like(tau); live = tn > EPS
    hat[live] = tau[live] / tn[live, None]
    Wh = Wh * np.cos(th)[:, None] + hat * np.sin(th)[:, None]
    return Wh / (np.linalg.norm(Wh, axis=1, keepdims=True) + EPS)


def train(V, h, k, lam, rng, epochs=EPOCHS):
    d = V.shape[1]
    W = rng.standard_normal((h, k, d))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    f = np.full(h, 1.0 / h)
    for ep in range(epochs):
        order = rng.permutation(len(V))
        for s in range(0, len(order), BATCH):
            B = V[order[s:s + BATCH]]
            if len(B) < MIN_S * 2:
                continue
            e = split_err(W, B, lam)[0] + GAMMA * (f - 1.0 / h)[:, None]   # conscience
            win = e.argmin(0)
            cnt = np.bincount(win, minlength=h)
            f = 0.995 * f + 0.005 * (cnt / max(cnt.sum(), 1))
            for j in range(h):
                m = win == j
                if m.sum() >= MIN_S:
                    W[j] = geo_step(W[j], B[m], ETA)
        print(f"    epoch {ep+1}/{epochs}", flush=True)
    return W


def opinions(W, U):
    """Each patch: which expert wins on INK alone, and its rebuild of the label."""
    B = join(U)
    S = np.einsum('bd,hkd->hbk', B, W, optimize=True)
    R = np.einsum('hbk,hkd->hbd', S, W, optimize=True)
    ink = ((B[None] - R)[:, :, :PS * PS] ** 2).sum(-1)
    win = ink.argmin(0)
    return win, R[win, np.arange(len(U)), PS * PS:]


def evaluate(W, X, y, rng):
    """Per-patch class call, and the image-level sum of expert opinions."""
    h = len(W)
    cnt = np.zeros(h); lab_dist = np.zeros((h, NL))
    pat_hit = pat_n = 0; img_hit = 0
    for a in range(0, len(X), 128):
        Q, keep = patches(X[a:a + 128])
        for i in range(len(Q)):
            idx = np.nonzero(keep[i])[0]
            if not len(idx):
                continue
            win, ev = opinions(W, Q[i, idx])
            cnt += np.bincount(win, minlength=h)
            np.add.at(lab_dist, (win, y[a + i]), 1.0)
            pat_hit += int((ev.argmax(1) == y[a + i]).sum()); pat_n += len(idx)
            img_hit += int(ev.sum(0).argmax() == y[a + i])
    return {"patch_acc": pat_hit / max(pat_n, 1), "image_acc": img_hit / len(X),
            "counts": cnt, "lab_dist": lab_dist}


def draw(W, r, tag):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, k = W.shape[0], W.shape[1]
    order = np.argsort(-r["counts"])
    tile = np.full((h * (PS + 1) - 1, k * (PS + 1) - 1), np.nan)
    for a, j in enumerate(order):
        for b in range(k):
            t = W[j, b, :PS * PS].reshape(PS, PS)
            tile[a * (PS + 1):a * (PS + 1) + PS, b * (PS + 1):b * (PS + 1) + PS] = t
    fig, ax = plt.subplots(1, 2, figsize=(11, max(6, h * 0.28)),
                           gridspec_kw={"width_ratios": [k, NL + 2]})
    ax[0].imshow(tile, cmap="RdBu_r", interpolation="nearest")
    ax[0].set_title(f"{h} experts x {k} templates (5x5 ink part)", fontsize=9)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    d = r["lab_dist"][order]
    d = d / np.maximum(d.sum(1, keepdims=True), 1)
    ax[1].imshow(d, cmap="magma", aspect="auto", interpolation="nearest")
    ax[1].set_title("labels each expert won (row-normalised)", fontsize=9)
    ax[1].set_xticks(range(NL)); ax[1].set_xlabel("class")
    ax[1].set_yticks(range(h))
    ax[1].set_yticklabels([f"{j}  n={int(r['counts'][j]):,}" for j in order], fontsize=6)
    plt.tight_layout(); plt.savefig(OUT / f"templates_{tag}.png", dpi=130); plt.close()


def run(Vtr, Xte, yte, h, k, lam, tag, save=False):
    t0 = time.time()
    W = train(Vtr, h, k, lam, np.random.default_rng(SEED + 1))
    r = evaluate(W, Xte, yte, np.random.default_rng(SEED + 2))
    live = int((r["counts"] > r["counts"].sum() * 0.002).sum())
    d = r["lab_dist"] / np.maximum(r["lab_dist"].sum(1, keepdims=True), 1)
    purity = float(d.max(1)[r["counts"] > 0].mean())
    out = {"H": h, "K": k, "lambda": lam, "live": live,
           "patch_acc": r["patch_acc"], "image_acc": r["image_acc"],
           "expert_label_purity": purity,
           "busiest_share": float(r["counts"].max() / max(r["counts"].sum(), 1)),
           "seconds": round(time.time() - t0, 1)}
    print(f"  H={h} K={k} lam={lam:<4} live {live}/{h}  patch {r['patch_acc']:.4f}  "
          f"IMAGE {r['image_acc']:.4f}  expert purity {purity:.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    if save:
        draw(W, r, tag)
        np.savez_compressed(OUT / f"weights_{tag}.npz", W=W.astype(np.float32),
                            counts=r["counts"], lab_dist=r["lab_dist"])
    return out


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    rng = np.random.default_rng(SEED + 1)
    U, L = sample(Xtr, ytr, rng, PER_IMG)
    Vtr = join(U, L)
    print(f"{len(U):,} patches of {PS}x{PS} + label, joint dim {Vtr.shape[1]}", flush=True)
    res = {"config": {"H": H, "K": K, "rho": RHO, "eta": ETA, "gamma": GAMMA,
                      "patch": PS, "n_train": N_TRAIN, "n_test": N_TEST,
                      "epochs": EPOCHS, "patches": int(len(U))}, "runs": []}
    res["runs"].append(run(Vtr, Xte, yte, H, K, LAM, "main", save=True))
    for lam in (0.0, 4.0):
        res["runs"].append(run(Vtr, Xte, yte, H, K, lam, f"lam{lam}",
                               save=(lam == 0.0)))
    for k in (4, 12):
        res["runs"].append(run(Vtr, Xte, yte, H, k, LAM, f"k{k}"))
    res["reference"] = {"one k-means layer on pixels+label": 0.9210,
                        "logistic on pixels": 0.9074}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "experts.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s -> results/experts.json")


if __name__ == "__main__":
    main()
