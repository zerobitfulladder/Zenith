"""10 hypercolumns, and only the one most confident in the label learns.

Same shape as `../dense_ensemble`: 08-30 dense `weighted` rule, 144
templates, whole MNIST, [image ; label] at rho=1.0. One change:

    per sample, write the image and leave the label blank, ask every
    hypercolumn to complete it, and let ONLY the hypercolumn whose
    completion most supports the TRUE label take a learning step
    (on the full joined vector).

Identity is not assigned; it is earned by being the one that learns. The
control arm is the same thing with every hypercolumn learning from every
sample -- the `dense_ensemble` rule -- so the two differ in exactly one line.

At test time the winner is the hypercolumn most confident in its OWN
prediction, and its label is the answer.
"""

import json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
EPS = 1e-12
H, K, RHO, ETA = 10, 144, 1.0, 0.5
N_TRAIN, N_TEST, EPOCHS, BATCH = 20000, 5000, 3, 64
SEED = 0
PRIOR = {"jointly trained 30, soft vote (n=10)": 0.7953,
         "jointly trained 30, best gate (margin/norm)": 0.7860,
         "jointly trained 30, one column (mean)": 0.7635,
         "jointly trained 30, ORACLE": 0.9140,
         "logistic regression on pixels": 0.9074}


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


def join(X, y=None):
    V = np.zeros((len(X), X.shape[1] + 10))
    V[:, :X.shape[1]] = X
    if y is not None:
        L = np.zeros((len(X), 10)); L[np.arange(len(X)), y] = 1.0
        g = RHO * np.linalg.norm(X, axis=1) / np.maximum(np.linalg.norm(L, axis=1), EPS)
        V[:, X.shape[1]:] = L * g[:, None]
    return center_norm(V)


def predict(W, Q, n_img):
    """(H, n, 10) -- every hypercolumn completes the blank label block."""
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k).transpose(1, 0, 2)
    return np.matmul(S, W)[:, :, n_img:]


def geo_step(Wh, B, eta):
    """One geodesic step for one hypercolumn: the 08-30 weighted batch rule."""
    S = B @ Wh.T
    E = B - S @ Wh
    M = (S.T @ E) / len(B)
    tau = M - (M * Wh).sum(1, keepdims=True) * Wh
    tn = np.linalg.norm(tau, axis=1)
    th = np.clip(eta * tn, 0.0, np.pi / 4)
    hat = np.zeros_like(tau)
    live = tn > EPS
    hat[live] = tau[live] / tn[live, None]
    Wh = Wh * np.cos(th)[:, None] + hat * np.sin(th)[:, None]
    return Wh / (np.linalg.norm(Wh, axis=1, keepdims=True) + EPS)


def train(Xtr, ytr, n_img, competitive, rng):
    d = n_img + 10
    W = rng.standard_normal((H, K, d))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H, 10), dtype=np.int64)
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            J = join(Xtr[b], ytr[b])
            if not competitive:
                for h in range(H):
                    W[h] = geo_step(W[h], J, ETA)
                continue
            P = predict(W, join(Xtr[b]), n_img)              # (H, B, 10)
            conf = P[:, np.arange(len(b)), ytr[b]] / np.maximum(
                np.linalg.norm(P, axis=2), EPS)              # (H, B)
            win = conf.argmax(0)
            for h in range(H):
                m = win == h
                if m.sum() >= 4:
                    W[h] = geo_step(W[h], J[m], ETA)
                np.add.at(wins[h], ytr[b][m], 1)
        print(f"  {'competitive' if competitive else 'control    '} "
              f"epoch {ep+1}/{EPOCHS}", flush=True)
    return W, wins


def evaluate(W, Xte, yte, n_img, tag):
    P = predict(W, join(Xte), n_img)
    pred = P.argmax(2)
    srt = np.sort(P, axis=2)
    margin = (srt[:, :, -1] - srt[:, :, -2]) / np.maximum(np.linalg.norm(P, axis=2), EPS)
    pick = margin.argmax(0)
    gate = float((pred[pick, np.arange(len(yte))] == yte).mean())
    Pn = P / np.maximum(np.linalg.norm(P, axis=2, keepdims=True), EPS)
    soft = float((Pn.mean(0).argmax(1) == yte).mean())
    single = np.array([float((pred[h] == yte).mean()) for h in range(H)])
    orc = float((pred == yte[None]).any(0).mean())
    share = np.bincount(pick, minlength=H) / len(yte)
    print(f"\n[{tag}]  gate (most confident) {gate:.4f}   soft vote {soft:.4f}   "
          f"one column mean {single.mean():.4f} (sd {single.std():.4f})   "
          f"oracle {orc:.4f}")
    print(f"    gate picks: busiest column {share.max()*100:.1f}%, "
          f"{int((share>0).sum())}/{H} ever used")
    return {"gate_most_confident": gate, "soft_vote": soft,
            "single_mean": float(single.mean()), "single_sd": float(single.std()),
            "single_all": single.tolist(), "oracle": orc,
            "gate_share_max": float(share.max()),
            "gate_columns_used": int((share > 0).sum())}


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    n_img = Xtr.shape[1]
    res = {}
    Wc, wins = train(Xtr, ytr, n_img, True, np.random.default_rng(SEED + 1))
    res["competitive"] = evaluate(Wc, Xte, yte, n_img, "competitive")
    Wj, _ = train(Xtr, ytr, n_img, False, np.random.default_rng(SEED + 1))
    res["control_all_learn"] = evaluate(Wj, Xte, yte, n_img, "control: all learn")

    ws = wins.sum(1) / max(wins.sum(), 1)
    pur = wins.max(1) / np.maximum(wins.sum(1), 1)
    print(f"\ntraining win share per hypercolumn: "
          f"{np.array2string(ws, precision=3, floatmode='fixed')}")
    print(f"class purity of what each one won:  "
          f"{np.array2string(pur, precision=3, floatmode='fixed')}")
    ov = lambda W: float(np.mean([
        (np.linalg.svd(W[i] @ W[j].T, compute_uv=False) ** 2).sum() /
        np.sqrt((np.linalg.svd(W[i] @ W[i].T, compute_uv=False) ** 2).sum() *
                (np.linalg.svd(W[j] @ W[j].T, compute_uv=False) ** 2).sum())
        for i in range(H) for j in range(i + 1, H)]))
    oc, oj = ov(Wc), ov(Wj)
    print(f"pairwise subspace overlap:  competitive {oc:.4f}   control {oj:.4f}"
          f"   (1.0 = identical span)")
    res.update({"train_win_share": ws.tolist(), "won_class_purity": pur.tolist(),
                "overlap_competitive": oc, "overlap_control": oj,
                "win_matrix": wins.tolist(), "prior_runs": PRIOR})

    np.savez_compressed(OUT / "weights.npz", W_competitive=Wc.astype(np.float32),
                        W_control=Wj.astype(np.float32), wins=wins)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    im = ax[0].imshow(wins / np.maximum(wins.sum(1, keepdims=True), 1), cmap="magma")
    ax[0].set_xlabel("true digit"); ax[0].set_ylabel("hypercolumn")
    ax[0].set_xticks(range(10)); ax[0].set_yticks(range(H))
    ax[0].set_title("what each hypercolumn won during training", fontsize=9)
    fig.colorbar(im, ax=ax[0], fraction=.046)
    lab = ["competitive: gate", "competitive: soft", "control: gate", "control: soft"] + list(PRIOR)
    val = [res["competitive"]["gate_most_confident"], res["competitive"]["soft_vote"],
           res["control_all_learn"]["gate_most_confident"],
           res["control_all_learn"]["soft_vote"]] + list(PRIOR.values())
    ax[1].barh(range(len(lab)), val,
               color=["#2f8f4e", "#2f8f4e", "#888", "#888"] + ["#1b6ca8"] * len(PRIOR))
    for i, v in enumerate(val):
        ax[1].text(v, i, f" {v:.4f}", va="center", fontsize=6.5)
    ax[1].set_yticks(range(len(lab))); ax[1].set_yticklabels(lab, fontsize=6.5)
    ax[1].invert_yaxis(); ax[1].set_xlim(0, 1.0); ax[1].grid(alpha=.25, axis="x")
    ax[1].set_title("classification accuracy", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "01_results.png", dpi=130); plt.close(fig)

    for tag, Wx in (("competitive", Wc), ("control", Wj)):
        fig, axes = plt.subplots(H, 16, figsize=(11, 0.72 * H))
        for h in range(H):
            for j in range(16):
                t = Wx[h, j, :n_img].reshape(28, 28)
                m = np.abs(t).max() + EPS
                axes[h, j].imshow(t, cmap="bwr", vmin=-m, vmax=m)
                axes[h, j].set_xticks([]); axes[h, j].set_yticks([])
            axes[h, 0].set_ylabel(f"h{h}", fontsize=6, rotation=0, ha="right", va="center")
        fig.suptitle(f"{tag} — first 16 templates of each hypercolumn "
                     f"(image half only)", fontsize=10)
        fig.subplots_adjust(left=.035, right=.995, top=.92, bottom=.005,
                            wspace=.05, hspace=.05)
        fig.savefig(OUT / f"02_templates_{tag}.png", dpi=130); plt.close(fig)

    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
