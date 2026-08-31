"""The competitive40 architecture, unchanged, on Fashion-MNIST.

Same everything as ../competitive40: 40 hypercolumns x 36 templates on the
joined vector [image ; label], rho = 1.0, the 08-30 weighted geodesic batch
rule, eta 0.5, 6 epochs, batch 128. Only the images and labels differ.

Three arms (competitive / conscience / control) and two gates (fit /
confidence), because on MNIST which gate wins depended on whether the arm
produced specialists.
"""

import json, sys, time
from pathlib import Path
import numpy as np
from common import load, join, EPS, geo_step, CLASSES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
H, K, RHO, ETA = 40, 36, 1.0, 0.5
EPOCHS, BATCH, MIN_S, GAMMA = 6, 128, 2, 1.0
SEED = 0
MNIST_BOARD = {"MNIST: competitive + fit gate": 0.9390,
               "MNIST: conscience + fit gate": 0.9392,
               "MNIST: control + soft vote": 0.8246,
               "MNIST: logistic on pixels": 0.9074}


def predict(W, Q, n_img):
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k).transpose(1, 0, 2)
    R = np.matmul(S, W)
    return R[:, :, n_img:], R[:, :, :n_img]


def train(Xtr, ytr, n_img, mode, rng):
    d = n_img + 10
    W = rng.standard_normal((H, K, d))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H, 10), dtype=np.int64)
    f = np.full(H, 1.0 / H)
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            J = join(Xtr[b], ytr[b], RHO)
            if mode == "control":
                for h in range(H):
                    W[h] = geo_step(W[h], J, ETA)
                continue
            P, _ = predict(W, join(Xtr[b], None, RHO), n_img)
            conf = P[:, np.arange(len(b)), ytr[b]] / np.maximum(
                np.linalg.norm(P, axis=2), EPS)
            if mode == "conscience":
                conf = conf + GAMMA * (1.0 / H - f)[:, None]
            win = conf.argmax(0)
            cnt = np.bincount(win, minlength=H)
            f = 0.99 * f + 0.01 * (cnt / max(cnt.sum(), 1))
            for h in range(H):
                m = win == h
                if m.sum() >= MIN_S:
                    W[h] = geo_step(W[h], J[m], ETA)
                np.add.at(wins[h], ytr[b][m], 1)
        print(f"  {mode:<12} epoch {ep+1}/{EPOCHS}", flush=True)
    return W, wins


def evaluate(W, Xte, yte, n_img, tag):
    Q = join(Xte, None, RHO)
    P, Rimg = predict(W, Q, n_img)
    pred = P.argmax(2)
    n = len(yte)
    err = np.linalg.norm(Q[None, :, :n_img] - Rimg, axis=2) / np.maximum(
        np.linalg.norm(Q[None, :, :n_img], axis=2), EPS)
    srt = np.sort(P, axis=2)
    marg = (srt[:, :, -1] - srt[:, :, -2]) / np.maximum(np.linalg.norm(P, axis=2), EPS)
    out, picks = {}, {}
    for name, c in (("fit", -err), ("confidence", marg)):
        pick = c.argmax(0)
        sh = np.bincount(pick, minlength=H) / n
        yp = pred[pick, np.arange(n)]
        picks[name] = yp
        out[f"gate_{name}"] = float((yp == yte).mean())
        out[f"gate_{name}_busiest"] = float(sh.max())
        out[f"gate_{name}_used"] = int((sh > 0).sum())
    Pn = P / np.maximum(np.linalg.norm(P, axis=2, keepdims=True), EPS)
    out["soft_vote"] = float((Pn.mean(0).argmax(1) == yte).mean())
    out["oracle"] = float((pred == yte[None]).any(0).mean())
    out["single_mean"] = float(np.mean([(pred[h] == yte).mean() for h in range(H)]))
    print(f"\n[{tag}]  fit-gate {out['gate_fit']:.4f}   conf-gate "
          f"{out['gate_confidence']:.4f}   soft {out['soft_vote']:.4f}   "
          f"one column {out['single_mean']:.4f}   ORACLE {out['oracle']:.4f}")
    print(f"    fit-gate uses {out['gate_fit_used']}/{H} columns, "
          f"busiest {out['gate_fit_busiest']*100:.1f}%")
    return out, picks


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load("fashion_mnist")
    n_img = Xtr.shape[1]
    print(f"Fashion-MNIST  {H} hypercolumns x {K} templates = {H*K}", flush=True)
    res, Ws, best = {}, {}, (None, -1.0, None)
    for mode in ("competitive", "conscience", "control"):
        W, wins = train(Xtr, ytr, n_img, mode, np.random.default_rng(SEED + 1))
        r, picks = evaluate(W, Xte, yte, n_img, mode)
        live = wins.sum(1) > wins.sum() * 0.002
        pur = wins.max(1) / np.maximum(wins.sum(1), 1)
        r.update({"dead_hypercolumns": int((~live).sum()),
                  "train_win_share_max": float((wins.sum(1) / max(wins.sum(), 1)).max()),
                  "mean_class_purity_live": float(pur[live].mean()) if live.any() else 0.0,
                  "pure_specialists_over_0.9": int((pur[live] > 0.9).sum()) if live.any() else 0})
        ov = [float((np.linalg.svd(W[i] @ W[j].T, compute_uv=False) ** 2).sum() /
                    np.sqrt((np.linalg.svd(W[i] @ W[i].T, compute_uv=False) ** 2).sum() *
                            (np.linalg.svd(W[j] @ W[j].T, compute_uv=False) ** 2).sum()))
              for i in range(0, H, 3) for j in range(i + 1, H, 3)]
        r["subspace_overlap"] = float(np.mean(ov))
        if mode != "control":
            print(f"    dead {r['dead_hypercolumns']}/{H}   busiest trains on "
                  f"{r['train_win_share_max']*100:.1f}%   "
                  f"{r['pure_specialists_over_0.9']} pure specialists   "
                  f"overlap {r['subspace_overlap']:.4f}")
        res[mode], Ws[mode], Ws[mode + "_wins"] = r, W, wins
        for g in ("fit", "confidence"):
            if r[f"gate_{g}"] > best[1]:
                best = (f"{mode} + {g} gate", r[f"gate_{g}"], picks[g])

    # per-class accuracy of the best arm/gate, next to the baseline's
    name, acc, yp = best
    per = {CLASSES[c]: float((yp[yte == c] == c).mean()) for c in range(10)}
    cm = np.zeros((10, 10), dtype=int); np.add.at(cm, (yte, yp), 1)
    print(f"\nbest: {name} {acc:.4f}\nper class:")
    for k_, v in sorted(per.items(), key=lambda kv: kv[1]):
        print(f"  {k_:<14} {v:.4f}")

    bl = OUT / "baselines_fashion_mnist.json"
    board = dict(MNIST_BOARD)
    if bl.exists():
        b = json.loads(bl.read_text())["results"]
        for k_ in ("logistic (lbfgs)", "MLP 256", "SVM rbf", "kNN k=3"):
            if k_ in b:
                board[f"Fashion: {k_}"] = b[k_]["acc"]
    print("\nagainst the board:")
    for k_, v in board.items():
        print(f"  {k_:<34} {v:.4f}")

    np.savez_compressed(OUT / "l1_weights.npz",
                        **{k_: (v if v.dtype == np.int64 else v.astype(np.float32))
                           for k_, v in Ws.items()})
    (OUT / "l1_metrics.json").write_text(json.dumps(
        {"config": {"H": H, "K": K, "rho": RHO, "eta": ETA, "epochs": EPOCHS,
                    "batch": BATCH, "gamma": GAMMA, "total_templates": H * K,
                    "n_train": len(Xtr), "n_test": len(Xte)},
         "results": res, "best": {"arm": name, "acc": acc, "per_class": per,
                                  "confusion": cm.tolist()},
         "board": board, "seconds": round(time.time() - t0, 1)}, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, mode in zip(ax[:2], ("competitive", "conscience")):
        w = Ws[mode + "_wins"]
        a.imshow(w / np.maximum(w.sum(1, keepdims=True), 1), cmap="magma", aspect="auto")
        a.set_xticks(range(10)); a.set_xticklabels(CLASSES, rotation=90, fontsize=6)
        a.set_ylabel("hypercolumn"); a.set_title(f"{mode}: what each won", fontsize=9)
    lab, val, col = [], [], []
    for mode in ("competitive", "conscience", "control"):
        lab += [f"{mode}: fit gate", f"{mode}: conf gate"]
        val += [res[mode]["gate_fit"], res[mode]["gate_confidence"]]
        col += ["#2f8f4e", "#c1462d"]
    lab += list(board); val += list(board.values())
    col += ["#1b6ca8" if k_.startswith("Fashion") else "#999" for k_ in board]
    ax[2].barh(range(len(lab)), val, color=col)
    for i, v in enumerate(val):
        ax[2].text(v, i, f" {v:.3f}", va="center", fontsize=6)
    ax[2].set_yticks(range(len(lab))); ax[2].set_yticklabels(lab, fontsize=6)
    ax[2].invert_yaxis(); ax[2].set_xlim(0, 1.0); ax[2].grid(alpha=.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "10_l1_results.png", dpi=130); plt.close(fig)

    for mode in ("competitive", "conscience"):
        W = Ws[mode]
        fig, axes = plt.subplots(8, 20, figsize=(11, 4.6))
        for i, a in enumerate(axes.ravel()):
            h, j = divmod(i, 20)
            t = W[h * 5][j, :n_img].reshape(28, 28)
            m = np.abs(t).max() + EPS
            a.imshow(t, cmap="bwr", vmin=-m, vmax=m); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{mode} — 20 templates from every 5th hypercolumn", fontsize=10)
        fig.subplots_adjust(left=.005, right=.995, top=.93, bottom=.005,
                            wspace=.05, hspace=.05)
        fig.savefig(OUT / f"11_templates_{mode}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="magma")
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=90, fontsize=6)
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES, fontsize=6)
    ax.set_title(f"{name} — {acc:.4f}", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "12_confusion.png", dpi=130); plt.close(fig)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
