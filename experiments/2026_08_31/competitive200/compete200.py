"""40 hypercolumns x 36 templates -- same 1440-template budget as 10 x 144.

Three arms, identical except for who learns:
  competitive            winner (most confident in the TRUE label) learns
  competitive+conscience same, with a win-frequency penalty so nobody starves
  control                everybody learns from everything

Read at test time two ways, because the 10x144 run showed the right gate
depends on whether the experts are specialists:
  fit         -- which hypercolumn reconstructs the image half best
  confidence  -- which is most sure of its own prediction
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "competitive"))
sys.path.insert(0, str(HERE.parent / "competitive40"))
from compete import center_norm, load, join, geo_step, EPS

OUT = HERE / "results"
H, K, RHO, ETA = 200, 9, 1.0, 0.5
EPOCHS, BATCH, MIN_S, GAMMA = 15, 1024, 2, 1.0
SEED = 0
BOARD = {"40x36 competitive + fit gate": 0.9390,
         "40x36 conscience + soft vote": 0.9402,
         "10x144 competitive + fit gate": 0.8698,
         "sparse + tally": 0.8672, "dense + layer three": 0.9232,
         "logistic on pixels": 0.9074, "10x144 control soft vote": 0.8090}


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
    f = np.full(H, 1.0 / H)                       # running win fraction
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            J = join(Xtr[b], ytr[b])
            if mode == "control":
                for h in range(H):
                    W[h] = geo_step(W[h], J, ETA)
                continue
            P, _ = predict(W, join(Xtr[b]), n_img)
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
    Q = join(Xte)
    P, Rimg = predict(W, Q, n_img)
    pred = P.argmax(2)
    n = len(yte)
    err = np.linalg.norm(Q[None, :, :n_img] - Rimg, axis=2) / np.maximum(
        np.linalg.norm(Q[None, :, :n_img], axis=2), EPS)
    srt = np.sort(P, axis=2)
    marg = (srt[:, :, -1] - srt[:, :, -2]) / np.maximum(np.linalg.norm(P, axis=2), EPS)
    out = {}
    for name, c in (("fit", -err), ("confidence", marg)):
        pick = c.argmax(0)
        sh = np.bincount(pick, minlength=H) / n
        out[f"gate_{name}"] = float((pred[pick, np.arange(n)] == yte).mean())
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
    return out


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    n_img = Xtr.shape[1]
    print(f"{H} hypercolumns x {K} templates = {H*K} total "
          f"(the 10x144 run had {10*144})", flush=True)
    res, Ws = {}, {}
    for mode in ("competitive", "conscience", "control"):
        W, wins = train(Xtr, ytr, n_img, mode, np.random.default_rng(SEED + 1))
        r = evaluate(W, Xte, yte, n_img, mode)
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
        res[mode], Ws[mode] = r, W
        Ws[mode + "_wins"] = wins

    print("\nagainst the board:")
    for k_, v in BOARD.items():
        print(f"  {k_:<34} {v:.4f}")
    np.savez_compressed(OUT / "weights.npz",
                        **{k_: v.astype(np.float32) if v.dtype != np.int64 else v
                           for k_, v in Ws.items()})
    (OUT / "metrics.json").write_text(json.dumps(
        {"config": {"H": H, "K": K, "rho": RHO, "eta": ETA, "epochs": EPOCHS,
                    "batch": BATCH, "gamma": GAMMA, "total_templates": H * K},
         "results": res, "board": BOARD,
         "seconds": round(time.time() - t0, 1)}, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, mode in zip(ax[:2], ("competitive", "conscience")):
        w = Ws[mode + "_wins"]
        a.imshow(w / np.maximum(w.sum(1, keepdims=True), 1), cmap="magma", aspect="auto")
        a.set_xlabel("true digit"); a.set_ylabel("hypercolumn")
        a.set_title(f"{mode}: what each won", fontsize=9)
    lab, val, col = [], [], []
    for mode in ("competitive", "conscience", "control"):
        lab += [f"{mode}: fit gate", f"{mode}: conf gate"]
        val += [res[mode]["gate_fit"], res[mode]["gate_confidence"]]
        col += ["#2f8f4e", "#c1462d"]
    lab += list(BOARD); val += list(BOARD.values()); col += ["#1b6ca8"] * len(BOARD)
    ax[2].barh(range(len(lab)), val, color=col)
    for i, v in enumerate(val):
        ax[2].text(v, i, f" {v:.3f}", va="center", fontsize=6)
    ax[2].set_yticks(range(len(lab))); ax[2].set_yticklabels(lab, fontsize=6)
    ax[2].invert_yaxis(); ax[2].set_xlim(0, 1.0); ax[2].grid(alpha=.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "01_results.png", dpi=130); plt.close(fig)

    for mode in ("competitive", "conscience"):
        W = Ws[mode]
        fig, axes = plt.subplots(8, min(K, 20), figsize=(11, 4.6))
        for i, a in enumerate(axes.ravel()):
            h, j = divmod(i, min(K, 20))
            t = W[h * 5][j, :n_img].reshape(28, 28)
            m = np.abs(t).max() + EPS
            a.imshow(t, cmap="bwr", vmin=-m, vmax=m); a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"{mode} — 4 templates from every 5th hypercolumn", fontsize=10)
        fig.subplots_adjust(left=.005, right=.995, top=.93, bottom=.005,
                            wspace=.05, hspace=.05)
        fig.savefig(OUT / f"02_templates_{mode}.png", dpi=130); plt.close(fig)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
