"""The k sweep, the diagnostics, the timing. `python run_sparse_column.py`."""

import json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "place_code"))
from place_code import PlaceCode, unit_rows, patches_of      # noqa: E402
from sparse_column import SparseColumn, NaiveColumn          # noqa: E402

OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
NB, PATCH, K, ETA = 8, 4, 64, 0.5
KS = [1, 2, 4, 8, 16]
N_TRAIN, BATCH, EPOCHS = 24000, 128, 3
plt.rcParams.update({"figure.dpi": 130, "font.size": 8})


def load(n=3000):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    if X.ndim == 2:
        X = X.reshape(-1, 28, 28)
    if X.max() > 1.5:
        X = X / 255.0
    P = patches_of(X[:n], PATCH)
    return P[np.linalg.norm(P, axis=1) > 1e-9]


def train(col, U, epochs=EPOCHS, rng=None):
    rng = rng or np.random.default_rng(0)
    for ep in range(epochs):
        idx = rng.permutation(len(U))
        for b in range(0, len(idx), BATCH):
            col.learn(U[idx[b:b + BATCH]])
        col.revive(U[rng.choice(len(U), 2000, replace=False)])
    return col


def supports(col, U):
    C = col.pursue(U)[0]
    return C > 0


def main():
    P = load()
    pc = PlaceCode(PATCH * PATCH, nb=NB, lo=0.0, hi=1.0, halfw=1.5)
    Uall, _ = unit_rows(pc.encode(P))
    rng = np.random.default_rng(0)
    tr = rng.choice(len(Uall), min(N_TRAIN, len(Uall)), replace=False)
    U, Ptr = Uall[tr], P[tr]
    te = rng.choice(len(Uall), 4000, replace=False)
    Ute, Pte = Uall[te], P[te]
    print(f"{len(P)} patches, place code {PATCH*PATCH}x{NB} = {pc.size} cells, "
          f"K={K} templates")

    res, cols = {}, {}
    for k in KS:
        t0 = time.time()
        col = train(SparseColumn(K, pc.size, kmax=k, eta=ETA, rng=np.random.default_rng(1)), U)
        C, R, used, _, _ = col.pursue(Ute)
        rec = float(np.sqrt((R * R).sum(1).mean() / (Ute * Ute).sum(1).mean()))
        back = pc.decode(col.decode(C))
        pix = float(np.sqrt(((back - Pte) ** 2).mean()))
        G = col.W @ col.W.T
        coh = float(G[~np.eye(K, dtype=bool)].mean())
        share = col.wins / max(col.wins.sum(), 1)
        res[k] = {"rebuild_err": rec, "pixel_rmse": pix,
                  "mean_used": float(used.mean()),
                  "dead_templates": int((share < 1e-3 / K).sum()),
                  "usage_entropy_bits": float(-(share[share > 0] *
                                                np.log2(share[share > 0])).sum()),
                  "distinct_supports": int(len(np.unique(supports(col, Ute), axis=0))),
                  "coherence_mean_pairwise_cos": round(coh, 3),
                  "train_s": round(time.time() - t0, 1)}
        cols[k] = col
        print(f"  k={k:>2}  rebuild {rec:.3f}  pixel {pix:.4f}  "
              f"used {used.mean():.2f}  dead {res[k]['dead_templates']:>2}  "
              f"supports {res[k]['distinct_supports']:>4}  {res[k]['train_s']}s")

    # ---- support stability -------------------------------------------- #
    stab = {}
    Q = Pte[:1500]
    for k in KS:
        col, row = cols[k], []
        S0 = supports(col, unit_rows(pc.encode(Q))[0])
        for eps in [0.01, 0.02, 0.05, 0.10, 0.20]:
            Qp = np.clip(Q + eps * rng.standard_normal(Q.shape), 0, 1)
            S1 = supports(col, unit_rows(pc.encode(Qp))[0])
            keep = (S0 & S1).sum(1) / np.maximum(S0.sum(1), 1)
            row.append(float(keep.mean()))
        stab[k] = row
    # control: the same rule on raw 16-dim pixel patches, no place code
    Vall, _ = unit_rows(P)
    raw = train(SparseColumn(K, PATCH * PATCH, kmax=4, eta=ETA,
                             rng=np.random.default_rng(1)), Vall[tr])
    S0 = supports(raw, unit_rows(Q)[0])
    raw_row = []
    for eps in [0.01, 0.02, 0.05, 0.10, 0.20]:
        Qp = np.clip(Q + eps * rng.standard_normal(Q.shape), 0, 1)
        S1 = supports(raw, unit_rows(Qp)[0])
        raw_row.append(float(((S0 & S1).sum(1) / np.maximum(S0.sum(1), 1)).mean()))

    # ---- why the template weights are not clamped nonnegative ---------- #
    def traj(U, col, steps=9):
        X = U[rng.choice(len(U), 2000, replace=False)]
        R, e0, out = X.copy(), (X * X).sum(1), []
        for _ in range(steps):
            out.append(round(float(np.sqrt(((R * R).sum(1) / e0).mean())), 3))
            S = R @ col.W.T; i = S.argmax(1); c = S[np.arange(len(R)), i]
            R = np.where((c > 0)[:, None], R - c[:, None] * col.W[i], R)
        G = col.W @ col.W.T
        return out, round(float(G[~np.eye(col.n_t, dtype=bool)].mean()), 3)

    Vraw, _ = unit_rows(P)
    ctl = {}
    for tag, dat, dim, nn in [("place code, signed weights", Uall, pc.size, False),
                              ("place code, nonneg weights", Uall, pc.size, True),
                              ("raw pixels, signed weights", Vraw, PATCH ** 2, False),
                              ("raw pixels, nonneg weights", Vraw, PATCH ** 2, True)]:
        c = train(SparseColumn(K, dim, kmax=8, eta=ETA, nonneg_w=nn,
                               rng=np.random.default_rng(1)), dat[tr])
        t, g = traj(dat, c)
        ctl[tag] = {"residual_by_step": t, "coherence": g}
        print(f"  {tag:<28} {t}   coherence {g}")

    # ---- timing --------------------------------------------------------- #
    timing = {}
    for k in [1, 4, 8]:
        fast = SparseColumn(K, pc.size, kmax=k, rng=np.random.default_rng(2))
        slow = NaiveColumn(K, pc.size, kmax=k, rng=np.random.default_rng(2))
        for c in (fast, slow):
            c.learn(U[:K])
        B = U[:2048]
        t = time.time(); fast.learn(B); tf = time.time() - t
        t = time.time(); slow.learn(B[:256]); ts = (time.time() - t) * 8
        timing[k] = {"batched_s_per_2048": round(tf, 4),
                     "naive_s_per_2048": round(ts, 3),
                     "speedup": round(ts / max(tf, 1e-9), 1)}
        print(f"  timing k={k}: batched {tf*1e6/2048:.1f} us/patch, "
              f"naive {ts*1e6/2048:.1f} us/patch, {timing[k]['speedup']}x")

    figures(pc, cols, res, stab, raw_row, Pte, Ute, ctl)
    out = {"config": {"nb": NB, "patch": PATCH, "K": K, "eta": ETA,
                      "dim": pc.size, "train": len(U), "epochs": EPOCHS},
           "by_k": res, "support_kept_vs_noise": stab,
           "support_kept_raw_pixels_k4": raw_row,
           "nonneg_weight_control": ctl,
           "noise_levels": [0.01, 0.02, 0.05, 0.10, 0.20], "timing": timing}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    return out


def figures(pc, cols, res, stab, raw_row, Pte, Ute, ctl):
    # 1 -- what each k learns
    fig, axes = plt.subplots(len(KS), 17, figsize=(11, 3.6))
    for r, k in enumerate(KS):
        W = cols[k].W
        order = np.argsort(-cols[k].wins)[:16]
        axes[r, 0].text(0.5, 0.5, f"k={k}", ha="center", va="center", fontsize=9)
        axes[r, 0].axis("off")
        for j, t in enumerate(order):
            v = pc.decode(W[t:t + 1])[0].reshape(PATCH, PATCH)
            a = axes[r, j + 1]
            a.imshow(v, cmap="gray_r", vmin=0, vmax=1)
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("layer-one templates, decoded back to 4x4 — the 16 most used, "
                 "per k", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "01_templates.png"); plt.close(fig)

    # 2 -- the sweep
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.7))
    ax[0].plot(KS, [res[k]["rebuild_err"] for k in KS], "o-", color="#1b6ca8")
    ax[0].set_title("rebuild error"); ax[0].set_xlabel("k — how many speak")
    ax[1].plot(KS, [res[k]["mean_used"] for k in KS], "o-", color="#2f8f4e")
    ax[1].plot(KS, KS, ":", color="#999", label="k_max")
    ax[1].set_title("templates actually used\n(energy early-stop)")
    ax[1].set_xlabel("k"); ax[1].legend(fontsize=7)
    ax[2].plot(KS, [res[k]["distinct_supports"] for k in KS], "o-", color="#c1462d")
    ax[2].set_title("distinct supports seen\n(the alphabet)"); ax[2].set_xlabel("k")
    ax[2].set_yscale("log")
    ax[3].plot(KS, [res[k]["usage_entropy_bits"] for k in KS], "o-", color="#7a4fa3")
    ax[3].axhline(np.log2(K), ls=":", color="#999", label="even use")
    ax[3].set_title("usage entropy (bits)"); ax[3].set_xlabel("k"); ax[3].legend(fontsize=7)
    for a in ax:
        a.grid(alpha=0.25); a.set_xscale("log"); a.set_xticks(KS); a.set_xticklabels(KS)
    fig.tight_layout(); fig.savefig(OUT / "02_sweep.png"); plt.close(fig)

    # 3 -- support stability
    eps = [0.01, 0.02, 0.05, 0.10, 0.20]
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for k in KS:
        ax[0].plot(eps, stab[k], "o-", label=f"k={k}")
    ax[0].plot(eps, raw_row, "s--", color="k", label="raw pixels, k=4")
    ax[0].set_xlabel("noise added to the patch"); ax[0].set_ylabel("fraction of support kept")
    ax[0].set_title("does the support survive a nudge?")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.25); ax[0].set_xscale("log")
    ax[0].set_ylim(0, 1.02)
    for k in KS:
        C = cols[k].pursue(Ute)[0]
        ax[1].hist((C > 0).sum(1), bins=np.arange(0, 18) - 0.5, histtype="step",
                   label=f"k_max={k}", density=True)
    ax[1].set_xlabel("templates used on one patch"); ax[1].set_ylabel("fraction")
    ax[1].set_title("adaptive sparsity from the energy stop")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "03_stability.png"); plt.close(fig)

    # 4 -- rebuilds
    fig, axes = plt.subplots(len(KS) + 1, 12, figsize=(9, 4.2))
    for j in range(12):
        axes[0, j].imshow(Pte[j].reshape(PATCH, PATCH), cmap="gray_r", vmin=0, vmax=1)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
    axes[0, 0].set_ylabel("truth", fontsize=7)
    for r, k in enumerate(KS):
        C = cols[k].pursue(Ute[:12])[0]
        back = pc.decode(cols[k].decode(C))
        for j in range(12):
            axes[r + 1, j].imshow(back[j].reshape(PATCH, PATCH), cmap="gray_r",
                                  vmin=0, vmax=1)
            axes[r + 1, j].set_xticks([]); axes[r + 1, j].set_yticks([])
        axes[r + 1, 0].set_ylabel(f"k={k}", fontsize=7)
    fig.suptitle("held-out patches, rebuilt from the sparse code", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "04_rebuilds.png"); plt.close(fig)


    # 5 -- the nonnegative-weights finding
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for tag, d in ctl.items():
        st = "--" if "nonneg" in tag else "-"
        cl = "#c1462d" if "place" in tag else "#1b6ca8"
        ax[0].plot(d["residual_by_step"], st, color=cl, marker="o", ms=3,
                   label=f"{tag}  (coh {d['coherence']})")
    ax[0].set_xlabel("templates used"); ax[0].set_ylabel("residual / input")
    ax[0].set_title("clamping the WEIGHTS nonnegative stalls the pursuit")
    ax[0].legend(fontsize=6.5); ax[0].grid(alpha=0.25); ax[0].set_ylim(0, 1.02)
    ax[1].bar(range(4), [d["coherence"] for d in ctl.values()],
              color=["#c1462d", "#c1462d", "#1b6ca8", "#1b6ca8"],
              alpha=[1, .5, 1, .5] if False else None)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([t.replace(", ", "\n") for t in ctl], fontsize=6)
    ax[1].set_ylabel("mean cosine between two templates")
    ax[1].set_title("why: nonneg templates all point one way")
    ax[1].grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "05_nonneg_weights.png"); plt.close(fig)


if __name__ == "__main__":
    main()
