"""One hypercolumn, 144 templates, all explaining MNIST together.

Trains the three learning rules in collective.py side by side, photographs
the templates at log-spaced points through training, and measures how well
the layer rebuilds held-out digits against three reference points:

  best possible   the top-144 principal subspace -- the lowest error any
                  144-direction linear code can reach
  untrained       144 random directions, never learned
  codebook        144 templates, winner-take-all, each digit rebuilt from a
                  single template alone (the "each explains alone" reading)

Reconstruction is measured three ways, because the readout and the templates
can fail independently:

  plain     recon = W^T s              scores used straight, as trained
  gain      recon = a* W^T s           one scalar fitted per digit -- removes
                                       any constant over/under-explanation
  fitted    recon = W^T c, c solving the ridge least squares -- ignores the
                                       readout entirely and asks only whether
                                       the templates *span* the digit

Usage:
    python experiments/2026_08_30/collective/run_collective.py                    # everything, ~6 min
    python experiments/2026_08_30/collective/run_collective.py --rules weighted   # one rule
    python experiments/2026_08_30/collective/run_collective.py --smoke            # 2k digits, sanity check
"""

import argparse, json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from collective import Hypercolumn, Codebook, center_norm_batch, RULES

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SIDE = 28
EPS = 1e-8

# `weighted` scales its step by |s_i| (~0.1 early on), so it needs a bigger
# nominal step to move at the same rate as the other two.
DEFAULT_ETA = {"raw": 0.05, "unit": 0.05, "weighted": 0.5}


# --------------------------------------------------------------------------- data
def load_mnist(n_train, n_test, seed=0):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    y = np.load(ROOT / "data/mnist/digits/train_labels.npy").astype(np.int64)
    perm = np.random.default_rng(seed).permutation(len(X))
    X, y = X[perm].reshape(len(X), -1), y[perm]
    return X[:n_train], y[:n_train], X[n_train:n_train + n_test], y[n_train:n_train + n_test]


# --------------------------------------------------------------------- measurement
def rebuild_errors(W, Xh, ridge=1e-3):
    """Relative rebuild error of held-out digits, three readouts. Lower is better."""
    S = Xh @ W.T
    R = S @ W
    plain = np.linalg.norm(Xh - R, axis=1)                      # ||x_hat|| == 1
    rr = np.einsum("ij,ij->i", R, R)
    a = np.einsum("ij,ij->i", Xh, R) / np.maximum(rr, EPS)      # best single scalar
    gain = np.linalg.norm(Xh - a[:, None] * R, axis=1)
    G = W @ W.T
    C = np.linalg.solve(G + ridge * np.eye(len(W)), S.T).T      # ridge least squares
    fitted = np.linalg.norm(Xh - C @ W, axis=1)
    return (float(plain.mean()), float(gain.mean()), float(fitted.mean()),
            float(a.mean()), S)


def diagnostics(W, Xh, pca_basis):
    """Numbers that name a collapse if one has happened."""
    plain, gain, fitted, gain_mean, S = rebuild_errors(W, Xh)
    G = W @ W.T
    off = np.abs(G - np.diag(np.diag(G)))
    lam = np.linalg.eigvalsh(G)                                  # trace == k
    rms = np.sqrt((S ** 2).mean(axis=0))
    # How much of the layer's span sits inside the best 144-dim subspace.
    Q = np.linalg.qr(W.T)[0]
    overlap = float(np.sum((pca_basis @ Q) ** 2) / len(W))
    return dict(
        err_plain=plain, err_gain=gain, err_fitted=fitted, gain=gain_mean,
        coh_max=float(off.max()), coh_mean=float(off.sum() / (len(W) * (len(W) - 1))),
        eff_rank=float(lam.sum() ** 2 / np.maximum((lam ** 2).sum(), EPS)),
        pca_overlap=overlap,
        score_rms=float(rms.mean()),
        quiet=float((rms < 0.1 * rms.mean()).mean()),            # near-silent templates
    )


# ------------------------------------------------------------------------ pictures
def save_templates(W, path, title):
    """12x12 sheet of templates. Each is signed, so red/blue around its own zero."""
    k = len(W)
    side = int(np.ceil(np.sqrt(k)))
    fig, axes = plt.subplots(side, side, figsize=(side * 0.62, side * 0.62))
    for i, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.3); sp.set_color("0.75")
        if i < k:
            t = W[i].reshape(SIDE, SIDE)
            m = np.abs(t).max() + EPS
            ax.imshow(t, cmap="bwr", vmin=-m, vmax=m, interpolation="nearest")
        else:
            ax.axis("off")
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.965, bottom=0.004,
                        wspace=0.06, hspace=0.06)
    fig.savefig(path, dpi=115); plt.close(fig)


def save_rebuilds(rows, labels, path, title):
    """Original digits on top, each rebuild underneath, in pixel space."""
    n = rows[0].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 0.75, len(rows) * 0.82))
    for r, (imgs, name) in enumerate(zip(rows, labels)):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c].reshape(SIDE, SIDE), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
            if c == 0:
                ax.text(-0.18, 0.5, name, transform=ax.transAxes, ha="right",
                        va="center", fontsize=8.5)
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.90, bottom=0.01,
                        wspace=0.05, hspace=0.05)
    fig.savefig(path, dpi=140); plt.close(fig)


def to_pixels(Xh_recon, norms, means):
    """Undo center+normalize so a rebuild can be looked at as a digit."""
    return np.clip(Xh_recon * norms[:, None] + means[:, None], 0.0, 1.0)


# ------------------------------------------------------------------------ training
def snapshot_points(total, n=12):
    """Log-spaced sample counts, always including 0 and the end."""
    pts = np.unique(np.round(np.geomspace(64, total, n - 1)).astype(int))
    return [0] + [int(p) for p in pts]


def train_rule(rule, Xtr_h, probe_h, pca_basis, k, eta, epochs, seed, outdir):
    rng = np.random.default_rng(seed)
    hc = Hypercolumn(k, Xtr_h.shape[1], eta, rng, rule=rule)
    total = len(Xtr_h) * epochs
    shots = set(snapshot_points(total))
    probe_at = set(np.unique(np.round(np.geomspace(64, total, 40)).astype(int)).tolist()) | {0}

    shot_dir = outdir / f"templates_{rule}"
    shot_dir.mkdir(parents=True, exist_ok=True)
    history, seen = [], 0

    def capture(n):
        if n in shots:
            save_templates(hc.W, shot_dir / f"n{n:07d}.png",
                           f"{rule} — templates after {n:,} digits")
        if n in probe_at:
            d = diagnostics(hc.W, probe_h, pca_basis)
            d["n"] = n
            history.append(d)

    capture(0)
    t0 = time.time()
    order = rng.permutation(len(Xtr_h))
    bar = tqdm(total=total, desc=f"{rule:8s}", ncols=88, unit="digit")
    for ep in range(epochs):
        if ep:
            order = rng.permutation(len(Xtr_h))
        for idx in order:
            hc.learn_one(Xtr_h[idx])
            seen += 1
            if seen in shots or seen in probe_at:
                capture(seen)
            if seen % 2000 == 0:
                bar.update(2000)
    bar.close()
    secs = time.time() - t0
    print(f"  {rule}: {secs:.1f}s  ({1e6 * secs / max(total,1):.0f} us/digit)")
    return hc, history, secs


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=144)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eta", type=float, default=None, help="override per-rule default")
    ap.add_argument("--rules", nargs="+", default=list(RULES), choices=list(RULES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.train, args.test, args.epochs, args.tag = 2000, 500, 1, "_smoke"

    out = HERE / f"results{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xte, yte = load_mnist(args.train, args.test, args.seed)
    Xtr_h, _ = center_norm_batch(Xtr)
    Xte_h, te_norms = center_norm_batch(Xte)
    te_means = Xte.mean(axis=1)
    probe_h = Xte_h[:min(1000, len(Xte_h))]

    # Ceiling: the best 144-direction linear code for these vectors.
    print(f"train {len(Xtr)}  test {len(Xte)}  k={args.k}  epochs={args.epochs}")
    _, _, Vt = np.linalg.svd(Xtr_h, full_matrices=False)
    pca_basis = Vt[:args.k]                                       # (k, 784) orthonormal
    pca_err = float(np.linalg.norm(
        Xte_h - (Xte_h @ pca_basis.T) @ pca_basis, axis=1).mean())

    rng0 = np.random.default_rng(args.seed + 99)
    Wr = rng0.standard_normal((args.k, Xtr_h.shape[1]))
    Wr -= Wr.mean(axis=1, keepdims=True)
    Wr /= np.linalg.norm(Wr, axis=1, keepdims=True)
    rand_diag = diagnostics(Wr, probe_h, pca_basis)

    summary = {
        "config": vars(args),
        "reference": {"pca144_err": pca_err, "untrained": rand_diag},
        "rules": {},
    }
    print(f"\nbest possible (top-{args.k} subspace) error: {pca_err:.4f}")
    print(f"untrained 144 random directions:  plain {rand_diag['err_plain']:.4f}"
          f"  fitted {rand_diag['err_fitted']:.4f}\n")

    # -- the codebook control ------------------------------------------------
    cb_rng = np.random.default_rng(args.seed)
    cb = Codebook(args.k, Xtr_h.shape[1], 0.05, cb_rng)
    order = cb_rng.permutation(len(Xtr_h))
    for ep in range(args.epochs):
        if ep:
            order = cb_rng.permutation(len(Xtr_h))
        for idx in tqdm(order, desc="codebook", ncols=88, unit="digit", leave=False):
            cb.learn_one(Xtr_h[idx])
    cb_recon = cb.rebuild(Xte_h)
    cb_err = float(np.linalg.norm(Xte_h - cb_recon, axis=1).mean())
    used = int((cb.win_counts > 0).sum())
    summary["reference"]["codebook"] = {"err": cb_err, "templates_used": used}
    print(f"codebook (winner alone): error {cb_err:.4f}, {used}/{args.k} templates used")
    save_templates(cb.W, out / "templates_codebook.png",
                   f"codebook (winner-take-all) — {args.epochs} epochs")

    # -- the collective rules ------------------------------------------------
    show = np.arange(12)
    rebuild_rows = [to_pixels(Xte_h[show], te_norms[show], te_means[show])]
    rebuild_names = ["digit"]

    for rule in args.rules:
        eta = args.eta if args.eta is not None else DEFAULT_ETA[rule]
        print(f"\n--- {rule} (eta={eta}) ---")
        hc, hist, secs = train_rule(rule, Xtr_h, probe_h, pca_basis,
                                    args.k, eta, args.epochs, args.seed, out)
        final = diagnostics(hc.W, Xte_h, pca_basis)
        summary["rules"][rule] = {"eta": eta, "seconds": secs,
                                  "final": final, "history": hist}
        print(f"  plain {final['err_plain']:.4f}   gain {final['err_gain']:.4f}"
              f"   fitted {final['err_fitted']:.4f}")
        print(f"  effective rank {final['eff_rank']:.1f}/{args.k}"
              f"   overlap with best subspace {final['pca_overlap']:.3f}"
              f"   max pair similarity {final['coh_max']:.3f}"
              f"   quiet templates {100*final['quiet']:.0f}%")
        np.save(out / f"W_{rule}.npy", hc.W)
        save_templates(hc.W, out / f"templates_{rule}_final.png",
                       f"{rule} — final, {len(Xtr_h)*args.epochs:,} digits")
        S = Xte_h[show] @ hc.W.T
        rebuild_rows.append(to_pixels(S @ hc.W, te_norms[show], te_means[show]))
        rebuild_names.append(rule)

    rebuild_rows.append(to_pixels(cb_recon[show], te_norms[show], te_means[show]))
    rebuild_names.append("codebook")
    P = (Xte_h[show] @ pca_basis.T) @ pca_basis
    rebuild_rows.append(to_pixels(P, te_norms[show], te_means[show]))
    rebuild_names.append(f"best {args.k}")
    save_rebuilds(rebuild_rows, rebuild_names, out / "rebuilds.png",
                  "held-out digits rebuilt from 144 templates")

    plot_history(summary, out / "training_curves.png", pca_err, cb_err)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


def plot_history(summary, path, pca_err, cb_err):
    panels = [("err_plain", "rebuild error (scores used straight)"),
              ("err_fitted", "rebuild error (templates refitted)"),
              ("eff_rank", "effective rank of the layer"),
              ("pca_overlap", "overlap with the best 144-dim subspace"),
              ("coh_max", "largest similarity between two templates"),
              ("quiet", "fraction of near-silent templates")]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for rule, blob in summary["rules"].items():
            h = blob["history"]
            ax.plot([d["n"] for d in h], [d[key] for d in h], label=rule, lw=1.7)
        if key == "err_plain":
            ax.axhline(pca_err, ls="--", c="0.4", lw=1, label="best possible")
            ax.axhline(cb_err, ls=":", c="0.55", lw=1, label="codebook")
        if key == "err_fitted":
            ax.axhline(pca_err, ls="--", c="0.4", lw=1, label="best possible")
        ax.set_xscale("symlog", linthresh=64)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("digits seen", fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=125); plt.close(fig)


if __name__ == "__main__":
    main()
