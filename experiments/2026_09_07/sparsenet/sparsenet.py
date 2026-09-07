"""Olshausen & Field 1996 (Nature 381:607-609), their cost and their rules, on
8x8 MNIST patches.

Their two objectives and nothing else:

    E = SUM_xy [ I(x,y) - SUM_i a_i phi_i(x,y) ]^2  +  lambda SUM_i S(a_i/sigma)

with S(x) = log(1 + x^2), the form used for their Fig. 4.

Coefficients settle by their equation (5), a local network:

    adot_i = b_i - SUM_j C_ij a_j - (lambda/sigma) S'(a_i/sigma)
    b = Phi I        the feedforward match
    C = Phi Phi^T    lateral inhibition between overlapping templates

Templates learn by their equation (6), Hebb on the residual, averaged over a
block of presentations:

    dphi_i = eta < a_i [ I - Ihat ] >

plus the piece that lives only in their Fig. 4 caption: the vector length
(gain) of each template is adapted to hold every coefficient at equal variance.

The difference from our unit is the reason to run it. Nothing competes for the
patch and no single winner learns: every template with a live coefficient takes
a step, and -Ca makes them divide the patch up between them. Sparseness is
bought with a price on activity, not with an argmax.

ON lambda.  The paper states lambda/sigma = 0.14 with sigma^2 the variance of
the images. That number is not portable: under I -> cI the error term grows as
c^2 and the sparseness term as c, so the balance depends on their absolute
pixel scale, which the paper does not give. At 0.14 on unit-variance MNIST
patches the code comes out dense. So lambda is pinned by the outcome the paper
DOES report -- reconstruction error 10% of the image variance (Fig. 4 text).
Every lambda is trained the full run and the whole sweep is reported: the code
sparsifies as the basis learns, so a short calibration would pick the wrong
one.

ON DEAD TEMPLATES.  Run exactly as written, 47 of 64 templates die: the gain
rule is a death spiral for an under-used template -- low coefficient variance
shrinks its length, which shrinks its drive phi.I, so it is used less still --
and eq. (6) hands a template with no coefficient exactly zero update, so death
is absorbing. The 17 survivors then fire on ~90% of patches each and the code
is dense. The pooled histogram of their Fig. 4d hides this completely: 47
spikes at zero plus 17 broad tents pools to kurtosis 12 and reads as sparse.

Re-seeding the dead templates does bring them back, but breeds near-duplicates
(mean coherence 0.028 -> 0.137, max pair 0.91), so it is not done here -- see
unitnorm.py, where unit-length templates remove the failure mode outright
rather than patching it.

Arms: 64 templates (complete) and 128 (2x overcomplete), whitened by their
filter R(f) = f exp(-(f/f0)^4), plus a raw-patch control. MNIST is not a
natural scene, so whether their whitening helps here is a question, not a
given. One more arm swaps S(x) = log(1+x^2) for S(x) = |x| -- also one of the
three forms they report trying -- solved by soft-thresholding, which is the one
choice that puts exact zeros in the code rather than merely small values.

Usage:  uv run python sparsenet.py [--smoke]
"""

import json, sys, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
SMOKE = "--smoke" in sys.argv

PS = 8                      # patch side -- their Fig. 1 size
DIM = PS * PS
N_IMG = 2000                # a slice of MNIST, not all of it
PER_IMG = 24                # patch sites sampled per image
FLOOR = 0.10                # a patch must have this much contrast to be used

PAPER_LAM = 0.14            # their stated lambda/sigma, kept in the sweep
LAM_GRID = [0.14, 1.0, 2.0, 4.0]
TARGET_RESID = 0.10         # their reported operating point: 10% of variance
BATCH = 100                 # their "updated every 100 image presentations"
UPDATES = 1000 if not SMOKE else 40
N_SETTLE = 200              # Euler steps of eq. (5) to equilibrium
ETA = 1.0                   # learning rate of eq. (6)
VAR_GOAL, VAR_ETA, ALPHA = 0.1, 0.02, 0.02   # gain adaptation
N_REPORT = 8000             # patches the firing histograms are measured on
SEED = 7
#      mode     K    S form
ARMS = [("white", 64, "log"), ("white", 128, "log"),
        ("raw", 64, "log"), ("white", 64, "abs")]
if SMOKE:
    ARMS = [("white", 64, "log")]


# ---------------------------------------------------------------- the data

def whiten(imgs, f0_frac=200 / 512):
    """Their zero-phase whitening/lowpass filter R(f) = f exp(-(f/f0)^4).

    f0 was 200 cycles/picture on 512-pixel scenes; the same fraction of the
    sampling grid puts it at ~11 cycles/picture on 28-pixel MNIST.
    """
    n = imgs.shape[-1]
    fx = np.fft.fftfreq(n) * n
    f = np.sqrt(fx[:, None] ** 2 + fx[None, :] ** 2)
    R = f * np.exp(-((f / (f0_frac * n)) ** 4))
    return np.real(np.fft.ifft2(np.fft.fft2(imgs, axes=(-2, -1)) * R, axes=(-2, -1)))


def patches(mode, rng):
    """Sampled 8x8 patches, DC removed, blank ones dropped, unit variance."""
    d = ROOT / "data"
    X = np.load(d / "mnist/digits/train_images.npy").astype(np.float64).reshape(-1, 28, 28)
    if X.max() > 1.5:
        X /= 255.0
    X = X[rng.permutation(len(X))[:N_IMG]]
    if mode == "white":
        X = whiten(X)
    P = sliding_window_view(X, (PS, PS), axis=(1, 2)).reshape(len(X), -1, DIM)
    pick = rng.integers(0, P.shape[1], size=(len(X), PER_IMG))
    P = np.take_along_axis(P, pick[:, :, None], axis=1).reshape(-1, DIM)
    P = P - P.mean(1, keepdims=True)           # no DC: it is not what a basis is for
    keep = P.std(1) > FLOOR * P.std()
    P = P[keep]
    return P / P.std(), float(keep.mean())     # sigma = 1 after this, so lam = lam/sigma


# ------------------------------------------------------- their equation (5)

def settle(Phi, I, lam, sform="log", diag=False):
    """Coefficients by Euler integration of eq. (5) to equilibrium.

    They solved for the same equilibrium by conjugate gradient; this is the
    local-network reading of the identical dynamics. For S(x) = |x| the last
    term is not differentiable at zero, so that step is taken as the exact
    proximal one (soft-thresholding) -- the same equilibrium, and the only
    form of S that puts exact zeros in the code.
    """
    b = I @ Phi.T                              # feedforward match
    C = Phi @ Phi.T                            # lateral inhibition
    step = 1.0 / (np.linalg.eigvalsh(C)[-1] + 2.0 * lam + 1e-9)
    a = np.zeros((len(I), len(Phi)))
    for _ in range(N_SETTLE):
        if sform == "log":
            d = b - a @ C - lam * (2.0 * a) / (1.0 + a * a)   # S'(x)=2x/(1+x^2)
            a += step * d
        else:
            u = a + step * (b - a @ C)
            new = np.sign(u) * np.maximum(np.abs(u) - step * lam, 0.0)
            d = (new - a) / step
            a = new
    if diag:                                   # how far from equilibrium we stopped
        return a, float(np.abs(d).mean() / (np.abs(a).mean() + 1e-12))
    return a


def cost(Phi, I, a, lam, sform="log"):
    R = I - a @ Phi
    S = np.log1p(a * a) if sform == "log" else np.abs(a)
    return float((R * R).sum(1).mean() + lam * S.sum(1).mean())


# ---------------------------------------------------------------- training

def train(P, K, lam, rng, updates, sform="log", log=None):
    Phi = rng.standard_normal((K, DIM))
    Phi /= np.linalg.norm(Phi, axis=1, keepdims=True)
    Phi0 = Phi.copy()
    g0 = np.sqrt((P * P).sum(1).mean() / (K * VAR_GOAL))
    gain = np.full(K, g0)
    Phi *= gain[:, None]
    avar = np.full(K, VAR_GOAL)
    for u in range(updates):
        I = P[rng.integers(0, len(P), BATCH)]
        a = settle(Phi, I, lam, sform)
        R = I - a @ Phi                        # the residual
        Phi += ETA * (a.T @ R) / BATCH         # eq. (6): Hebb on the residual
        # Fig. 4 caption: gain adapted to hold equal variance on each coefficient
        av = (a * a).mean(0)
        avar = (1 - VAR_ETA) * avar + VAR_ETA * av
        gain *= (avar / VAR_GOAL) ** ALPHA
        Phi *= (gain / (np.linalg.norm(Phi, axis=1) + 1e-12))[:, None]
        if log is not None and (u % 200 == 0 or u == updates - 1):
            e = (R * R).sum(1).mean() / (I * I).sum(1).mean()
            log.append(dict(update=u, E=cost(Phi, I, a, lam, sform), resid=float(e),
                            dead=int((np.linalg.norm(Phi, axis=1) < 1e-6).sum())))
            print(f"      update {u:5d}  E {log[-1]['E']:8.3f}  "
                  f"residual {100 * e:5.1f}% of variance", flush=True)
    return Phi, Phi0 * gain.mean()


def measure(Phi, P, lam, rng, sform="log", n=N_REPORT):
    """Residual, sparseness, and every template's coefficient distribution."""
    I = P[rng.integers(0, len(P), n)]
    a, conv = settle(Phi, I, lam, sform, diag=True)
    R = I - a @ Phi
    s = a.std() + 1e-12
    thr = 0.1 * s                              # "firing" = coefficient this far from 0
    e = np.sort(a * a, 1)[:, ::-1]             # how many carry 90% of the energy
    cum = np.cumsum(e, 1) / (e.sum(1, keepdims=True) + 1e-12)
    return dict(
        A=a, sigma_a=float(s), converged=conv,
        resid=float((R * R).sum(1).mean() / (I * I).sum(1).mean()),
        kurtosis=float(((a / s) ** 4).mean() - 3.0),
        n90=float((cum < 0.9).sum(1).mean() + 1),
        active=float((np.abs(a) > thr).mean() * a.shape[1]),
        dead=int((np.abs(a).max(0) == 0).sum()),
        exact_zeros=float((a == 0).mean()),
        median_template_kurtosis=float(np.median(
            [((a[:, i] / (a[:, i].std() + 1e-12)) ** 4).mean() - 3.0
             for i in range(a.shape[1]) if a[:, i].std() > 1e-12] or [np.nan])),
        per_template=[dict(
            i=int(i), std=float(a[:, i].std()),
            active_rate=float((np.abs(a[:, i]) > thr).mean()),
            kurtosis=float(((a[:, i] / (a[:, i].std() + 1e-12)) ** 4).mean() - 3.0),
            mean_when_on=float(np.abs(a[:, i])[np.abs(a[:, i]) > thr].mean())
            if (np.abs(a[:, i]) > thr).any() else 0.0,
            length=float(np.linalg.norm(Phi[i]))) for i in range(a.shape[1])])


# ----------------------------------------------------------------- drawing

def tile(Phi, order):
    """Each template scaled to fill the greyscale, zero always the same grey."""
    n = int(np.ceil(np.sqrt(len(Phi))))
    canvas = np.full((n * (PS + 1) + 1, n * (PS + 1) + 1), np.nan)
    for j, i in enumerate(order):
        t = Phi[i] / (np.abs(Phi[i]).max() + 1e-12)
        r, c = divmod(j, n)
        canvas[r * (PS + 1) + 1:r * (PS + 1) + 1 + PS,
               c * (PS + 1) + 1:c * (PS + 1) + 1 + PS] = t.reshape(PS, PS)
    return canvas


def grid(ax, Phi, order, title):
    ax.imshow(tile(Phi, order), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9); ax.axis("off")


def firing_figure(Phi, A, stats, order, name, m, path):
    """Every template beside its own coefficient histogram over the patches.

    Shared bins on a_i / sigma_a, log counts, so the shape of one template's
    firing can be read against another's.
    """
    K = len(Phi); n = int(np.ceil(np.sqrt(K)))
    s = A.std(); bins = np.linspace(-6, 6, 61)
    fig = plt.figure(figsize=(2.0 * n, 2.3 * n))
    gs = fig.add_gridspec(2 * n, n, height_ratios=[0.55, 1] * n,
                          hspace=0.15, wspace=0.15)
    for j, i in enumerate(order):
        r, c = divmod(j, n)
        ax = fig.add_subplot(gs[2 * r, c])
        ax.imshow(Phi[i].reshape(PS, PS) / (np.abs(Phi[i]).max() + 1e-12),
                  cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"#{i}  fires {100 * stats[i]['active_rate']:.0f}%",
                     fontsize=6, pad=1.5); ax.axis("off")
        ax = fig.add_subplot(gs[2 * r + 1, c])
        ax.hist(A[:, i] / s, bins=bins, log=True, color="0.25")
        ax.axvline(0, color="crimson", lw=0.5)
        ax.set_ylim(0.7, len(A)); ax.set_xlim(-6, 6)
        ax.tick_params(labelsize=5, length=1.5)
        if c: ax.set_yticklabels([])
        if r < n - 1: ax.set_xticklabels([])
        else: ax.set_xlabel("$a_i/\\sigma_a$", fontsize=6)
    fig.suptitle(f"how each template fires over the training patches -- {name}\n"
                 f"shared bins, log counts; ordered by coefficient variance; "
                 f"'fires' = $|a_i| > 0.1\\sigma_a$\n"
                 f"median per-template kurtosis "
                 f"{m['median_template_kurtosis']:.1f} "
                 f"(pooled over all templates it reads as "
                 f"{m['kurtosis']:.1f}); {m['dead']} of {len(Phi)} never fire",
                 fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------- main

def main():
    t0 = time.time(); results = {}
    print(f"Olshausen & Field 1996 on {PS}x{PS} MNIST patches from {N_IMG} images, "
          f"{UPDATES} updates of {BATCH}\n")
    cache = {}
    for mode, K, sform in ARMS:
        stem = f"{mode}_{K}" + ("" if sform == "log" else "_abs")
        print(f"  {stem}  (S = {'log(1+x^2)' if sform == 'log' else '|x|'}):")
        if mode not in cache:
            cache[mode] = patches(mode, np.random.default_rng(SEED))
        P, kept = cache[mode]
        print(f"      {len(P)} patches ({100 * kept:.0f}% of sites had contrast)")

        # lambda pinned by their reported 10%-of-variance operating point; each
        # candidate is a full run, and the winner's basis is the one kept
        sweep, runs = [], {}
        for lam in LAM_GRID:
            log = []
            Phi, Phi0 = train(P, K, lam, np.random.default_rng(SEED + K),
                              UPDATES, sform, log)
            m = measure(Phi, P, lam, np.random.default_rng(1), sform, n=4000)
            runs[lam] = (Phi, Phi0, log)
            sweep.append(dict(lam=lam, resid=m["resid"], active=m["active"],
                              n90=m["n90"], kurtosis=m["kurtosis"], dead=m["dead"],
                              per_template_kurtosis=m["median_template_kurtosis"]))
            print(f"      lam {lam:6.2f}  residual {100 * m['resid']:5.1f}%  "
                  f"{m['active']:5.1f} of {K} active  {m['n90']:4.1f} hold 90%  "
                  f"pooled kurt {m['kurtosis']:6.1f}  per-template kurt "
                  f"{m['median_template_kurtosis']:5.1f}  dead {m['dead']}", flush=True)
        lam = min(sweep, key=lambda r: abs(r["resid"] - TARGET_RESID))["lam"]
        print(f"      -> lambda/sigma = {lam} (closest to {100*TARGET_RESID:.0f}% residual)")

        for _ in (0,):
            name = stem
            Phi, Phi0, log = runs[lam]
            m = measure(Phi, P, lam, np.random.default_rng(1), sform)
            m0 = measure(Phi0, P, lam, np.random.default_rng(1), sform, n=4000)
            A, A0, stats = m.pop("A"), m0.pop("A"), m["per_template"]
            order = np.argsort([-t["std"] for t in stats])
            np.save(OUT / f"phi_{name}.npy", Phi)                       # the templates
            np.save(OUT / f"coeff_{name}.npy", A.astype(np.float32))    # their firing
            results[name] = dict(lam=lam, sform=sform, sweep=sweep, log=log,
                                 n_patches=len(P), kept=kept,
                                 learned=m, random_init=m0)
            print(f"      residual {100 * m['resid']:.1f}%, "
                  f"{m['dead']} dead, {m['active']:.1f} of {K} active, "
                  f"{m['n90']:.1f} hold 90%, pooled kurtosis {m['kurtosis']:.1f}, "
                  f"per-template kurtosis {m['median_template_kurtosis']:.1f}, "
                  f"{100 * m['exact_zeros']:.0f}% exact zeros", flush=True)

            fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))
            grid(ax[0], Phi, order, f"learned basis  (K={K}, $\\lambda/\\sigma$={lam})")
            Cv = np.cov(P.T); w, V = np.linalg.eigh(Cv)   # their Fig. 1, same patches
            PC = (V[:, ::-1].T)[:K]
            grid(ax[1], PC, np.arange(len(PC)),
                 "principal components of the same patches")
            for a_, ls, lab in ((A, "-", "learned"), (A0, "--", "random init")):
                ax[2].hist(a_.ravel() / a_.std(), bins=200, range=(-4, 4), density=True,
                           histtype="step", ls=ls, log=True, label=lab)
            ax[2].set_xlabel("$a_i$"); ax[2].set_ylabel("$P(a_i)$")
            ax[2].legend(fontsize=8)
            ax[2].set_title(f"pooled over all templates (their Fig. 4d)\n"
                            f"kurtosis {m['kurtosis']:.0f} vs {m0['kurtosis']:.0f} "
                            f"at init -- but per template only "
                            f"{m['median_template_kurtosis']:.0f}", fontsize=8)
            fig.suptitle(f"Olshausen & Field 1996 on 8x8 MNIST -- {mode}, K={K}, "
                         f"S={'log(1+x^2)' if sform == 'log' else '|x|'} -- "
                         f"residual {100 * m['resid']:.1f}% of variance, "
                         f"{m['dead']} dead templates, {m['active']:.1f} active "
                         f"per patch", fontsize=11)
            fig.tight_layout()
            fig.savefig(OUT / f"basis_{name}.png", dpi=140); plt.close(fig)

            firing_figure(Phi, A, stats, order, name, m, OUT / f"firing_{name}.png")

            fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
            ax[0].bar(np.arange(K), [100 * stats[i]["active_rate"] for i in order],
                      color="0.3")
            ax[0].set_xlabel("template (by coefficient variance)")
            ax[0].set_ylabel("% of patches it fires on")
            ax[1].bar(np.arange(K), [stats[i]["kurtosis"] for i in order], color="0.3")
            ax[1].set_xlabel("template (by coefficient variance)")
            ax[1].set_ylabel("lifetime kurtosis of $a_i$")
            fig.suptitle(f"per-template firing -- {name}", fontsize=11)
            fig.tight_layout(); fig.savefig(OUT / f"usage_{name}.png", dpi=140)
            plt.close(fig)

    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n  {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
