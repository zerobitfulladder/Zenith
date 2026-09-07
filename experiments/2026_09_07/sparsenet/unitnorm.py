"""Olshausen & Field's objective, rebuilt on our conventions: unit-length
patches, unit-length templates, and learning that only ever ROTATES a template.

What is kept from the paper: the cost, and the settling that finds the code.

    E = || I - SUM_i a_i phi_i ||^2  +  lambda SUM_i S(a_i)
    adot_i = b_i - a_i - SUM_{j!=i} C_ij a_j - lambda S'(a_i)

With ||phi_i|| = 1 the Gram diagonal is exactly 1, so eq. (5) separates into a
common leak plus inhibition in proportion to overlap. S = |x| is solved by the
exact proximal step (soft-thresholding), which is the only form that puts real
zeros in the code -- on MNIST it is worth ~50x the per-template kurtosis of
log(1+x^2) at equal reconstruction error, so it is the default here.

What is dropped: the gain. Their Fig. 4 caption adapts each template's vector
length to equalise coefficient variance, and that is a death spiral -- an
unused template is shortened, so it is driven less, so it is used less, and
eq. (6) hands a template with no coefficient exactly zero update. 46 of 64 die.
Fixing that by re-seeding dead templates works but breeds near-duplicates
(mean coherence 0.028 -> 0.137), so it is not fixed that way here: the length
is simply not a free parameter. Templates rotate on the unit sphere and
nothing can shrink out of the competition.

What is added: a per-template LIFETIME objective. Their eq. (6) carries no
sparseness term, and not by oversight -- at the equilibrium of eq. (5) the
envelope theorem makes the reconstruction gradient the whole gradient, so
within their per-patch cost the sparsity pressure on templates is already
exhausted. To push a single template's own firing distribution you have to
leave the per-patch cost for a lifetime one, over that template's coefficients
across the dataset:

    minimize  SUM_i H(a_i)   subject to reconstruction staying low

which is ICA with the -log|det W| traded for the reconstruction constraint --
so it never asks the templates to be orthogonal, only to be independent.
Three terms, each an average over the batch, tangential to the sphere:

    g_i  =      < a_i r >                       r = I - Ihat, the paper's term
             -  beta  < S'(p_i/sig_i) I > /sig_i    sharpen its own histogram
             -  gamma < |p_j|_{j!=i} sign(p_i) I >  fire when the others do not

    g_i <- g_i - (g_i.phi_i) phi_i               rotate, never rescale
    phi_i <- (phi_i + eta g_i) / || . ||

Two details decide whether this works. The lifetime terms use the RAW
projection p_i = phi_i . I, never the settled coefficient: a settled
coefficient is exactly zero most of the time, which is precisely the gating
that made templates die, while a projection is never zero, so every template
has a gradient on every patch whether or not it won anything. And sig_i, a
running RMS of p_i, is not optional -- without it "minimise E[S(p_i)] at unit
norm" is solved by rotating to the lowest-variance direction in the data, and
the template goes quiet instead of getting sparse.

The three gradients are each scaled to unit norm before they are combined, so
beta and gamma are relative weights rather than functions of the data scale.

Arms separate the terms:  recon (the paper under unit norm) | +sparsity |
+independence | both.

Usage:  uv run python unitnorm.py [--smoke]
"""

import json, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sparsenet as S

OUT = S.OUT / "unitnorm"; OUT.mkdir(exist_ok=True)
SMOKE = S.SMOKE

K = 64
SFORM = "abs"
LAM_GRID = [0.01, 0.03, 0.06, 0.12]
TARGET_RESID = 0.10
UPDATES = 1000 if not SMOKE else 40
BATCH = S.BATCH
ETA = 0.05                  # step on the sphere
VAR_ETA = 0.02              # running estimate of each template's sig_i
SEED = 7
#       name              beta  gamma
ARMS = [("recon", 0.0, 0.0),
        ("sparsity", 0.3, 0.0),
        ("sparsity_hi", 1.0, 0.0),
        ("independence", 0.0, 0.3),
        ("both", 0.3, 0.3)]
if SMOKE:
    ARMS = ARMS[:2]


def prep(mode):
    """Their patches, then onto the unit sphere -- the contrast gate first.

    The gate has to come before the normalisation: a near-blank patch scaled to
    unit length is amplified noise at full strength, and the dictionary would
    learn it.
    """
    P, kept = S.patches(mode, np.random.default_rng(SEED))
    return P / np.linalg.norm(P, axis=1, keepdims=True), kept


def dS(u, sform):
    return np.sign(u) if sform == "abs" else 2.0 * u / (1.0 + u * u)


def unit(G):
    return G / (np.linalg.norm(G) + 1e-12)


def train(P, lam, beta, gamma, rng, updates, sform=SFORM, log=None):
    Phi = rng.standard_normal((K, S.DIM))
    Phi /= np.linalg.norm(Phi, axis=1, keepdims=True)
    Phi0 = Phi.copy()
    sig2 = np.ones(K)
    for u in range(updates):
        I = P[rng.integers(0, len(P), BATCH)]
        a = S.settle(Phi, I, lam, sform)
        R = I - a @ Phi
        g = unit((a.T @ R) / BATCH)                 # explain what is left over
        if beta or gamma:
            p = I @ Phi.T                           # the RAW projection: never zero
            sig2 = (1 - VAR_ETA) * sig2 + VAR_ETA * (p * p).mean(0)
            sig = np.sqrt(sig2) + 1e-9
        if beta:                                    # sharpen its own histogram
            g = g - beta * unit(((dS(p / sig, sform) / sig).T @ I) / BATCH)
        if gamma:                                   # fire when the others do not
            m = np.abs(p)
            others = (m.sum(1, keepdims=True) - m) / (K - 1)
            g = g - gamma * unit(((np.sign(p) * others).T @ I) / BATCH)
        g = g - (g * Phi).sum(1, keepdims=True) * Phi        # tangential only
        Phi = Phi + ETA * g
        Phi /= np.linalg.norm(Phi, axis=1, keepdims=True)    # rotation, not rescale
        if log is not None and (u % 200 == 0 or u == updates - 1):
            e = (R * R).sum(1).mean() / (I * I).sum(1).mean()
            log.append(dict(update=u, resid=float(e),
                            E=S.cost(Phi, I, a, lam, sform)))
            print(f"      update {u:5d}  residual {100 * e:5.1f}% of variance",
                  flush=True)
    return Phi, Phi0


def coherence(Phi):
    live = np.linalg.norm(Phi, axis=1) > 1e-6
    U = Phi[live] / np.linalg.norm(Phi[live], axis=1, keepdims=True)
    G = np.abs(U @ U.T); np.fill_diagonal(G, 0)
    iu = np.triu_indices(len(U), 1)
    return float(G[iu].mean()), float(G.max())


def main():
    t0 = time.time(); results = {}
    P, kept = prep("white")
    print(f"unit-length {S.PS}x{S.PS} MNIST patches, {len(P)} of them "
          f"({100 * kept:.0f}% of sites passed the contrast gate)")
    print(f"K={K} unit-length templates, S={'|x|' if SFORM == 'abs' else 'log(1+x^2)'}, "
          f"rotation-only learning, {UPDATES} updates of {BATCH}\n")

    print("  lambda, on the recon-only arm:")
    sweep = []
    for lam in LAM_GRID:
        Phi, _ = train(P, lam, 0.0, 0.0, np.random.default_rng(SEED), UPDATES)
        m = S.measure(Phi, P, lam, np.random.default_rng(1), SFORM, n=4000)
        m.pop("A")
        sweep.append(dict(lam=lam, **{k: m[k] for k in
                     ("resid", "active", "n90", "kurtosis",
                      "median_template_kurtosis", "dead")}))
        print(f"      lam {lam:7.3f}  residual {100 * m['resid']:5.1f}%  "
              f"{m['active']:5.1f} of {K} active  {m['n90']:4.1f} hold 90%  "
              f"per-template kurt {m['median_template_kurtosis']:7.1f}  "
              f"dead {m['dead']}", flush=True)
    lam = min(sweep, key=lambda r: abs(r["resid"] - TARGET_RESID))["lam"]
    print(f"      -> lambda = {lam}\n")

    for name, beta, gamma in ARMS:
        print(f"  {name}  (beta={beta}, gamma={gamma}):")
        log = []
        Phi, Phi0 = train(P, lam, beta, gamma, np.random.default_rng(SEED),
                          UPDATES, log=log)
        m = S.measure(Phi, P, lam, np.random.default_rng(1), SFORM)
        A, stats = m.pop("A"), m["per_template"]
        cm, cx = coherence(Phi)
        order = np.argsort([-t["std"] for t in stats])
        np.save(OUT / f"phi_{name}.npy", Phi)
        np.save(OUT / f"coeff_{name}.npy", A.astype(np.float32))
        results[name] = dict(lam=lam, beta=beta, gamma=gamma, sweep=sweep, log=log,
                             coherence_mean=cm, coherence_max=cx, learned=m)
        print(f"      residual {100 * m['resid']:.1f}%, {m['dead']} dead, "
              f"{m['active']:.1f} of {K} active, {m['n90']:.1f} hold 90%, "
              f"per-template kurtosis {m['median_template_kurtosis']:.1f}, "
              f"pooled {m['kurtosis']:.1f}, {100 * m['exact_zeros']:.0f}% exact "
              f"zeros, coherence {cm:.3f} mean / {cx:.3f} max", flush=True)

        fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.6))
        S.grid(ax[0], Phi, order, f"{name}  ($\\beta$={beta}, $\\gamma$={gamma})")
        ax[1].hist(A.ravel() / A.std(), bins=200, range=(-4, 4), density=True,
                   histtype="step", log=True)
        ax[1].set_xlabel("$a_i$"); ax[1].set_ylabel("$P(a_i)$")
        ax[1].set_title(f"per-template kurtosis "
                        f"{m['median_template_kurtosis']:.0f}", fontsize=9)
        fig.suptitle(f"unit-length patches and templates, rotation-only -- {name}: "
                     f"residual {100 * m['resid']:.1f}%, {m['active']:.1f} active, "
                     f"coherence {cm:.3f}", fontsize=10)
        fig.tight_layout(); fig.savefig(OUT / f"basis_{name}.png", dpi=140)
        plt.close(fig)
        S.firing_figure(Phi, A, stats, order, name, m, OUT / f"firing_{name}.png")

    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"\n  {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
