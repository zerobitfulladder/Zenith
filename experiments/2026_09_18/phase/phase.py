"""Angles on the ring: can prediction error move codes?   python phase.py

The smallest test of "make blocks continuous, treat the search trajectory as virtual layers,
push prediction error back along it as a rotation on angles".  No library, no chunks, no
search -- only the substrate and one prediction.

Substrate.  A code is B angles, one per block (a point on the B-torus).  Binding is adding
angles; bundling is adding unit phasors, so a bundle has a direction (the mean) and a length
(the agreement).  Matching is the real part of the inner product, i.e. the length-weighted
mean cosine -- smooth, unlike the presence test the rest of the repo uses.

Task (self-supervised, III.6).  Leave one inked cell out of a digit; every other cell votes
for what is at the missing one through the relation code of its relative offset:

    Z = mean_j exp(i * (theta[stroke_j] + rho[offset_j]))          <- the pile
    prediction = argmax_v  Re( Z . conj(exp(i*theta[v])) )         <- cleanup

Learning.  Because binding is addition, the derivative of the match with respect to every
angle in the chain is the same scalar, +-sin(disagreement).  So the backward pass is the walk
run in reverse: each participant rotates a little.  The loss is -log softmax of the match
score over the vocabulary, so the true code is pulled onto the consensus and every other code
is pushed off it in proportion to how much it is currently believed.  That competitive term is
load-bearing: pushing only the single wrong winner away collapses all the codes onto one angle
within 500 digits (all-pairs similarity 0.00 -> 0.48, effective vocabulary 64 -> 37).

    delta_jvb = theta[s_j,b] + rho[d_j,b] - theta[v,b]
    score_v   = mean_j sum_b cos(delta_jvb)          p = softmax(beta * score)
    theta[v]  -= lr * (p_v - [v==true]) * mean_j sin(delta_jvb)
    theta[s_j], rho[d_j] -= lr * ( sin(delta_j,true) - sum_v p_v sin(delta_jv) )

Three questions, and they are the whole point:
  1. does held-out cell prediction improve over frozen random codes, and how close does it get
     to an explicit count table doing the same job with ~70x more numbers?
  2. do codes of strokes with the same outcome converge, and codes of different outcomes not?
     (Part III's claim, for the first time with a live signal rather than a one-shot hashing)
  3. do the codes collapse?

Measured (DESIGN.md XVII): 49.0% at 32 blocks against 53.5% for the count table and 1.5% for
frozen codes, floor 2.8%; same-class code similarity +0.091 against +0.056 for different
classes, with the label never shown; no collapse.
"""
import argparse, math, time
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2026_09_16" / "glimpse_loop"))   # sdr.py, mnist.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2026_09_17" / "tally"))          # tally3.py
import mnist, tally3


# ---------------------------------------------------------------- vocabulary of strokes

def coalesce(means, counts, K, rng, iters=25):
    """Weighted Lloyd's over the allocated prototypes: 400 near-duplicates -> K columns with mass."""
    p = counts / counts.sum()
    c = means[rng.choice(len(means), K, replace=False, p=p)].copy()
    for _ in range(iters):
        a = ((means[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                c[k] = (means[m] * counts[m, None]).sum(0) / counts[m].sum()
    a = ((means[:, None] - c[None]) ** 2).sum(-1).argmin(1)
    return c, np.array([counts[a == k].sum() for k in range(K)])


def build_vocab(X, rng, ps, stride, K):
    scale = math.sqrt(ps * ps / 9.0)
    voc = tally3.Vocab(0.9 * scale, ink=2)
    for i in rng.permutation(len(X))[:3000]:
        for p in tally3.patches(X[i]):
            voc.learn(p)
    voc.finalize(min_count=20)
    raw = len(voc.means)
    if K and K < raw:
        voc.means, voc.counts = coalesce(voc.means, voc.counts, K, rng)
    return voc, raw


# ---------------------------------------------------------------- geometry

def neighbourhood(ng, radius):
    """For every cell: the in-grid neighbours within the radius, and the offset id of each."""
    ds = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1) if (dr, dc) != (0, 0)]
    oid = {d: i for i, d in enumerate(ds)}
    nb_cell, nb_off = [], []
    for r in range(ng):
        for c in range(ng):
            cc, oo = [], []
            for (dr, dc) in ds:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < ng and 0 <= c2 < ng:
                    cc.append(r2 * ng + c2); oo.append(oid[(dr, dc)])
            nb_cell.append(np.array(cc)); nb_off.append(np.array(oo))
    return ds, nb_cell, nb_off


def context(grid, cell, nb_cell, nb_off, blank, keep_blanks, hidden=None):
    """Strokes and offsets of everything visible around `cell`.  -1 (no ink) becomes the blank code."""
    cells, offs = nb_cell[cell], nb_off[cell]
    s = grid[cells]
    if hidden is not None:
        m = ~hidden[cells]
        s, offs = s[m], offs[m]
    s = np.where(s < 0, blank, s)
    if not keep_blanks:
        m = s != blank
        s, offs = s[m], offs[m]
    return s, offs


# ---------------------------------------------------------------- the algebra

def scatter(arr, idx, upd):
    """Add `upd` to `arr` at `idx`, averaging the rows that land on the same parameter, so a
    code appearing ten times in one context takes one step rather than ten."""
    cnt = np.bincount(idx, minlength=len(arr))
    acc = np.zeros_like(arr); np.add.at(acc, idx, upd)
    nz = cnt > 0
    arr[nz] += acc[nz] / cnt[nz, None]


def harmonics(m, kind):
    """Weights of the m Fourier coefficients kept per block.  The score of a candidate is the
    pile's distribution over the ring, blurred to m waves, evaluated at the candidate's angle:
    m=1 is one arrow (mean direction and spread only, cannot hold two hypotheses), larger m
    resolves finer structure, m ~ L/2 is the exact slot histogram.  Fejer weights keep the
    implied kernel non-negative and damp the high end, which the k factor in the gradient
    otherwise makes dominate."""
    k = np.arange(1, m + 1, dtype=float)
    w = {"flat": np.ones(m), "fejer": 1 - k / (m + 1), "inv": 1 / k}[kind]
    return k, w / w.sum()


def pile(theta, rho, s, o):
    """The bundle: one bound item per visible neighbour.  Returned as angles, so any harmonic
    can be taken from it -- the k-th coefficient is mean_j exp(i*k*ang_j)."""
    return theta[s] + rho[o]


def score_all(ang, theta, V, k, w):
    """Match of every candidate against the pile, plus |Z_1| (the agreement of the voters).

    score_v = sum_b sum_k w_k * mean_j cos(k * (ang_jb - theta_vb))
    """
    dl = ang[:, None, :] - theta[:V][None, :, :]                  # (J, V, B)
    kd = k[:, None, None, None] * dl[None]                        # (m, J, V, B)
    sc = np.einsum("m,mjvb->v", w, np.cos(kd)) / len(ang)
    return sc, dl, kd, np.abs(np.exp(1j * ang).mean(0)).mean()


def grad_factor(kd, k, w):
    """d/d(angle) of the score is sum_k w_k * k * sin(k * disagreement) -- one number per block,
    the same one handed to every angle in the chain, because binding is addition."""
    return np.einsum("m,mjvb->jvb", w * k, np.sin(kd))


def step(theta, rho, s, o, true, lr, beta, V, k, w, move_codes=True, move_rel=True, lrnorm=False):
    """Rotate every angle that took part.

    The loss is -log softmax(beta * score) at the true stroke: the true code is pulled onto the
    consensus, every other code is pushed off it in proportion to how much it is currently
    believed (the term that stops all the codes sliding onto one angle), and each contributor
    and each relation is pulled toward what would have made the truth win.
    """
    ang = pile(theta, rho, s, o)
    sc, dl, kd, _ = score_all(ang, theta, V, k, w)
    G = grad_factor(kd, k, w)                                     # (J, V, B)
    if lrnorm:
        G = G / float((w * k).sum())                              # matched step size across m
    p = np.exp(beta * (sc - sc.max())); p /= p.sum()
    dscore = p.copy(); dscore[true] -= 1.0

    if move_codes:
        theta[:V] -= lr * dscore[:, None] * G.sum(0) / len(s)
    gj = G[:, true, :] - np.einsum("v,jvb->jb", p, G)
    if move_codes:
        scatter(theta, s, -lr * gj)
    if move_rel:
        scatter(rho, o, -lr * gj)


def capacity(rng, V=2000, Bs=(16,), Ms=(1, 2, 4, 8, 16), Ns=(2, 4, 8, 16, 32, 64), trials=40, hw="fejer"):
    """Bundling capacity with no learning anywhere: draw V random codes, bundle N of them, score
    every candidate against the pile, and ask how many of the top N are really in it.

    This isolates the representational question from the optimization one.  If m harmonics carry
    more items than one arrow, higher m is a better pile whatever the learning did, because
    crosstalk in different harmonics of random codes is close to independent: signal stays flat
    while noise should fall like 1/sqrt(m).
    """
    print(f"\nBundling capacity, no learning: {V} random codes, precision@N over {trials} trials\n")
    for B in Bs:
        print(f"  B = {B} blocks" + "".join(f"{('m=' + str(m)):>10}" for m in Ms))
        for N in Ns:
            row = f"  N = {N:<3} items "
            for m in Ms:
                k, w = harmonics(m, hw)
                hit = 0
                for _ in range(trials):
                    th = rng.uniform(0, 2 * np.pi, (V, B))
                    mem = rng.choice(V, N, replace=False)
                    kd = k[:, None, None] * th[mem][None]                    # (m, N, B)
                    Z = np.exp(1j * kd).mean(1)                              # (m, B)
                    E = np.exp(-1j * k[:, None, None] * th[None])            # (m, V, B)
                    sc = np.einsum("m,mb,mvb->v", w, Z, E).real
                    top = np.argpartition(-sc, N)[:N]
                    hit += len(set(top.tolist()) & set(mem.tolist())) / N
                row += f"{hit / trials:>10.3f}"
            print(row)
        print()


# ---------------------------------------------------------------- evaluation

def make_eval(grids, rng, n_targets, nb_cell, nb_off, blank, keep_blanks):
    """Fixed leave-one-out targets so the curve is comparable at every report point."""
    ev = []
    for g in grids:
        ink = np.flatnonzero(g >= 0)
        if len(ink) < 4:
            continue
        for c in rng.choice(ink, min(n_targets, len(ink)), replace=False):
            hid = np.zeros(len(g), bool); hid[c] = True
            s, o = context(g, c, nb_cell, nb_off, blank, keep_blanks, hid)
            if len(s):
                ev.append((int(g[c]), s, o))
    return ev


def evaluate(ev, theta, rho, V, near, k, w, kf=None, wf=None, coarse=0, recall=(1, 3, 5, 10)):
    """Score every candidate with the (k, w) kernel.  With `coarse` > 0 this is only the first
    pass: the top `coarse` candidates are then rescored with the sharper (kf, wf) kernel and the
    winner is taken from those.  Blurry to propose, sharp to choose — the same shape as the
    loop's own hypothesise-then-verify step, with m as the dial."""
    hit = nearhit = 0; preds = set(); ok, mag, marg = [], [], []
    rec = {r: 0 for r in recall}
    for true, s, o in ev:
        ang = pile(theta, rho, s, o)
        sc, _, _, z1 = score_all(ang, theta, V, k, w)
        order = np.argsort(sc)
        for r in recall:
            rec[r] += int(true in order[-r:])
        if coarse and kf is not None:
            cand = order[-coarse:]
            sf, _, _, _ = score_all(ang, theta[cand], len(cand), kf, wf)
            sc = np.full(V, -1e9); sc[cand] = sf
            order = np.argsort(sc)
        p = int(order[-1]); preds.add(p)
        r = p == true
        hit += r; nearhit += near[true, p]
        ok.append(r); mag.append(z1); marg.append(float(sc[order[-1]] - sc[order[-2]]))
    ok = np.array(ok); n = len(ev)

    def split(c):
        """accuracy among the most- and least-confident quarter, by this confidence measure"""
        i = np.argsort(np.array(c)); q = max(1, n // 4)
        return ok[i[-q:]].mean(), ok[i[:q]].mean()
    hi_m, lo_m = split(mag); hi_g, lo_g = split(marg)
    return dict(acc=hit / n, near=nearhit / n, distinct=len(preds),
                **{('rec%d' % r): rec[r] / n for r in recall},
                mag_ok=float(np.mean(np.array(mag)[ok])) if ok.any() else 0.0,
                mag_no=float(np.mean(np.array(mag)[~ok])) if (~ok).any() else 0.0,
                hi_mag=hi_m, lo_mag=lo_m, hi_marg=hi_g, lo_marg=lo_g)


def eval_counts(ev, CNT, alpha=0.5):
    hit = 0
    for true, s, o in ev:
        hit += int(np.log(CNT[o, s] + alpha).sum(0).argmax()) == true
    return hit / len(ev)


def similarity(theta, V, B):
    Zv = np.exp(1j * theta[:V])
    return (Zv @ Zv.conj().T).real / B


# ---------------------------------------------------------------- run

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=5000); ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--ps", type=int, default=5); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=64, help="coalesce the allocated prototypes to K columns (0 = keep all)")
    ap.add_argument("--blocks", type=int, default=16); ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--harm", type=int, default=1, help="Fourier coefficients kept per pile block (1 = one arrow)")
    ap.add_argument("--hw", default="fejer", choices=["flat", "fejer", "inv"], help="harmonic weighting")
    ap.add_argument("--read-harm", type=int, default=0, help="evaluate with a different m than training (0 = same)")
    ap.add_argument("--matrix", action="store_true", help="at the end, sweep shortlist size against read-time m")
    ap.add_argument("--coarse", type=int, default=0, help="propose with the blurry kernel, then rescore the top K with the sharp one")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=1.0, help="inverse temperature of the contrastive term")
    ap.add_argument("--decay", type=float, default=0.1, help="lr at the end of training, as a fraction of the start")
    ap.add_argument("--targets", type=int, default=6); ap.add_argument("--eval-targets", type=int, default=8)
    ap.add_argument("--blanks", action="store_true", help="let blank neighbours vote too")
    ap.add_argument("--freeze-codes", action="store_true"); ap.add_argument("--freeze-rel", action="store_true")
    ap.add_argument("--report", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capacity", action="store_true", help="bundling capacity vs m, no learning; then exit")
    ap.add_argument("--lrnorm", action="store_true", help="divide the step by the mean harmonic, so every m trains at a matched step size")
    args = ap.parse_args(argv)

    t0 = time.time(); rng = np.random.default_rng(args.seed)
    if args.capacity:
        capacity(rng); return
    tally3.configure(args.ps, args.stride)
    NG, NC = tally3.NG, tally3.NC
    X, y = mnist.load("train"); Xt, yt = mnist.load("t10k")
    sel = rng.permutation(len(X))[:args.train]; X, y = X[sel], y[sel]
    Xt, yt = Xt[:args.test], yt[:args.test]

    voc, raw = build_vocab(X, rng, args.ps, args.stride, args.vocab)
    V = len(voc.means); B = args.blocks; BLANK = V
    grids, grids_t = voc.grids(X), voc.grids(Xt)
    ds, nb_cell, nb_off = neighbourhood(NG, args.radius); ND = len(ds)
    keep_blanks = args.blanks

    # "near" = the leniency the record uses elsewhere: right if it lands on a look-alike stroke
    d = np.sqrt(((voc.means[:, None] - voc.means[None]) ** 2).sum(-1))
    near = d < 0.9 * math.sqrt(args.ps * args.ps / 9.0)
    np.fill_diagonal(near, True)

    # dominant class of each stroke, for the Part III measurement only
    lab = np.zeros((V, 10), int)
    for g, c in zip(grids, y):
        for s_ in np.unique(g[g >= 0]):
            lab[s_, c] += 1
    dom = lab.argmax(1)
    same = dom[:, None] == dom[None, :]; np.fill_diagonal(same, False)
    diff = ~(dom[:, None] == dom[None, :])

    kk, hw = harmonics(args.harm, args.hw)
    kr, hr = harmonics(args.read_harm or args.harm, args.hw)
    theta = rng.uniform(0, 2 * np.pi, (V + 1, B))
    rho = rng.uniform(0, 2 * np.pi, (ND, B))
    theta0, rho0 = theta.copy(), rho.copy()
    CNT = np.zeros((ND, V + 1, V), np.int32)

    ev = make_eval(grids_t, np.random.default_rng(1), args.eval_targets, nb_cell, nb_off, BLANK, keep_blanks)

    n_ang = (V + 1) * B + ND * B
    print(f"{raw} prototypes coalesced to {V} strokes, {args.ps}x{args.ps} stride {args.stride} "
          f"({NC} cells), {B} blocks, {ND} offsets within radius {args.radius}, "
          f"{'blanks vote' if keep_blanks else 'inked only'}, {args.harm} harmonic(s) [{args.hw}]")
    print(f"angles: {n_ang:,} numbers    count table: {CNT.size:,} numbers    "
          f"eval: {len(ev):,} held-out cells from {args.test} unseen digits    [{time.time()-t0:.0f}s]")

    # trivial floors
    freq = np.bincount(np.concatenate([g[g >= 0] for g in grids]), minlength=V)
    floor = np.mean([t == freq.argmax() for t, _, _ in ev])
    print(f"floor (always the commonest stroke): {floor:.3f}\n")
    print("  digits |  learned  frozen  counting |   near  distinct |  same-class  diff-class  all-pairs |"
          "  |Z| hi/lo quartile |  margin hi/lo")

    def report(n):
        m = evaluate(ev, theta, rho, V, near, kk, hw, kr, hr, args.coarse)
        f = evaluate(ev, theta0, rho0, V, near, kk, hw, kr, hr, args.coarse)
        cb = eval_counts(ev, CNT) if CNT.sum() else 0.0
        S = similarity(theta, V, B)
        print(f"{n:8,} |   {m['acc']:.3f}   {f['acc']:.3f}    {cb:.3f}   |  {m['near']:.3f}    {m['distinct']:3d}    |"
              f"    {S[same].mean():+.3f}      {S[diff].mean():+.3f}     {S[~np.eye(V,dtype=bool)].mean():+.3f}  |"
              f"   {m['hi_mag']:.3f} / {m['lo_mag']:.3f}      |  {m['hi_marg']:.3f} / {m['lo_marg']:.3f}")
        return m

    report(0)
    n_up = 0
    for ep, (g, _) in enumerate(zip(grids, y), 1):
        lr = args.lr * (1 - (1 - args.decay) * ep / len(grids))
        ink = np.flatnonzero(g >= 0)
        if len(ink) < 4:
            continue
        for c in rng.choice(ink, min(args.targets, len(ink)), replace=False):
            hid = np.zeros(NC, bool); hid[c] = True
            s, o = context(g, c, nb_cell, nb_off, BLANK, keep_blanks, hid)
            if not len(s):
                continue
            true = int(g[c])
            step(theta, rho, s, o, true, lr, args.beta, V, kk, hw,
                 move_codes=not args.freeze_codes, move_rel=not args.freeze_rel, lrnorm=args.lrnorm)
            np.add.at(CNT, (o, s, true), 1)
            n_up += 1
        if ep % args.report == 0:
            theta %= 2 * np.pi; rho %= 2 * np.pi
            report(ep)

    theta %= 2 * np.pi; rho %= 2 * np.pi
    m = report(len(grids))
    print(f"\n{n_up:,} updates, {time.time()-t0:.0f}s total")
    S = similarity(theta, V, B); S0 = similarity(theta0, V, B)
    print(f"same-class vs different-class code similarity: {S[same].mean():+.3f} / {S[diff].mean():+.3f} "
          f"(at init {S0[same].mean():+.3f} / {S0[diff].mean():+.3f})")
    print(f"confidence: |Z| separates {m['mag_ok']:.3f} right / {m['mag_no']:.3f} wrong; "
          f"top-quarter accuracy {m['hi_mag']:.3f} by |Z|, {m['hi_marg']:.3f} by margin "
          f"(bottom quarter {m['lo_mag']:.3f} / {m['lo_marg']:.3f}, overall {m['acc']:.3f})")
    if args.matrix:
        print("\ncoarse-to-fine: propose with the blurry kernel (m=" + str(args.harm) +
              "), rescore the shortlist with a sharper one\n")
        print("   shortlist |" + "".join(f"{('read m=' + str(r)):>12}" for r in (1, 2, 4, 8, 16, 32)))
        for K in (1, 3, 5, 10, 20):
            row = f"   top-{K:<7}|"
            for r in (1, 2, 4, 8, 16, 32):
                kf, wf = harmonics(r, args.hw)
                row += f"{evaluate(ev, theta, rho, V, near, kk, hw, kf, wf, K)['acc']:>12.3f}"
            print(row)
        print()
    print("recall of the true stroke in the blurry pass: " +
          "  ".join(f"top-{r} {m['rec%d' % r]:.3f}" for r in (1, 3, 5, 10)))
    print(f"collapse check: largest pairwise similarity {S[~np.eye(V,dtype=bool)].max():+.3f}, "
          f"{m['distinct']} of {V} strokes ever predicted")


if __name__ == "__main__":
    main()
