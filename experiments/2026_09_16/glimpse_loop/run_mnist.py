"""MNIST experiments.   python run_mnist.py [--train 10000] [--test 2000] [--stages abc]

  a  vocabulary + codes: similar strokes overlap, same-outcome strokes overlap, noisy sensing
  b  image as one bundle (stroke ⊗ position + label ⊗ class): Part I's lookups on real data
  c  the glimpse loop: program = chunks at poses -> label; policy, metacognition, rotation
"""
import argparse, time
from collections import defaultdict
import numpy as np
import sdr, mnist


def bundles(alg, C, poses, grids, labels=None, label_role=None, class_codes=None):
    """Image -> one bundle: sum over inked cells of code(stroke) ⊗ pose(cell)  (+ label ⊗ class)."""
    N = len(grids)
    W = np.zeros((N, alg.B, alg.L), np.int16)
    ar = np.arange(alg.B)
    for k in range(mnist.GRID * mnist.GRID):
        ids = grids[:, k]
        inked = ids >= 0
        slots = (C.codes[ids[inked]] + poses.cell_codes[k][None, :]) % alg.L      # bind
        np.add.at(W, (np.flatnonzero(inked)[:, None], ar[None, :], slots), 1)
    if labels is not None:
        slots = (class_codes[labels] + label_role[None, :]) % alg.L
        np.add.at(W, (np.arange(N)[:, None], ar[None, :], slots), 1)
    return W


def stage_a(args, alg, voc, C, X, y, Xt, yt, grids, rng):
    print(f"\nSTAGE A  vocabulary: {len(voc)} prototypes from 7x7 patches (gated allocation, tau_new={args.tau}); "
          f"blank patches: {(grids < 0).mean():.0%} of cells")
    m = mnist.measure_codes(voc, C, Xt, rng)
    print("  similar strokes share bits?   sim-block overlap (of %d) by pixel-distance quantile of prototype pairs:" % C.B_sim)
    for lo, hi, ov in m["sim_by_distance"]:
        print(f"      distance {lo:4.2f}-{hi:4.2f}:  overlap {ov:4.2f}")
    for key, name in [("sim_overlap_same_vs_diff", "sim"), ("ctx_overlap_same_vs_diff", "ctx"), ("lab_overlap_same_vs_diff", "lab")]:
        s, d, B = m[key]
        print(f"  {name} blocks: overlap between prototypes with the same dominant label {s:4.2f} vs different {d:4.2f}  (of {B})")
    print("  noisy sensing: noisy patch cleans up to the same prototype as the clean patch?")
    print("                          pixel-nearest   code-overlap    exact-pixel dict")
    for name, (pix, code, exact) in m["noise"].items():
        print(f"      {name:12s}       {pix:6.1%}         {code:6.1%}         {exact:6.1%}")


def stage_b(args, alg, C, poses, grids, y, grids_t, yt, rng):
    print(f"\nSTAGE B  image = bundle of stroke ⊗ position (+ label ⊗ class), {args.train} train / {args.test} test")
    # Part I on real data: the exact dict on the 16-cell grid
    keys = {}
    for g, lab in zip(grids, y):
        keys.setdefault(tuple(g), []).append(lab)
    hits = [tuple(g) in keys for g in grids_t]
    print(f"  exact dict on the grid: {len(keys):,} keys for {len(grids):,} images; test hits {np.mean(hits):.1%}"
          f"  (write-only memory, as in Part I)")
    label_role = alg.random(); class_codes = np.stack([alg.random() for _ in range(10)])
    Wtr = bundles(alg, C, poses, grids, y, label_role, class_codes)
    Wte = bundles(alg, C, poses, grids_t)
    # read the label back out of a bundle by unbinding the role (VII: relation as pose)
    read = []
    for i in range(500):
        U = alg.unbind(Wtr[i].astype(np.int32), label_role)
        read.append(int(alg.presence(U, class_codes).argmax()))
    print(f"  unbind(label) on stored bundles returns the class: {np.mean(np.array(read) == y[:500]):.1%}")
    # nearest neighbours by bundle overlap
    A = (Wtr > 0).reshape(len(Wtr), -1).astype(np.float32)
    Bm = (Wte > 0).reshape(len(Wte), -1).astype(np.float32)
    S = Bm @ A.T                                              # overlap = shared active slots
    for k in [1, 5, 15]:
        nn = np.argsort(-S, axis=1)[:, :k]
        votes = np.zeros((len(Wte), 10))
        for j in range(k):
            np.add.at(votes, (np.arange(len(Wte)), y[nn[:, j]]), 1)
        print(f"  {k:2d}-NN over {len(Wtr):,} stored bundles: {np.mean(votes.argmax(1) == yt):.1%}")
    # the same vote, done in the algebra: superpose the k neighbours' bundles, unbind the label
    k = 5; nn = np.argsort(-S, axis=1)[:, :k]
    alg_pred = []
    for i in range(len(Wte)):
        H = Wtr[nn[i]].astype(np.int32).sum(0)
        alg_pred.append(int(alg.presence(alg.unbind(H, label_role), class_codes).argmax()))
    print(f"  same 5-NN vote through the algebra (superpose neighbours, unbind label): {np.mean(np.array(alg_pred) == yt):.1%}")
    # compressed tables: K prototype bundles per class (latent classes, Part I)
    print("  compressed table: K rows per class (k-means over bundles), nearest row decides")
    for K in [1, 5, 20, 60]:
        protos, plab = [], []
        for c in range(10):
            Ac = A[y == c]
            cent = Ac[rng.choice(len(Ac), min(K, len(Ac)), replace=False)].copy()
            for _ in range(8):
                assign = (Ac @ cent.T).argmax(1)
                for kk in range(len(cent)):
                    if (assign == kk).any():
                        cent[kk] = Ac[assign == kk].mean(0)
            protos.append(cent); plab += [c] * len(cent)
        Pm = np.vstack(protos); plab = np.array(plab)
        pred = plab[(Bm @ Pm.T).argmax(1)]
        print(f"      K={K:3d}: {len(Pm):4d} rows  ->  {np.mean(pred == yt):.1%}")
    # sim-blocks only vs all blocks: does the outcome-hashed part of the code help?
    for name, keep in [("sim blocks only", slice(0, C.B_sim)), ("sim+ctx", slice(0, C.B_sim + C.B_ctx)), ("all blocks", slice(0, alg.B))]:
        A2 = (Wtr[:, keep] > 0).reshape(len(Wtr), -1).astype(np.float32)
        B2 = (Wte[:, keep] > 0).reshape(len(Wte), -1).astype(np.float32)
        nn = np.argsort(-(B2 @ A2.T), axis=1)[:, :5]
        votes = np.zeros((len(Wte), 10))
        for j in range(5):
            np.add.at(votes, (np.arange(len(Wte)), y[nn[:, j]]), 1)
        print(f"  5-NN using {name:16s}: {np.mean(votes.argmax(1) == yt):.1%}")
    return label_role, class_codes


def rotation_map(voc):
    """stroke -> the prototype nearest to its 90-degree rotation (an operator on the vocabulary)."""
    rot = np.array([np.rot90(m.reshape(mnist.PATCH, mnist.PATCH)).reshape(-1) for m in voc.means])
    return np.array([voc.nearest(r)[0] for r in rot])


def evaluate(agent, grids, labels, n, **kw):
    res = [agent.run_episode(g, int(l), ep=0, mode="test", **kw) for g, l in zip(grids[:n], labels[:n])]
    acc = np.mean([r["correct"] for r in res]); conf = np.mean([r["confident"] for r in res])
    acc_c = np.mean([r["correct"] for r in res if r["confident"]]) if conf > 0 else float("nan")
    wc = np.mean([r["wrong_confident"] for r in res]); gl = np.mean([r["glimpses"] for r in res])
    return dict(acc=acc, conf=conf, acc_conf=acc_c, wrong_conf=wc, glimpses=gl, res=res,
                vetoes=np.mean([r["vetoes"] for r in res]))


def fmt(e):
    return (f"accuracy {e['acc']:5.1%}   confident {e['conf']:4.0%} (of which right {e['acc_conf']:5.1%})   "
            f"confident-wrong {e['wrong_conf']:4.1%}   glimpses {e['glimpses']:4.1f}")


def stage_c(args, alg, voc, C, poses, grids, y, grids_t, yt, Xt, rng):
    import digits
    print(f"\nSTAGE C  the glimpse loop on digits: {args.train} training episodes, {args.test} test digits")
    log = print if args.verbose else (lambda s: None)
    table = digits.DigitTable(alg, C, tau_match=args.sim_tau, blank_overlap=args.blank, prov_max=args.prov_max,
                              require_test=not args.no_test, log=log)
    n_sim = table.sim[:-1, :-1].sum(1).mean()
    print(f"  evidence per cell = (overlap - {args.neutral})/({alg.B} - {args.neutral}); a stroke counts as matching at overlap >= {args.sim_tau} "
          f"({n_sim:.1f} similar strokes each); weight = {args.base}^evidence")
    cfg = dict(H=args.H, budget=16, eps=args.eps, concentration=args.concentration, min_ink=args.min_ink,
               verify_min=args.verify_min, meta_min=3, meta_veto=0.35, meta_max=3, policy_k=7, policy_store=30000,
               base=args.base, tau=args.neutral)
    rot_map = rotation_map(voc)
    agent = digits.DigitAgent(alg, table, poses, cfg, log, rot_map=rot_map)
    t0 = time.time(); block = []
    order = rng.permutation(len(grids))[:args.train]
    meta_early = None
    print("  training (online, one pass):")
    print("     episodes   accuracy  confident  conf-wrong  glimpses(+study)  exemplars prov/cons  chunks prov/cons  clicks  ideas")
    tt = defaultdict(float)
    for t, i in enumerate(order, 1):
        t1 = time.time()
        st = agent.run_episode(grids[i], int(y[i]), ep=t, mode="train", verbose=args.verbose and t <= args.verbose)
        tt["episode"] += time.time() - t1
        block.append(st)
        if t == 500:
            meta_early = {k: list(v) for k, v in agent.meta.items()}
        if t % args.report == 0 or t == len(order):
            ex_p = len(table.exemplars("provisional")); ex_c = len(table.exemplars("consolidated"))
            ch_p = len(table.chunks("provisional")); ch_c = len(table.chunks("consolidated"))
            clicks = sum(1 for e in table.events if e[1] == "click" and e[0] <= t)
            print(f"     {t:8d}   {np.mean([s['correct'] for s in block]):6.1%}    {np.mean([s['confident'] for s in block]):4.0%}"
                  f"      {np.mean([s['wrong_confident'] for s in block]):4.1%}     {np.mean([s['glimpses'] for s in block]):4.1f} (+{np.mean([s['study'] for s in block]):3.1f})"
                  f"        {ex_p:5d}/{ex_c:<5d}        {ch_p:4d}/{ch_c:<4d}     {clicks:4d}   {sum(s.get('ideas', 0) for s in block):4d}"
                  f"   [{time.time()-t0:.0f}s]")
            block = []
    agent.freeze_policy()
    table.clicks(len(order), prune=True)
    print(f"  ({time.time()-t0:.0f}s training)")
    # ---- test ----
    print("\n  TEST")
    main = evaluate(agent, grids_t, yt, args.test)
    print(f"    computed policy (VI.5):        {fmt(main)}")
    whys = defaultdict(int)
    for r in main["res"]:
        whys[r["why"]] += 1
    print("      at the last step, the declare rule was blocked by: " + ", ".join(f"{k} {v/len(main['res']):.0%}" for k, v in sorted(whys.items(), key=lambda x: -x[1])))
    n2 = min(args.test, args.variant_n)
    for pol in ["learned", "random", "raster"]:
        e = evaluate(agent, grids_t, yt, n2, policy=pol)
        print(f"    {pol:8s} policy ({n2}):        {fmt(e)}")
    em = evaluate(agent, grids_t, yt, n2, use_meta=True)
    print(f"    computed + self-model (IX):    {fmt(em)}   vetoes/episode {em['vetoes']:.2f}")
    if meta_early is not None:
        full_meta = agent.meta; agent.meta = defaultdict(lambda: [0, 0], {k: v for k, v in meta_early.items()})
        e = evaluate(agent, grids_t, yt, n2, use_meta=True)
        print(f"    self-model from first 500 only: {fmt(e)}   vetoes/episode {e['vetoes']:.2f}")
        agent.meta = full_meta
    # noisy sensing
    Xn = np.clip(Xt[:n2] + rng.normal(0, 0.25, Xt[:n2].shape), 0, 1)
    gn = np.stack([voc.grid(img) for img in Xn])
    e = evaluate(agent, gn, yt, n2)
    print(f"    pixel noise sigma=0.25 (fuzzy): {fmt(e)}")
    O_saved, sim_saved = table.O, table.sim
    table.O = np.where(np.eye(len(O_saved), dtype=bool), alg.B, 0).astype(np.int16); table.sim = np.eye(len(O_saved), dtype=bool)
    table.O[table.EMPTY, table.EMPTY] = args.blank
    e = evaluate(agent, gn, yt, n2)
    e0 = evaluate(agent, grids_t, yt, n2)
    print(f"    exact stroke identity, clean:   {fmt(e0)}")
    print(f"    exact stroke identity, noisy:   {fmt(e)}")
    table.O, table.sim = O_saved, sim_saved
    # rotation as an operator pose
    Xr = np.stack([np.rot90(img) for img in Xt[:n2]])
    gr = np.stack([voc.grid(img) for img in Xr])
    e_no = evaluate(agent, gr, yt, n2)
    e_op = evaluate(agent, gr, yt, n2, rotations=(0, 1, 2, 3))
    print(f"    rotated 90, no operator:        {fmt(e_no)}")
    print(f"    rotated 90, rotation operator:  {fmt(e_op)}")
    conf69 = sum(1 for r, l in zip(e_op["res"], yt[:n2]) if (l, r["label_hat"]) in [(6, 9), (9, 6)])
    n69 = sum(1 for l in yt[:n2] if l in (6, 9))
    print(f"      6/9 swapped under rotation: {conf69} of {n69} sixes and nines")
    e_up = evaluate(agent, grids_t, yt, n2, rotations=(0, 1, 2, 3))
    print(f"    upright, operator on anyway:    {fmt(e_up)}")
    back = np.arange(len(voc))
    for k in range(1, 5):
        back = rot_map[back]
        print(f"      rot^{k}: {np.mean(back == np.arange(len(voc))):4.0%} of strokes return exactly, "
              f"mean code overlap with the original {C.overlap[np.arange(len(voc)), back].mean():4.1f}/{alg.B}")
    print("      (VII.3: the relation has order 4 - visible through the cleanup noise of a vocabulary not closed under rotation)")
    # ---- library ----
    ex_c = table.exemplars("consolidated"); ch_c = table.chunks("consolidated"); ex_all = table.exemplars()
    flat = sum(table.bits_name() + len(table.footprint(r.id)) * table.bits_part() for r in ex_all)
    used = sum(table.row_cost(r) for r in ex_all) + sum(table.row_cost(r) for r in ch_c)
    print(f"\n  LIBRARY  {len(ex_c)} consolidated exemplars ({len(table.exemplars('provisional'))} provisional), "
          f"{len(ch_c)} consolidated chunks; unlearned {sum(1 for e in table.events if e[1] == 'unlearn')}")
    print(f"    all remembered digits written flat: {flat:,.0f} bits;  with chunks (incl. chunk rows): {used:,.0f} bits")
    reuse = defaultdict(set)
    for r in ex_c:
        for pid, _ in r.parts:
            if not table.is_prim(pid):
                reuse[pid].add(r.label)
    multi = sum(1 for s in reuse.values() if len(s) >= 2)
    print(f"    chunks used by consolidated exemplars: {len(reuse)}; used across >=2 digit classes: {multi}")
    chunk_parts = np.mean([np.mean([not table.is_prim(p) for p, _ in r.parts]) for r in ex_c]) if ex_c else 0
    print(f"    exemplar parts that are chunks: {chunk_parts:.0%}")
    top_chunks = sorted(ch_c, key=lambda r: -len(reuse.get(r.id, ())))[:6]
    for r in top_chunks:
        fp = table.footprint(r.id)
        rs = [i for i, _ in fp]; cs = [j for _, j in fp]
        pic = [["." for _ in range(max(cs) - min(cs) + 1)] for _ in range(max(rs) - min(rs) + 1)]
        for (i, j), p in fp.items():
            pic[i - min(rs)][j - min(cs)] = "#"
        print(f"    {table.label_str(r.id):6s} {len(fp)} strokes, in classes {sorted(reuse.get(r.id, ()))}  shape {' '.join(''.join(row) for row in pic)}")
    shown = 0
    for r in sorted(ex_c, key=lambda r: -r.top_uses):
        if any(not table.is_prim(p) for p, _ in r.parts):
            print(f"    program for a {r.label}: {table.label_str(r.id)} = {table.describe(r)}   (recognised {r.top_uses}x)")
            shown += 1
            if shown == 3:
                break
    return agent, table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=10000); ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--vocab-images", type=int, default=6000); ap.add_argument("--tau", type=float, default=2.2)
    ap.add_argument("--B-sim", type=int, default=8); ap.add_argument("--B-ctx", type=int, default=4); ap.add_argument("--B-lab", type=int, default=4)
    ap.add_argument("--L", type=int, default=41); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stages", default="abc")
    ap.add_argument("--sim-tau", type=int, default=4); ap.add_argument("--neutral", type=float, default=2.0)
    ap.add_argument("--base", type=float, default=4.0); ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--eps", type=float, default=0.5); ap.add_argument("--concentration", type=float, default=0.95)
    ap.add_argument("--min-ink", type=float, default=1.0); ap.add_argument("--verify-min", type=float, default=1.0)
    ap.add_argument("--blank", type=int, default=6); ap.add_argument("--prov-max", type=int, default=1500)
    ap.add_argument("--no-test", action="store_true"); ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--report", type=int, default=1000); ap.add_argument("--variant-n", type=int, default=500)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    X, y = mnist.load("train"); Xt, yt = mnist.load("t10k")
    sel = rng.permutation(len(X))[:args.train]; X, y = X[sel], y[sel]
    selt = rng.permutation(len(Xt))[:args.test]; Xt, yt = Xt[selt], yt[selt]
    voc = mnist.build_vocabulary(X, n_images=args.vocab_images, seed=args.seed, tau_new=args.tau)
    grids = np.stack([voc.grid(img) for img in X]); grids_t = np.stack([voc.grid(img) for img in Xt])
    alg = sdr.Algebra(args.B_sim + args.B_ctx + args.B_lab, args.L, args.seed)
    poses = sdr.Poses(alg, mnist.GRID, mnist.GRID)
    C = mnist.Codes(alg, voc, grids, y, args.B_sim, args.B_ctx, args.B_lab, seed=args.seed)
    print(f"algebra: {alg.B} blocks x {alg.L} slots = {alg.B*alg.L} bits ({args.B_sim} sim + {args.B_ctx} ctx + {args.B_lab} lab); "
          f"data prepared in {time.time()-t0:.0f}s")
    if "a" in args.stages:
        stage_a(args, alg, voc, C, X, y, Xt, yt, grids, rng)
    if "b" in args.stages:
        stage_b(args, alg, C, poses, grids, y, grids_t, yt, rng)
    if "c" in args.stages:
        stage_c(args, alg, voc, C, poses, grids, y, grids_t, yt, Xt, rng)
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
