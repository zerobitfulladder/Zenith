"""Run the toy world.   python run.py [--episodes N] [--transfer M] [--verbose K] [--compare]"""
import argparse, math, sys, time
from collections import Counter
import numpy as np
import sdr, memory, world, agent


def build(args, seed, log):
    alg = sdr.Algebra(args.B, args.L, seed)
    poses = sdr.Poses(alg, args.rows, args.cols)
    w = world.World(alg, args.rows, args.cols, n_prims=args.prims, n_subs=args.subs,
                    n_objects=args.objects, seed=seed)
    table = memory.Table(alg, n_cells=args.rows * args.cols, n_rel=11 * 11, prov_max=args.prov_max,
                         click_margin=args.margin, require_test=not args.no_test,
                         prune_margin=args.prune_margin, log=log)
    for lab, code in zip(w.prim_labels, w.prim_codes):
        table.add_primitive(lab, code)
    table.EMPTY = table.add_primitive(".", w.empty_code)
    cfg = dict(H=args.H, budget=args.budget, eps=args.eps, policy=args.policy, compact=args.compact,
               concentration=args.concentration, verify_min=args.verify_min)
    ag = agent.Agent(alg, table, poses, cfg, log)
    return alg, poses, w, table, ag


def block_summary(stats, label):
    n = len(stats)
    out = Counter(s["outcome"] for s in stats)
    g = np.mean([s["glimpses"] for s in stats])
    gr = [s["glimpses"] for s in stats if s["outcome"] == "recognized"]
    fs = [s for s in stats if not s["known_before"]]
    first = f"first sightings {len(fs):2d}: {np.mean([s['glimpses'] for s in fs]):4.1f} glimpses, peak {np.mean([s['peak'] for s in fs]):4.1f}" if fs else "first sightings  0"
    return (f"{label:>12s}  glimpses {g:5.1f}  (recognized: {np.mean(gr) if gr else float('nan'):4.1f})  "
            f"recognized {out['recognized']/n:4.0%}  wrong {out['wrong']/n:4.0%}  novel {out['novel']/n:4.0%}  "
            f"missed {out['missed']/n:4.0%}  completion {np.mean([s['completion'] for s in stats]):4.0%}  "
            f"WM peak {np.mean([s['peak'] for s in stats]):4.1f}  ghosts {sum(s['ghosts'] for s in stats):3d}  "
            f"tries {sum(s['tries'] for s in stats):2d}  overwhelmed {sum(s['overwhelmed'] for s in stats):2d}  | {first}")


def run(args, seed, verbose, quiet=False):
    lines = []
    def log(s):
        if verbose_now[0]:
            print(s)
    verbose_now = [False]
    alg, poses, w, table, ag = build(args, seed, log)
    if not quiet:
        print(f"algebra: B={alg.B} blocks x L={alg.L} slots = {alg.B*alg.L} bits, {alg.B} on   "
              f"VI.4 composition check: {'ok' if poses.check_composition() else 'FAILED'}")
        print(f"world: {args.rows}x{args.cols} canvas, {len(w.prim_labels)} primitives, "
              f"{len(w.subs)} hidden sub-assemblies, {len(w.objects)} objects (sub pairs {w.object_subs})")
        print(f"agent: H={args.H} hypotheses held, pressure eps={args.eps} ghosts/read, policy={args.policy}, compact={args.compact}\n")
    history = []
    t0 = time.time()
    phase_a = args.episodes
    for ep in range(1, args.episodes + args.transfer + 1):
        if ep == phase_a + 1:
            n_before = len(w.objects)
            w.add_objects(args.new_objects)
            new_ids = list(range(n_before, len(w.objects)))
            if not quiet:
                print(f"\n---- transfer: {len(new_ids)} new objects from the same sub-assemblies "
                      f"(sub pairs {w.object_subs[n_before:]}) ----")
        verbose_now[0] = ep <= verbose or (ep > phase_a and ep <= phase_a + verbose)
        obj_id = int(w.rng.choice(new_ids)) if ep > phase_a else None     # transfer: new objects only
        st = ag.run_episode(w, ep, obj_id)
        history.append(st)
        if not verbose_now[0] and not quiet:
            ev = [f"{k} {t}" for (e, k, t) in table.events if e == ep]
            line = (f"ep {ep:3d} obj {st['obj']}  {st['outcome']:10s} {st['declared']:>5s}  "
                    f"{st['glimpses']:2d} glimpses  peak {st['peak']:2d}"
                    + (f"  ghosts {st['ghosts']}" if st["ghosts"] else "")
                    + (f"  ideas {st['ideas']}" if st["ideas"] else ""))
            print(line)
            for e in ev:
                print(f"         ** {e}")
    return history, table, w, time.time() - t0


def library_report(table, w):
    cons = table.chunks("consolidated")
    print(f"\nLIBRARY  ({len(cons)} consolidated rows, {len(table.chunks('provisional'))} provisional, "
          f"{table.table_bits():.0f} bits)")
    hidden = {fp: f"sub{i}" for i, fp in enumerate(w.subs)}
    hidden.update({fp: f"obj{i}" for i, fp in enumerate(w.objects)})
    for r in sorted(cons, key=lambda r: r.consolidated_at):
        fp = table.footprint(r.id)
        tag = hidden.get(fp, "")
        pic = "  ".join(table.sketch(r.id))
        print(f"  {r.label:5s} ep {r.consolidated_at:3d}  used {r.uses:3d}  right {r.worked:3d}  gain {table.gain(r):+6.1f}  "
              f"= {table.describe(r):36s} {tag:5s} [{pic}]")
    found_subs = sum(1 for fp in w.subs if any(table.footprint(r.id) == fp for r in cons))
    found_objs = sum(1 for fp in w.objects if any(table.footprint(r.id) == fp for r in cons))
    spurious = sum(1 for r in cons if table.footprint(r.id) not in hidden)
    print(f"  hidden library recovered: {found_subs}/{len(w.subs)} sub-assemblies, "
          f"{found_objs}/{len(w.objects)} objects; {spurious} rows match nothing hidden")
    obj_rows = [r for r in cons if table.footprint(r.id) in {fp for fp in w.objects}]
    if obj_rows:
        reuse = np.mean([np.mean([not table.rows[p].is_prim for p, _ in r.parts]) for r in obj_rows])
        print(f"  object rows written in terms of chunks: {reuse:.0%} of their parts")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--transfer", type=int, default=60)
    ap.add_argument("--new-objects", type=int, default=4)
    ap.add_argument("--verbose", type=int, default=4, help="print the first K episodes step by step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=10); ap.add_argument("--L", type=int, default=41)
    ap.add_argument("--no-test", action="store_true", help="click on MDL gain alone, without a verified prediction")
    ap.add_argument("--rows", type=int, default=12); ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--prims", type=int, default=6); ap.add_argument("--subs", type=int, default=5)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--H", type=int, default=5); ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--eps", type=float, default=0.5); ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--prov-max", type=int, default=60)
    ap.add_argument("--prune-margin", type=float, default=10.0, help="bits below zero before a chunk is unlearned")
    ap.add_argument("--concentration", type=float, default=0.95, help="leader's share of hypothesis weight to declare")
    ap.add_argument("--verify-min", type=int, default=2, help="predictions that must come true before declaring")
    ap.add_argument("--policy", default="disagree", choices=["disagree", "frontier"])
    ap.add_argument("--compact", default="greedy", choices=["greedy", "pressure"])
    ap.add_argument("--block", type=int, default=30)
    ap.add_argument("--capacity", action="store_true", help="measure XI.1 for this algebra first")
    ap.add_argument("--compare", action="store_true", help="also run the frontier policy and pressure-only compaction")
    args = ap.parse_args(argv)

    if args.capacity:
        alg = sdr.Algebra(args.B, args.L, args.seed)
        V = args.prims + 1 + 30
        print(f"XI.1  ghosts per full read-out (3 tags x {args.rows*args.cols} cells x {V} names) vs items in WM")
        print("   n   measured  predicted")
        for n, m, p in sdr.measure_capacity(alg, args.rows * args.cols, 3, V, trials=20):
            if n % 3 == 0 or m > 0.3:
                print(f"  {n:3d}   {m:7.2f}   {p:8.2f}")
        print()

    history, table, w, dt = run(args, args.seed, args.verbose)
    print(f"\n({dt:.1f}s)\nSUMMARY per block of {args.block} episodes")
    A = history[:args.episodes]
    for i in range(0, len(A), args.block):
        print(block_summary(A[i:i + args.block], f"ep {i+1}-{min(i+args.block, len(A))}"))
    if args.transfer:
        T = history[args.episodes:]
        for i in range(0, len(T), args.block):
            print(block_summary(T[i:i + args.block], f"new {i+1}-{min(i+args.block, len(T))}"))
    clicks = [(e, t) for e, k, t in table.events if k == "click"]
    print(f"\nclicks: {len(clicks)}   first ten:")
    for e, t in clicks[:10]:
        print(f"  ep {e:3d}  {t}")
    library_report(table, w)
    unl = [(e, t) for e, k, t in table.events if k == "unlearn"]
    print(f"\nMDL  library {table.table_bits():.0f} bits;  items left in WM at episode end: "
          f"first block {np.mean([s['end_items'] for s in history[:args.block]]):.1f}, "
          f"last block {np.mean([s['end_items'] for s in history[-args.block:]]):.1f};  unlearned rows: {len(unl)}")

    if args.compare:
        print("\nCOMPARE (same seed): mean glimpses per block")
        rows = {}
        for name, kw in [("disagree/greedy", {}), ("frontier/greedy", dict(policy="frontier")),
                         ("disagree/pressure", dict(compact="pressure")), ("B=12 (492 bits)", dict(B=12)),
                         ("B=8 (328 bits)", dict(B=8)), ("no test", dict(no_test=True))]:
            a2 = argparse.Namespace(**vars(args)); [setattr(a2, k, v) for k, v in kw.items()]
            h2, t2, w2, _ = run(a2, args.seed, 0, quiet=True)
            rows[name] = h2
        blocks = list(range(0, len(history), args.block))
        print(f"  {'':18s}" + "".join(f"{i+1:>4d}-{min(i+args.block,len(history)):<4d}" for i in blocks))
        for name, h2 in rows.items():
            print(f"  {name:18s}" + "".join(f"{np.mean([s['glimpses'] for s in h2[i:i+args.block]]):8.1f} " for i in blocks))
        print(f"  {'recognized %':18s}")
        for name, h2 in rows.items():
            print(f"  {name:18s}" + "".join(f"{np.mean([s['correct'] for s in h2[i:i+args.block]]):8.0%} " for i in blocks))
        print(f"  {'totals':18s}" + "   ".join(
            f"{name}: ghosts {sum(s['ghosts'] for s in h2)}, tries {sum(s['tries'] for s in h2)}, "
            f"overwhelmed {sum(s['overwhelmed'] for s in h2)}, wrong {sum(s['outcome'] == 'wrong' for s in h2)}"
            for name, h2 in rows.items()))


if __name__ == "__main__":
    main()
