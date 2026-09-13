"""Completeness check: best-first search over the add lattice, against the beam.

Same exact cost as the engine. From the empty set every subset is reachable by
adds alone, so no drops or swaps are needed for completeness. Ordering and
pruning use an optimistic bound: adding a template can gain at most the
unexplained energy on the region it would take (the pixels where it is sharper
than the current owners), and a set of adds can gain at most the sum of the
best such gains net of price, and never more than the total unexplained
energy. The beam's own answer is the starting incumbent, so a node is dropped
as soon as its bound cannot beat it. If the heap empties or the cheapest bound
left is no better than the incumbent, the incumbent is proved optimal (up to
the bound assumption that gains do not reinforce each other; violations are
counted).

Also runs the beam at widths 1, 4 and 16 on the same images.

    python astar.py <rule> <price> [n_images] [expansion cap] [branching cap]
"""

import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE))
import engine as E  # noqa: E402
from run import load  # noqa: E402


def state_maps(net, state, x):
    S = E.S
    idx = np.full(S, -1)
    idx[:len(state)] = state
    _, _, xh, cost, _ = net.exact(idx[None, None], x[None])
    Mx = (net.Wp[list(state)].max(0) if state else np.zeros(E.D, np.float32))
    return idx, xh[0, 0], float(cost[0, 0]), Mx


def bound_gain(net, x, xh, Mx, count):
    """Most a set of further adds could gain, net of price."""
    e = (x - xh) ** 2
    U = float(e.sum())
    regE = (net.Wp > Mx[None, :]).astype(np.float32) @ e          # energy on the region each template would take
    ub = np.sort(regE - net.lam)[::-1][:max(E.S - count, 0)]
    return min(float(np.maximum(ub, 0).sum()), U), regE


def astar(net, x, incumbent_cost, incumbent_set, cap, branch):
    lam = net.lam
    idx0, xh0, c0, Mx0 = state_maps(net, (), x)
    bg0, _ = bound_gain(net, x, xh0, Mx0, 0)
    heap = [(c0 - bg0, c0, ())]
    closed = set()
    best_cost, best_set = incumbent_cost, incumbent_set
    expansions = violations = truncated = 0
    proved = False
    while heap:
        f, cost, state = heapq.heappop(heap)
        if f >= best_cost - 1e-9:                       # nothing left can beat the incumbent
            proved = True
            break
        if state in closed:
            continue
        closed.add(state)
        if cost < best_cost - 1e-9:
            best_cost, best_set = cost, state
        if expansions >= cap or len(state) >= E.S:
            break
        expansions += 1
        idx, xh, _, Mx = state_maps(net, state, x)
        _, regE = bound_gain(net, x, xh, Mx, len(state))
        regE[list(state)] = -np.inf
        cands = np.argsort(-regE)
        cands = cands[regE[cands] > lam]
        if len(cands) > branch:
            truncated += 1
            cands = cands[:branch]
        if len(cands) == 0:
            continue
        children = np.repeat(idx[None], len(cands), 0)
        children[:, len(state)] = cands
        _, _, cxh, ccost, _ = net.exact(children[None], x[None])
        for j, t in enumerate(cands):
            c = float(ccost[0, j])
            if (cost - c) + lam > regE[t] + 1e-6:        # gained more energy than the region held
                violations += 1
            child = tuple(sorted(state + (int(t),)))
            if child in closed:
                continue
            Mxc = np.maximum(Mx, net.Wp[t])
            bg, _ = bound_gain(net, x, cxh[0, j], Mxc, len(child))
            fc = c - bg
            if fc >= best_cost - 1e-9:
                continue
            heapq.heappush(heap, (fc, c, child))
    if not heap:
        proved = True
    return best_cost, best_set, proved, expansions, violations, truncated


def main():
    rule, lam = sys.argv[1], float(sys.argv[2])
    n_img = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    branch = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    Z = np.load(OUT / f"weights_{rule}_l{lam:g}.npz")
    net = E.Ownership(rule, lam, np.random.default_rng(0))
    net.W, net.Wp = Z["W"], Z["Wp"]
    _, _, Xte, _ = load(0)
    X = Xte[:n_img]
    widths = {}
    for w in (1, 4, 16):
        E.BEAM = w
        st = net.search(X)
        widths[w] = dict(cost=st["cost"].copy(), idx=st["idx"].copy())
    E.BEAM = 4
    rows = []
    t0 = time.time()
    for i in range(n_img):
        beam = float(widths[4]["cost"][i])
        beam_set = tuple(sorted(int(t) for t in widths[4]["idx"][i] if t >= 0))
        bc, bs, proved, ex, viol, trunc = astar(net, X[i], beam, beam_set, cap, branch)
        rows.append(dict(beam1=float(widths[1]["cost"][i]), beam=beam, beam16=float(widths[16]["cost"][i]),
                         astar=bc, proved=proved, expansions=ex, violations=viol, truncated=trunc,
                         same_set=beam_set == tuple(bs), beam_n=len(beam_set), astar_n=len(bs)))
        print(f"  img {i:2d}  beam1 {rows[-1]['beam1']:.4f}  beam4 {beam:.4f} ({len(beam_set)} on)  beam16 {rows[-1]['beam16']:.4f}  "
              f"best-first {bc:.4f} ({len(bs)} on)  {'proved' if proved else 'cap'}  {ex} exp  {viol} viol  {trunc} trunc", flush=True)
    gap = np.array([r["beam"] - r["astar"] for r in rows])
    summ = dict(rule=rule, lam=lam, n=n_img, cap=cap, branch=branch, t=time.time() - t0,
                beam_matches=float((np.abs(gap) < 1e-6).mean()), astar_better=float((gap > 1e-6).mean()),
                mean_gap=float(gap.mean()), max_gap=float(gap.max()),
                proved=float(np.mean([r["proved"] for r in rows])),
                violations=int(sum(r["violations"] for r in rows)), truncated=int(sum(r["truncated"] for r in rows)),
                expansions=float(np.mean([r["expansions"] for r in rows])),
                beam1=float(np.mean([r["beam1"] for r in rows])), beam4=float(np.mean([r["beam"] for r in rows])),
                beam16=float(np.mean([r["beam16"] for r in rows])), rows=rows)
    with open(OUT / f"astar_{rule}_l{lam:g}.json", "w") as f:
        json.dump(summ, f)
    print(f"beam4 == best-first on {summ['beam_matches']*100:.0f}%; best-first better on {summ['astar_better']*100:.0f}% "
          f"(mean gap {summ['mean_gap']:.4f}, max {summ['max_gap']:.4f}); proved {summ['proved']*100:.0f}%; "
          f"violations {summ['violations']}; truncated {summ['truncated']}; {summ['expansions']:.0f} exp/img; "
          f"mean cost beam1 {summ['beam1']:.4f} beam4 {summ['beam4']:.4f} beam16 {summ['beam16']:.4f}; {summ['t']:.0f}s")


if __name__ == "__main__":
    main()
