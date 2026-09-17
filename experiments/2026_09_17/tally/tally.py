"""Counting only.   python tally.py [--train 10000] [--seen 8] [--recur 4]

The label is always shown.  A digit is sampled one cell at a time, at random, until `seen`
inked strokes have been seen.  Everything learned is a count:

    class table   count[y][(item, cell)]      absolute positions - digits are centred
    pair table    count[(a, b, offset)]       relative - a part is stored relative to itself

A pair that recurs `recur` times gets a fresh fingerprint and becomes a chunk card (parts at
offsets from its own origin).  From then on, wherever the chunk is recognised in a digit, the
strokes it covers are counted as the chunk, and the chunk pairs with its neighbours, so chunks
grow.  Strokes are compared by fingerprint overlap, so near-identical strokes count as the same.

Reading the library: unbind a class's count bundle by `label` -> its compact description.
Test: label hidden, sample the same way, ask which class's table predicts what was seen.
"""
import argparse, math, time
from collections import defaultdict
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2026_09_16" / "glimpse_loop"))   # sdr.py, mnist.py
import sdr, mnist

STRIDE, PATCH = 3, 7
POS = list(range(0, 28 - PATCH + 1, STRIDE))          # 0,3,...,21 -> 8 positions
NG = len(POS)
CELLS = [(i, j) for i in range(NG) for j in range(NG)]
NC = len(CELLS)


def patches(img):
    return np.stack([img[r:r + PATCH, c:c + PATCH].reshape(-1) for r in POS for c in POS])


def build_vocab(X, n_images, rng, tau_new=2.2):
    voc = mnist.Vocabulary(tau_new=tau_new)
    for t, i in enumerate(rng.permutation(len(X))[:n_images]):
        for p in patches(X[i]):
            voc.learn(p)
        if t % 1000 == 999:
            voc.sweep()
    voc.finalize()
    return voc


def grid_of(voc, img):
    return np.array([voc.sense(p) for p in patches(img)])          # -2 = blank


class Codes:
    """Prototype fingerprints: 8 pixel blocks, 4 context blocks, 4 label blocks (as Part XIV)."""

    def __init__(self, alg, voc, grids, labels, rng, B_sim=8, B_ctx=4, B_lab=4, K_coarse=12):
        P, L = len(voc), alg.L
        self.coarse, _ = mnist.kmeans(voc.means, K_coarse, rng)
        ctx = np.zeros((P, 4, K_coarse + 1)); lab = np.zeros((P, 10))
        for g, y in zip(grids, labels):
            g2 = g.reshape(NG, NG)
            for i in range(NG):
                for j in range(NG):
                    p = g2[i, j]
                    if p < 0:
                        continue
                    lab[p, y] += 1
                    for d, (di, dj) in enumerate(mnist.NEIGH4):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < NG and 0 <= jj < NG:
                            q = g2[ii, jj]
                            ctx[p, d, self.coarse[q] if q >= 0 else K_coarse] += 1
        ctx = np.sqrt(ctx.reshape(P, -1) / (ctx.reshape(P, -1).sum(1, keepdims=True) + 1e-9))
        lab = np.sqrt(lab / (lab.sum(1, keepdims=True) + 1e-9))
        self.codes = np.zeros((P, alg.B), int)
        b = 0
        for _ in range(B_sim):
            mask = rng.random(PATCH * PATCH) < 0.6
            self.codes[:, b], _ = mnist.kmeans(voc.means[:, mask], L, rng); b += 1
        for _ in range(B_ctx):
            mask = rng.random(ctx.shape[1]) < 0.7
            self.codes[:, b], _ = mnist.kmeans(ctx[:, mask], L, rng); b += 1
        for _ in range(B_lab):
            self.codes[:, b], _ = mnist.kmeans(lab + rng.normal(0, 0.02, lab.shape), L, rng); b += 1
        self.overlap = (self.codes[:, None, :] == self.codes[None, :, :]).sum(-1)


class Library:
    """Chunk cards: id -> {(dr, dc): stroke}, plus the two count tables."""

    def __init__(self, alg, overlap, tau, recur, exact):
        self.alg, self.O, self.tau, self.recur, self.exact = alg, overlap, tau, recur, exact
        self.chunks = {}                      # cid -> footprint {(dr,dc): stroke}
        self.names = {}                       # cid -> fingerprint
        self.by_fp = {}
        self.klass = defaultdict(lambda: defaultdict(int))   # y -> {(kind, id, cell): n}
        self.pairs = defaultdict(int)         # ((kind,id),(kind,id),(dr,dc)) -> n
        self.born = {}
        self.used_by = defaultdict(set)       # cid -> classes
        self.uses = defaultdict(int)

    def match(self, part, stroke):
        return part == stroke if self.exact else self.O[part, stroke] >= self.tau

    # ---- recognise chunks in the evidence (largest first, >=2 parts seen, none contradicted)
    def recognise(self, ev):
        used, found = set(), []
        for cid in sorted(self.chunks, key=lambda c: -len(self.chunks[c])):
            fp = self.chunks[cid]
            anchors = set()
            for (dr, dc), p in fp.items():
                for (i, j), s in ev.items():
                    if s >= 0 and self.match(p, s):
                        anchors.add((i - dr, j - dc))
            for a in anchors:
                cells = {(a[0] + dr, a[1] + dc): p for (dr, dc), p in fp.items()}
                if any(c in used for c in cells):
                    continue
                seen = [c for c in cells if c in ev]
                if len(seen) < 2 or len(seen) < len(cells) / 2:
                    continue
                if any(ev[c] < 0 or not self.match(cells[c], ev[c]) for c in seen):
                    continue
                used |= set(cells)
                found.append((cid, a, cells))
                break
        return found, used

    # ---- one digit's worth of counting
    def learn(self, ev, y, ep):
        found, used = self.recognise(ev)
        items = [(("c", cid), a) for cid, a, _ in found]
        items += [(("s", s), c) for c, s in ev.items() if s >= 0 and c not in used]
        for (kind, i), c in items:
            self.klass[y][(kind, i, c)] += 1
            if kind == "c":
                self.used_by[i].add(y); self.uses[i] += 1
        for c, s in ev.items():
            if s < 0:
                self.klass[y][("e", 0, c)] += 1
        n_new = 0
        for x in range(len(items)):
            for z in range(x + 1, len(items)):
                (ka, ca), (kb, cb) = items[x], items[z]
                if ca > cb:
                    (ka, ca), (kb, cb) = (kb, cb), (ka, ca)
                off = (cb[0] - ca[0], cb[1] - ca[1])
                if max(abs(off[0]), abs(off[1])) > 4:
                    continue
                key = (ka, kb, off)
                self.pairs[key] += 1
                if self.pairs[key] == self.recur:
                    n_new += self.make_chunk(ka, kb, off, ep)
        return len(items), sum(1 for c, s in ev.items() if s >= 0), n_new

    def expand(self, kind, i):
        return dict(self.chunks[i]) if kind == "c" else {(0, 0): i}

    def make_chunk(self, ka, kb, off, ep):
        fp = self.expand(*ka)
        for (dr, dc), p in self.expand(*kb).items():
            fp[(dr + off[0], dc + off[1])] = p
        r0, c0 = min(fp)
        fp = {(r - r0, c - c0): p for (r, c), p in fp.items()}
        key = frozenset(fp.items())
        if key in self.by_fp or len(fp) > 12:
            return 0
        cid = len(self.chunks)
        self.chunks[cid] = fp; self.names[cid] = self.alg.random(); self.by_fp[key] = cid; self.born[cid] = ep
        return 1

    # ---- description length of a digit: items x bits
    def bits_item(self):
        return math.log2(len(self.chunks) + 300) + math.log2(NC)

    # ---- label hidden: which class predicts what was seen?
    def classify(self, ev, alpha=0.5):
        found, used = self.recognise(ev)
        items = [("c", cid, a) for cid, a, _ in found]
        items += [("s", s, c) for c, s in ev.items() if s >= 0 and c not in used]
        items += [("e", 0, c) for c, s in ev.items() if s < 0]
        scores = {}
        for y, tab in self.klass.items():
            n = self.n_digits[y]
            s = 0.0
            for kind, i, c in items:
                if kind == "s" and not self.exact:
                    cnt = sum(v for (k2, i2, c2), v in self.cell_index[y].get(c, []) if k2 == "s" and self.match(i2, i))
                else:
                    cnt = tab.get((kind, i, c), 0)
                s += math.log((cnt + alpha) / (n + alpha * 20))
            scores[y] = s
        return max(scores, key=scores.get)

    def index(self, n_digits):
        self.n_digits = n_digits
        self.cell_index = {y: defaultdict(list) for y in self.klass}
        for y, tab in self.klass.items():
            for (k, i, c), v in tab.items():
                self.cell_index[y][c].append(((k, i, c), v))


def sample(grid, rng, seen):
    """Random cells without replacement until `seen` inked strokes have been seen."""
    ev, n_ink = {}, 0
    for k in rng.permutation(NC):
        c = CELLS[k]; ev[c] = int(grid[k]); n_ink += grid[k] >= 0
        if n_ink >= seen:
            break
    return ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=10000); ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seen", type=int, default=8); ap.add_argument("--recur", type=int, default=4)
    ap.add_argument("--tau", type=int, default=4); ap.add_argument("--exact", action="store_true")
    ap.add_argument("--report", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed); t0 = time.time()
    X, y = mnist.load("train"); Xt, yt = mnist.load("t10k")
    sel = rng.permutation(len(X))[:args.train]; X, y = X[sel], y[sel]
    Xt, yt = Xt[:args.test], yt[:args.test]
    voc = build_vocab(X, min(6000, len(X)), rng)
    grids = np.stack([grid_of(voc, im) for im in X]); grids_t = np.stack([grid_of(voc, im) for im in Xt])
    alg = sdr.Algebra(16, 41, args.seed)
    C = Codes(alg, voc, grids[:6000], y[:6000], rng)
    lib = Library(alg, C.overlap, args.tau, args.recur, args.exact)
    print(f"{len(voc)} strokes from 7x7 patches at stride {STRIDE} ({NG}x{NG} cells); inked cells per digit {np.mean(grids >= 0) * NC:.1f}; "
          f"{'exact stroke identity' if args.exact else f'strokes match at overlap >= {args.tau}/16'}; a pair becomes a chunk after {args.recur} recurrences  [{time.time()-t0:.0f}s]")
    print(f"training: label shown, random cells until {args.seen} inked strokes seen\n")
    print("   digits   chunks  items/digit (strokes seen -> after chunking)  chunks used by >=2 classes  sampled cells")
    block = []
    for ep, (g, lab) in enumerate(zip(grids, y), 1):
        ev = sample(g, rng, args.seen)
        n_items, n_ink, n_new = lib.learn(ev, int(lab), ep)
        block.append((n_items, n_ink, len(ev)))
        if ep % args.report == 0:
            b = np.array(block); block = []
            multi = sum(1 for c in lib.chunks if len(lib.used_by[c]) >= 2)
            print(f"  {ep:7d}   {len(lib.chunks):5d}        {b[:,1].mean():4.1f} -> {b[:,0].mean():4.1f}                        "
                  f"{multi:4d} of {sum(1 for c in lib.chunks if lib.uses[c] > 0)} used        {b[:,2].mean():4.1f}   [{time.time()-t0:.0f}s]")
    n_digits = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    lib.index(n_digits)
    # ---- the library
    print(f"\nLIBRARY  {len(lib.chunks)} chunks; sizes: " + ", ".join(f"{s}-stroke: {n}" for s, n in sorted(
        zip(*np.unique([len(f) for f in lib.chunks.values()], return_counts=True)))))
    top = sorted(lib.chunks, key=lambda c: -lib.uses[c])[:8]
    for c in top:
        fp = lib.chunks[c]; rs = [r for r, _ in fp]; cs = [k for _, k in fp]
        pic = [["." for _ in range(max(cs) - min(cs) + 1)] for _ in range(max(rs) - min(rs) + 1)]
        for (r, k) in fp:
            pic[r - min(rs)][k - min(cs)] = "#"
        print(f"  chunk {c:4d}: {len(fp)} strokes, used {lib.uses[c]:5d}x in classes {sorted(lib.used_by[c])}   {' '.join(''.join(row) for row in pic)}")
    # ---- read a class back through the algebra: unbind label, presence of items
    label_role = alg.random(); class_codes = np.stack([alg.random() for _ in range(10)])
    poses = sdr.Poses(alg, NG, NG)
    W = alg.empty()
    for yy, tab in lib.klass.items():
        for (k, i, c), v in sorted(tab.items(), key=lambda x: -x[1])[:12]:
            name = lib.names[i] if k == "c" else C.codes[i] if k == "s" else alg.random()
            alg.add(W, alg.bind(alg.bind(alg.bind(label_role, class_codes[yy]), name), poses(c)))
    ok = 0
    for yy, tab in lib.klass.items():
        U = alg.unbind(W, alg.bind(label_role, class_codes[yy]))
        (k, i, c), v = max(tab.items(), key=lambda x: x[1])
        name = lib.names[i] if k == "c" else C.codes[i]
        ok += int(alg.presence(alg.unbind(U, poses(c)), name[None, :])[0] > 0)
    print(f"  unbind(label ⊗ class) on one bundle of all classes' top items returns each class's most frequent item: {ok}/10")
    # ---- label hidden
    print(f"\nTEST  label hidden, {args.test} digits, random cells until n inked strokes seen; which class's table fits best?")
    print("   strokes seen   accuracy")
    for seen in [2, 4, 8, 16, 32]:
        acc = np.mean([lib.classify(sample(g, rng, seen)) == int(l) for g, l in zip(grids_t, yt)])
        print(f"   {seen:8d}       {acc:6.1%}")
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
