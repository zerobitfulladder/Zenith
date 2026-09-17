"""Usefulness-built library.   python tally2.py [--train 10000] [--seen 12]

3x3 patches at stride 2 -> 13x13 cells, a small primitive alphabet.  Label shown in training.
A digit is sampled one random cell at a time until `seen` inked cells are seen.

Everything is a count, everything is scored in bits of surprise:
    class table       count[y, cell, stroke]  and  count[y][(chunk, cell)]
    candidate chunks  from (a) recurring pairs of items at relative offsets,
                           (b) a small beam search over this digit's items when the digit is
                               still surprising, scored by how label-specific the combination
                               has been in the past (an inverted index of recent digits)
    a chunk's worth   = surprise it removed on unseen cells + surprise it removed about the
                        label, summed over uses, minus the bits to store it
                      -> consolidated when positive, dropped when clearly negative
Recognition is bottom-up and never deletes stroke evidence; predictions use the finest level.
"""
import argparse, math, time
from collections import defaultdict
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2026_09_16" / "glimpse_loop"))   # sdr.py, mnist.py
import sdr, mnist

PS, STRIDE = 3, 2


def configure(ps, stride):
    global PS, STRIDE, POS, NG, CELLS, NC, CELL_ID
    PS, STRIDE = ps, stride
    POS = list(range(0, 28 - PS + 1, STRIDE))
    NG = len(POS)
    CELLS = [(i, j) for i in range(NG) for j in range(NG)]
    NC = len(CELLS)
    CELL_ID = {c: k for k, c in enumerate(CELLS)}


configure(PS, STRIDE)


def patches(img):
    return np.stack([img[r:r + PS, c:c + PS].reshape(-1) for r in POS for c in POS])


class Vocab:
    def __init__(self, tau_new, ink):
        self.tau_new, self.ink = tau_new, ink
        self.means = np.zeros((0, PS * PS), np.float32); self.counts = np.zeros(0, int)

    def blank(self, p):
        return (p > 0.5).sum() < self.ink

    def nearest(self, p):
        d = np.sqrt(((self.means - p) ** 2).sum(1)); i = int(d.argmin()); return i, float(d[i])

    def learn(self, p):
        if self.blank(p):
            return
        if len(self.means) == 0 or self.nearest(p)[1] > self.tau_new:
            self.means = np.vstack([self.means, p[None]]); self.counts = np.append(self.counts, 1); return
        i = self.nearest(p)[0]; self.counts[i] += 1
        self.means[i] += (p - self.means[i]) / min(self.counts[i], 500)

    def finalize(self, min_count):
        keep = self.counts >= min_count
        self.means, self.counts = self.means[keep], self.counts[keep]

    def sense(self, p):
        return -1 if self.blank(p) else self.nearest(p)[0]

    def grid(self, img):
        return np.array([self.sense(p) for p in patches(img)])


def graded_codes(voc, X, rng, B=8, L=41, topk=3, n_patches=30000):
    """Population code per block (Lavender's proposal): each block is a spherical k-means over
    mean-centred, L2-normalised patches; a stroke's block holds the cosines to its top-k
    centroids, clipped at 0 and L2-normalised, so every stroke carries the same mass."""
    ps = np.vstack([patches(X[i]) for i in rng.permutation(len(X))[:n_patches // NC + 1]])
    ps = ps[(ps > 0.5).sum(1) >= 2]
    ps = ps - ps.mean(1, keepdims=True); ps /= np.linalg.norm(ps, axis=1, keepdims=True) + 1e-9
    means = voc.means - voc.means.mean(1, keepdims=True); means /= np.linalg.norm(means, axis=1, keepdims=True) + 1e-9
    codes = np.zeros((len(means), B * L))
    for b in range(B):
        cent = ps[rng.choice(len(ps), L, replace=False)].copy()
        for _ in range(12):
            a = (ps @ cent.T).argmax(1)
            for k in range(L):
                if (a == k).any():
                    v = ps[a == k].mean(0); cent[k] = v / (np.linalg.norm(v) + 1e-9)
        cos = means @ cent.T
        for s in range(len(means)):
            top = np.argsort(-cos[s])[:topk]
            bump = np.clip(cos[s][top], 0, None)
            codes[s, b * L + top] = bump / (np.linalg.norm(bump) + 1e-9)
    return codes


class Library:
    def __init__(self, alg, sim, P, rng, recur, beam, prune_after, log, codes=None, depth=3):
        self.alg, self.sim, self.P, self.rng, self.recur, self.beam, self.prune_after, self.log = alg, sim, P, rng, recur, beam, prune_after, log
        self.depth = depth
        self.codes = codes                        # graded: (P, D) bumps; None = hard counts
        self.B = 8
        self.parts = {}          # cid -> [((kind, id), (dr, dc))]
        self.fp = {}             # cid -> {(dr, dc): stroke}   expanded
        self.level = {}
        self.status = {}         # provisional | consolidated | dead
        self.gain_pred = defaultdict(float); self.gain_lab = defaultdict(float); self.trials = defaultdict(int)
        self.origin = {}; self.origin_ep = {}
        self.by_key = {}
        self.index = defaultdict(list)          # stroke -> [(cid, (dr, dc))]  (graded: similar strokes too)
        self.cstroke = np.zeros((10, NC, P if codes is None else codes.shape[1]))     # class table (hard ids or soft mass)
        self.cblank = np.zeros((10, NC))
        self.cchunk = defaultdict(lambda: defaultdict(float))   # y -> {(cid, cell): n}
        self.n_digits = np.zeros(10)
        self.pairs = defaultdict(int)
        self.used_by = defaultdict(set); self.uses = defaultdict(int)
        self.recent = []                          # (y, {cell: stroke}) for the beam's index
        self.inv = defaultdict(set)               # (cell, stroke) -> set of recent ids
        self.surprise_mean = 5.0
        self.clicks = []
        self.pred_hit = defaultdict(int); self.pred_n = defaultdict(int)

    # ---------------- bits ----------------
    def V(self):
        return self.P + 1 + len(self.parts)

    def cost(self, cid):
        return math.log2(self.V()) + len(self.parts[cid]) * (math.log2(self.V()) + math.log2(81))

    # ---------------- chunks ----------------
    def make(self, parts, ep, origin):
        fp = {}
        for (kind, i), (dr, dc) in parts:
            sub = self.fp[i] if kind == "c" else {(0, 0): i}
            for (r, c), p in sub.items():
                fp[(r + dr, c + dc)] = p
        if len(fp) > 16:
            return None
        r0 = min(r for r, _ in fp); c0 = min(c for _, c in fp)
        fp = {(r - r0, c - c0): p for (r, c), p in fp.items()}
        key = frozenset(fp.items())
        if key in self.by_key:
            return None
        cid = len(self.parts)
        self.parts[cid] = [((k, i), (dr - r0, dc - c0)) for (k, i), (dr, dc) in parts]
        self.fp[cid] = fp; self.by_key[key] = cid
        self.level[cid] = 1 + max((self.level[i] for (k, i), _ in parts if k == "c"), default=0)
        self.status[cid] = "provisional"; self.origin[cid] = origin; self.origin_ep[cid] = ep
        for (dr, dc), p in fp.items():
            for q in np.flatnonzero(self.sim[p]):
                self.index[int(q)].append((cid, (dr, dc)))
        return cid

    def kill(self, cid):
        self.status[cid] = "dead"
        for p in list(self.index):
            self.index[p] = [(c, o) for c, o in self.index[p] if c != cid]

    # ---------------- recognition (bottom-up on leaf strokes; >= half the cells seen, none contradicted)
    def recognise(self, ev):
        votes = defaultdict(set)
        for (i, j), s in ev.items():
            if s < 0:
                continue
            for cid, (dr, dc) in self.index[s]:
                if self.status[cid] != "dead":
                    votes[(cid, (i - dr, j - dc))].add((i, j))
        found = []
        for (cid, a), seen in votes.items():
            fp = self.fp[cid]
            if len(seen) < 2 or len(seen) < len(fp) / 2:
                continue
            cells = {(a[0] + dr, a[1] + dc): p for (dr, dc), p in fp.items()}
            if any(not (0 <= i < NG and 0 <= j < NG) for (i, j) in cells):
                continue
            if any(c in ev and (ev[c] < 0 or not self.sim[p, ev[c]]) for c, p in cells.items()):
                continue
            found.append((cid, a, cells, seen))
        found.sort(key=lambda f: (-len(f[2]), f[0]))
        return found

    # ---------------- surprise under the class table ----------------
    def p_stroke(self, y, cell, s, alpha=0.5):
        k = CELL_ID[cell]
        tot = self.n_digits[y] + alpha * (self.P + 1)
        if s < 0:
            return (self.cblank[y, k] + alpha) / tot
        if self.codes is None:
            return (self.cstroke[y, k] @ self.sim[:, s] + alpha) / tot
        return (self.cstroke[y, k] @ self.codes[s] / self.B + alpha) / tot

    def posterior(self, items_s, items_c, alpha=0.5):
        lp = np.log(self.n_digits + 1.0)
        for y in range(10):
            for cell, s in items_s:
                lp[y] += math.log(self.p_stroke(y, cell, s, alpha))
            for cid, a in items_c:
                lp[y] += math.log((self.cchunk[y].get((cid, a), 0.0) + alpha) / (self.n_digits[y] + alpha * 50))
        lp -= lp.max(); p = np.exp(lp); return p / p.sum()

    # ---------------- learning from one digit ----------------
    def learn(self, ev, y, ep):
        self.n_digits[y] += 1
        cells = list(ev)
        self.rng.shuffle(cells)
        A = {c: ev[c] for c in cells[: len(cells) // 2]}; B = {c: ev[c] for c in cells[len(cells) // 2:]}
        # ---- test chunks: recognise on half A, score predictions on half B (unseen cells)
        found_A = self.recognise(A)
        for cid, a, cl, seen in found_A:
            for c, p in cl.items():
                if c in B and c not in A:
                    s = B[c]
                    base = self.p_stroke(y, c, s)
                    ok = s >= 0 and self.sim[p, s]
                    self.pred_n[cid] += 1; self.pred_hit[cid] += ok
                    pc = 0.8 if ok else 0.2 / (self.P + 1)
                    self.gain_pred[cid] += math.log2(pc) - math.log2(base)
        # ---- count this digit (full evidence): strokes, blanks, chunks
        for c, s in ev.items():
            k = CELL_ID[c]
            if s < 0:
                self.cblank[y, k] += 1
            elif self.codes is None:
                self.cstroke[y, k, s] += 1
            else:
                self.cstroke[y, k] += self.codes[s]
        found = self.recognise(ev)
        top_items, covered = [], set()
        bits_item = math.log2(self.V()) + math.log2(NC)
        for cid, a, cl, seen in found:
            if any(c in covered for c in cl):
                continue
            covered |= set(cl); top_items.append((("c", cid), a))
            self.cchunk[y][(cid, a)] += 1; self.used_by[cid].add(y); self.uses[cid] += 1; self.trials[cid] += 1
            # surprise absorbed: the seen strokes it stands for, minus one item to assert it
            absorbed = sum(-math.log2(self.p_stroke(y, c, ev[c])) + math.log2(NC) for c in seen)
            self.gain_lab[cid] += absorbed - bits_item
        top_items += [(("s", s), c) for c, s in ev.items() if s >= 0 and c not in covered]
        # ---- candidates (a): recurring pairs
        n_new = 0
        for x in range(len(top_items)):
            for z in range(x + 1, len(top_items)):
                (ka, ca), (kb, cb) = top_items[x], top_items[z]
                if ca > cb:
                    (ka, ca), (kb, cb) = (kb, cb), (ka, ca)
                off = (cb[0] - ca[0], cb[1] - ca[1])
                if max(abs(off[0]), abs(off[1])) > 4:
                    continue
                key = (ka, kb, off); self.pairs[key] += 1
                if self.pairs[key] == self.recur:
                    if self.make([(ka, (0, 0)), (kb, off)], ep, "pair") is not None:
                        n_new += 1
        # ---- surprise of this digit; candidates (b): search when surprised
        surprise = -sum(math.log2(self.p_stroke(y, c, s)) for c, s in ev.items()) / max(len(ev), 1)
        searched = 0
        if surprise > self.surprise_mean and self.beam > 0 and len(self.recent) > 500:
            n_new += self.search(ev, y, top_items, ep); searched = 1
        self.surprise_mean += 0.01 * (surprise - self.surprise_mean)
        # ---- clicks and deaths
        for cid in list(self.parts):
            if self.status[cid] != "provisional":
                continue
            g = self.gain_pred[cid] + self.gain_lab[cid]
            if self.trials[cid] >= 3 and g > self.cost(cid):
                self.status[cid] = "consolidated"; self.clicks.append((ep, cid, g))
            elif self.trials[cid] >= self.prune_after and g < 0:
                self.kill(cid)
            elif ep - self.origin_ep.get(cid, ep) > 3000 and self.trials[cid] < 3:
                self.kill(cid)                                   # never exercised
        # ---- remember for the beam's index
        rid = len(self.recent); self.recent.append((y, dict(ev)))
        for c, s in ev.items():
            if s >= 0:
                self.inv[(c, s)].add(rid)
        return len(top_items), surprise, n_new, searched

    def search(self, ev, y, top_items, ep):
        """Beam over combinations of this digit's items, ordered by pair counts (plus one slot
        of similar-but-uncounted items).  Score = bits of information the combination carries
        about the label, times its support, from the inverted index of recent digits."""
        leaves = [(c, s) for c, s in ev.items() if s >= 0]
        if len(leaves) < 3:
            return 0
        def support(combo):
            sets = [self.inv.get((c, s), set()) for c, s in combo]
            ids = set.intersection(*sets) if sets else set()
            return ids
        def score(combo):
            ids = support(combo)
            if len(ids) < 3:
                return -1.0, ids
            ys = np.array([self.recent[i][0] for i in ids])
            p_y = (np.mean(ys == y) * len(ids) + 0.5) / (len(ids) + 5)
            prior = (self.n_digits[y] + 1) / (self.n_digits.sum() + 10)
            return len(ids) * math.log2(p_y / prior), ids
        beams = [((c, s),) for c, s in leaves]
        best, best_score = None, 0.0
        for depth in range(self.depth):
            scored = []
            for combo in beams:
                sc, ids = score(combo)
                scored.append((sc, combo))
            scored.sort(key=lambda x: -x[0])
            beams = [c for sc, c in scored[: self.beam]]
            if scored and scored[0][0] > best_score and len(scored[0][1]) >= 2:
                best, best_score = scored[0][1], scored[0][0]
            ext = []
            for combo in beams:
                last = combo[-1]
                cands = [l for l in leaves if l not in combo]
                cands.sort(key=lambda l: -self.pairs.get((("s", last[1]), ("s", l[1]), (l[0][0] - last[0][0], l[0][1] - last[0][1])), 0))
                for l in cands[: self.beam]:
                    ext.append(combo + (l,))
                if cands:
                    ext.append(combo + (cands[self.rng.integers(len(cands))],))     # the creative slot
            beams = ext
            if not beams:
                break
        if best is None or best_score < 8.0:
            return 0
        cells = sorted(best)
        parts = [(("s", s), (c[0] - cells[0][0][0], c[1] - cells[0][0][1])) for c, s in cells]
        cid = self.make(parts, ep, "search")
        if cid is None:
            return 0
        self.gain_lab[cid] += best_score / 4          # the hit: an estimate, still to be tested
        self.log(f"  ep {ep}: search hit ({best_score:.0f} bits) -> chunk {cid} of {len(parts)} strokes for class {y}")
        return 1

    def cchunk_items(self):
        return [k for y in self.cchunk for k in self.cchunk[y]]

    # ---------------- label hidden ----------------
    def classify(self, ev, use_chunks=True):
        items_s = [(c, s) for c, s in ev.items()]
        items_c = []
        if use_chunks:
            for cid, a, cl, seen in self.recognise(ev):
                if self.status[cid] == "consolidated":
                    items_c.append((cid, a))
        return int(self.posterior(items_s, items_c).argmax())

    def complete(self, A, B, use_chunks=True):
        """Predict the strokes at B's inked cells from A (label hidden).  Returns hits, n."""
        y = self.classify(A, use_chunks)
        pred = {}
        if use_chunks:
            for cid, a, cl, seen in self.recognise(A):
                if self.status[cid] == "consolidated":
                    for c, p in cl.items():
                        pred.setdefault(c, p)
        hit = n = 0
        for c, s in B.items():
            if s < 0:
                continue
            n += 1
            p = pred.get(c)
            if p is None:
                row = self.cstroke[y, CELL_ID[c]]
                p = int((row @ self.sim).argmax()) if self.codes is None else int((self.codes @ row).argmax())
            hit += bool(self.sim[p, s])
        return hit, n


def sample(grid, rng, seen):
    ev, n_ink = {}, 0
    for k in rng.permutation(NC):
        c = CELLS[k]; ev[c] = int(grid[k]); n_ink += grid[k] >= 0
        if n_ink >= seen:
            break
    return ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=10000); ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seen", type=int, default=24); ap.add_argument("--recur", type=int, default=4)
    ap.add_argument("--beam", type=int, default=4); ap.add_argument("--tau-new", type=float, default=0.9)
    ap.add_argument("--sim", type=float, default=0.9, help="strokes match if pixel distance below this")
    ap.add_argument("--prune-after", type=int, default=12)
    ap.add_argument("--report", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--ps", type=int, default=3); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--graded", action="store_true", help="population codes per block; soft counts; dot-product matching")
    ap.add_argument("--match", type=float, default=0.5, help="graded: strokes match if code agreement >= this (1 = identical)")
    args = ap.parse_args(argv)
    configure(args.ps, args.stride)
    scale = math.sqrt(args.ps * args.ps / 9.0)          # pixel-distance thresholds scale with patch size
    if args.ps != 3:
        args.tau_new *= scale; args.sim *= scale
    rng = np.random.default_rng(args.seed); t0 = time.time()
    X, y = mnist.load("train"); Xt, yt = mnist.load("t10k")
    sel = rng.permutation(len(X))[:args.train]; X, y = X[sel], y[sel]
    Xt, yt = Xt[:args.test], yt[:args.test]
    voc = Vocab(args.tau_new, ink=2)
    for i in rng.permutation(len(X))[:3000]:
        for p in patches(X[i]):
            voc.learn(p)
    voc.finalize(min_count=20)
    grids = np.stack([voc.grid(im) for im in X]); grids_t = np.stack([voc.grid(im) for im in Xt])
    P = len(voc.means)
    d = np.sqrt(((voc.means[:, None] - voc.means[None]) ** 2).sum(-1))
    codes = None
    if args.graded:
        codes = graded_codes(voc, X, rng)
        agree = codes @ codes.T / 8
        sim = agree >= args.match
    else:
        sim = d < args.sim
    alg = sdr.Algebra(16, 41, args.seed)
    log = (lambda s: None) if args.quiet else print
    lib = Library(alg, sim, P, rng, args.recur, args.beam, args.prune_after, log, codes=codes, depth=args.depth)
    print(f"{P} primitives from {PS}x{PS} patches at stride {STRIDE} ({NG}x{NG} = {NC} cells); inked cells per digit {np.mean(grids >= 0) * NC:.0f}; "
          f"{'GRADED codes (8 blocks x 41, top-3 bumps), soft counts' if args.graded else 'hard ids'}; similar strokes per stroke {sim.sum(1).mean():.1f}  [{time.time()-t0:.0f}s]")
    print(f"training: label shown, random cells until {args.seen} inked seen; pairs -> chunk after {args.recur}; beam {args.beam} x depth {args.depth} when surprised")
    n_par = voc.means.size + (codes.size if codes is not None else 0) + lib.cstroke.size + lib.cblank.size
    print(f"parameters (numbers stored before any chunk): vocabulary {voc.means.size:,} + codes {(codes.size if codes is not None else 0):,} + class tables {lib.cstroke.size + lib.cblank.size:,} = {n_par:,}\n")
    print("   digits   chunks prov/cons/dead   levels (cons)   items/digit   surprise  searches  search-born cons")
    block = []
    for ep, (g, lab) in enumerate(zip(grids, y), 1):
        ev = sample(g, rng, args.seen)
        n_items, surprise, n_new, searched = lib.learn(ev, int(lab), ep)
        block.append((n_items, surprise, searched))
        if ep % args.report == 0:
            b = np.array(block); block = []
            st = defaultdict(int)
            for c, s in lib.status.items():
                st[s] += 1
            lv = defaultdict(int)
            for c, s in lib.status.items():
                if s == "consolidated":
                    lv[lib.level[c]] += 1
            sb = sum(1 for c, s in lib.status.items() if s == "consolidated" and lib.origin[c] == "search")
            print(f"  {ep:7d}   {st['provisional']:5d}/{st['consolidated']:4d}/{st['dead']:5d}      "
                  f"{' '.join(f'L{l}:{n}' for l, n in sorted(lv.items())):18s} {b[:,0].mean():6.1f}      {b[:,1].mean():5.2f}    {int(b[:,2].sum()):5d}     {sb:4d}   [{time.time()-t0:.0f}s]")
    # ---- diagnostics
    tried = [c for c in lib.parts if lib.trials[c] >= 3]
    if tried:
        gp = np.array([lib.gain_pred[c] / lib.trials[c] for c in tried]); gl = np.array([lib.gain_lab[c] / lib.trials[c] for c in tried])
        hr = np.array([lib.pred_hit[c] / max(lib.pred_n[c], 1) for c in tried]); ct = np.array([lib.cost(c) for c in tried])
        print(f"\nDIAG  {len(tried)} chunks tried >=3x: per trial  pred gain mean {gp.mean():+.2f} bits (25/50/75%: {np.percentile(gp,25):+.2f} {np.percentile(gp,50):+.2f} {np.percentile(gp,75):+.2f}),"
              f"  label gain mean {gl.mean():+.2f} bits;  prediction hit rate {hr.mean():.0%};  storage cost {ct.mean():.0f} bits;  trials mean {np.mean([lib.trials[c] for c in tried]):.1f}")
    # ---- library
    cons = [c for c, s in lib.status.items() if s == "consolidated"]
    n_chunk_par = sum(len(lib.parts[c]) * 3 for c in cons)
    print(f"\nlearned structure: {len(lib.pairs):,} pair counts, {len(lib.cchunk_items()):,} chunk-at-cell counts, {n_chunk_par:,} numbers in consolidated chunk cards")
    print(f"\nLIBRARY  {len(cons)} consolidated chunks ({sum(1 for s in lib.status.values() if s == 'dead')} tried and dropped); "
          f"clicks {len(lib.clicks)}; by origin: pairs {sum(1 for c in cons if lib.origin[c] == 'pair')}, search {sum(1 for c in cons if lib.origin[c] == 'search')}")
    print("  most valuable (bits of surprise absorbed when used + removed on unseen cells, minus storage):")
    for c in sorted(cons, key=lambda c: -(lib.gain_pred[c] + lib.gain_lab[c] - lib.cost(c)))[:10]:
        fp = lib.fp[c]; rs = [r for r, _ in fp]; cs = [k for _, k in fp]
        pic = [["." for _ in range(max(cs) + 1)] for _ in range(max(rs) + 1)]
        for (r, k) in fp:
            pic[r][k] = "#"
        print(f"    chunk {c:5d} L{lib.level[c]} {len(fp):2d} cells  unseen {lib.gain_pred[c]:7.0f}  absorbed {lib.gain_lab[c]:7.0f}  cost {lib.cost(c):4.0f}  "
              f"used {lib.uses[c]:5d}x in {sorted(lib.used_by[c])}  {lib.origin[c]:6s} {' '.join(''.join(r) for r in pic)}")
    # ---- tests
    print(f"\nTEST  label hidden, {args.test} digits")
    print("   strokes seen    class accuracy: strokes only / with chunks     completion of held-out inked cells: strokes only / with chunks")
    for seen in [4, 8, 16, 32]:
        acc = [[], []]; comp = [[0, 0], [0, 0]]
        for g, l in zip(grids_t, yt):
            ev = sample(g, rng, seen)
            for u in (0, 1):
                acc[u].append(lib.classify(ev, use_chunks=bool(u)) == int(l))
            B = sample(g, rng, 8)
            B = {c: s for c, s in B.items() if c not in ev}
            for u in (0, 1):
                h, n = lib.complete(ev, B, use_chunks=bool(u)); comp[u][0] += h; comp[u][1] += n
        print(f"   {seen:8d}           {np.mean(acc[0]):6.1%} / {np.mean(acc[1]):6.1%}                          "
              f"{comp[0][0]/max(comp[0][1],1):6.1%} / {comp[1][0]/max(comp[1][1],1):6.1%}")
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
