"""Integrate as you go.   python tally3.py [--train 5000] [--ps 5] [--graded] [--cap 20]

One cell at a time, in random order, nearly the whole digit.  After every cell the parse may
change: the new stroke votes for chunks; a chunk with at least half its cells present and none
contradicted may be placed, and may displace items it overlaps if it explains more; displaced
chunks dissolve back into strokes; repeat until nothing changes.  Working memory holds at most
`cap` top-level items: over the cap it must compact (a known chunk, else a provisional chunk from
the pair it holds that has co-occurred most) or drop its oldest stroke.

Learning is counting.  The class table counts every shown cell (label shown).  Pairs are counted
over the final parse.  A placed chunk predicts its unseen cells; when they arrive, the prediction
is scored in bits against the class table.  A chunk consolidates when its earnings (prediction +
label) exceed its storage, dies when clearly negative.  Search (beam over this digit's parse,
keyed by stroke families, run when the digit is surprising) proposes label-specific combinations.

Test: hold out a few inked cells, show everything else, ask the class, then the held-out cells,
answered from the parse (chunk covering the cell), else from the class opinion.
"""
import argparse, math, time
from collections import defaultdict
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2026_09_16" / "glimpse_loop"))   # sdr.py, mnist.py
import mnist

PS, STRIDE = 5, 2


def configure(ps, stride):
    global PS, STRIDE, POS, NG, CELLS, NC, CELL_ID
    PS, STRIDE = ps, stride
    POS = list(range(0, 28 - PS + 1, STRIDE)); NG = len(POS)
    CELLS = [(i, j) for i in range(NG) for j in range(NG)]; NC = len(CELLS)
    CELL_ID = {c: k for k, c in enumerate(CELLS)}


configure(PS, STRIDE)


def patches(img):
    return np.stack([img[r:r + PS, c:c + PS].reshape(-1) for r in POS for c in POS])


class Vocab:
    def __init__(self, tau_new, ink):
        self.tau_new, self.ink = tau_new, ink
        self.means = np.zeros((0, PS * PS), np.float32); self.counts = np.zeros(0, int)

    def blank(self, p): return (p > 0.5).sum() < self.ink
    def nearest(self, p):
        d = np.sqrt(((self.means - p) ** 2).sum(1)); i = int(d.argmin()); return i, float(d[i])
    def learn(self, p):
        if self.blank(p): return
        if len(self.means) == 0 or self.nearest(p)[1] > self.tau_new:
            self.means = np.vstack([self.means, p[None]]); self.counts = np.append(self.counts, 1); return
        i = self.nearest(p)[0]; self.counts[i] += 1
        self.means[i] += (p - self.means[i]) / min(self.counts[i], 500)
    def finalize(self, min_count):
        keep = self.counts >= min_count; self.means, self.counts = self.means[keep], self.counts[keep]
    def sense(self, p): return -1 if self.blank(p) else self.nearest(p)[0]
    def grid(self, img): return self.grids(img[None])[0]
    def grids(self, imgs):
        """All patches of many images -> nearest prototype, in one matrix product per batch."""
        out = []
        m2 = (self.means ** 2).sum(1)
        for lo in range(0, len(imgs), 200):
            P_ = np.stack([patches(im) for im in imgs[lo:lo + 200]])          # (b, NC, d)
            flat = P_.reshape(-1, P_.shape[-1])
            d2 = (flat ** 2).sum(1)[:, None] - 2 * flat @ self.means.T + m2[None, :]
            g = d2.argmin(1)
            g[(flat > 0.5).sum(1) < self.ink] = -1
            out.append(g.reshape(P_.shape[0], -1))
        return np.concatenate(out)


def skmeans(X, K, rng, iters=25):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cent = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        a = (X @ cent.T).argmax(1)
        for k in range(K):
            if (a == k).any():
                v = X[a == k].mean(0); cent[k] = v / (np.linalg.norm(v) + 1e-9)
    return (X @ cent.T), cent


def family_bumps(voc, rng, B=8, K=10, topk=3, temp=8.0):
    """Coarse-family graded codes: per block, K stroke families on a pixel subset; a stroke's
    bump is its softmax membership over families, top-k, summing to 1 per block."""
    M = voc.means - voc.means.mean(1, keepdims=True)
    P = len(M); bumps = np.zeros((P, B, K))
    for b in range(B):
        mask = rng.random(M.shape[1]) < 0.6
        cos, _ = skmeans(M[:, mask], K, rng)
        w = np.exp(temp * cos)
        thr = np.sort(w, 1)[:, -topk][:, None]
        w = np.where(w >= thr, w, 0.0); bumps[:, b, :] = w / w.sum(1, keepdims=True)
    return bumps


class HardReader:
    def __init__(self, P, sim):
        self.P, self.sim = P, sim.astype(float)
        self.c = np.zeros((10, NC, P)); self.n = np.zeros((10, NC))
    def add(self, y, k, s): self.c[y, k, s] += 1; self.n[y, k] += 1
    def probs(self, y, k, alpha=0.5):
        return (self.c[y, k] @ self.sim + alpha) / (self.n[y, k] + alpha * (self.P + 1))
    def prob(self, y, k, s, alpha=0.5): return float(self.probs(y, k, alpha)[s])
    def size(self): return self.c.size


class GradedReader:
    """Family mixture: P(s | y, cell) = geometric mean over blocks of  sum_f P(f | y, cell, b) P(s | f, b)."""
    def __init__(self, bumps):
        self.bumps = bumps; P, B, K = bumps.shape; self.P, self.B, self.K = P, B, K
        self.col = bumps.sum(0) + 1e-9                             # (B, K) family mass over the vocabulary
        self.ps_f = bumps / self.col[None]                         # (P, B, K)  P(s | f, b)
        self.c = np.zeros((10, NC, B, K)); self.n = np.zeros((10, NC))
    def add(self, y, k, s): self.c[y, k] += self.bumps[s]; self.n[y, k] += 1
    def probs(self, y, k, alpha=0.5):
        pf = (self.c[y, k] + alpha / self.K) / (self.n[y, k] + alpha)      # (B, K)
        per_block = np.einsum("bk,pbk->pb", pf, self.ps_f)                 # (P, B)
        return np.exp(np.log(per_block + 1e-12).mean(1)) + 1e-9
    def prob(self, y, k, s, alpha=0.5): return float(self.probs(y, k, alpha)[s])
    def size(self): return self.c.size


class Item:
    __slots__ = ("kind", "id", "anchor", "cells", "seen", "level", "t", "subst")
    def __init__(self, kind, id, anchor, cells, seen, level, t):
        self.kind, self.id, self.anchor, self.cells, self.seen, self.level, self.t = kind, id, anchor, cells, seen, level, t
        self.subst = None


class System:
    def __init__(self, P, sim, reader, families, rng, cfg, log):
        self.P, self.sim, self.R, self.fam, self.rng, self.cfg, self.log = P, sim, reader, families, rng, cfg, log
        self.nsim = [np.flatnonzero(sim[s])[np.argsort(-sim[s][np.flatnonzero(sim[s])])][:4] for s in range(P)] if sim.dtype != bool else \
                    [np.flatnonzero(sim[s])[:6] for s in range(P)]
        self.parts, self.fp, self.level, self.status, self.origin, self.born = {}, {}, {}, {}, {}, {}
        self.fparr = {}
        self._cards = None                      # padded arrays (rows, cols, strokes, mask) over card ids
        self.by_key = {}; self.index = defaultdict(list)
        self.gain = defaultdict(float); self.trials = defaultdict(int); self.uses = defaultdict(int); self.used_by = defaultdict(set)
        self.cchunk = defaultdict(lambda: defaultdict(float)); self.blank = np.zeros((10, NC)); self.nd = np.zeros(10)
        self.pairs = defaultdict(int); self.recent = []; self.inv = defaultdict(set); self.smean = 6.0; self.clicks = []
        self.sim0 = sim.copy(); self.canon = np.arange(P); self.slot_fill = defaultdict(lambda: defaultdict(int))
        self.cat_events = []
        self.learned = defaultdict(set)          # (cid, cell index in card) -> strokes readable as that part
        self.subst_count = defaultdict(int)      # (cid, k, S) -> validated uses
        self.subst_events = []
        self.used_by_count = defaultdict(lambda: defaultdict(int))

    # ---------------- chunk cards ----------------
    def cost(self, cid):
        V = self.P + 1 + len(self.parts); return math.log2(V) + len(self.parts[cid]) * (math.log2(V) + math.log2(81))

    def make(self, parts, ep, origin):
        fp = {}
        for (kind, i), (dr, dc) in parts:
            for (r, c), p in (self.fp[i] if kind == "c" else {(0, 0): int(self.canon[i])}).items():
                fp[(r + dr, c + dc)] = p
        if len(fp) > self.cfg["max_cells"]: return None
        r0 = min(r for r, _ in fp); c0 = min(c for _, c in fp)
        fp = {(r - r0, c - c0): p for (r, c), p in fp.items()}
        key = frozenset(fp.items())
        if key in self.by_key: return None
        cid = len(self.parts)
        self.parts[cid] = [((k, i), (dr - r0, dc - c0)) for (k, i), (dr, dc) in parts]
        self.fp[cid] = fp; self.by_key[key] = cid; self.born[cid] = ep; self.origin[cid] = origin
        self.fparr[cid] = (np.array([r for r, _ in fp]), np.array([c for _, c in fp]), np.array([p for p in fp.values()]))
        self._cards = None
        self.level[cid] = 1 + max((self.level[i] for (k, i), _ in parts if k == "c"), default=0)
        self.status[cid] = "provisional"
        for (dr, dc), p in fp.items():
            for q in self.nsim[p]:
                self.index[int(q)].append((cid, (dr, dc)))
        return cid

    def cards(self):
        if self._cards is None:
            n = len(self.parts); M = max((len(f) for f in self.fp.values()), default=1)
            R = np.zeros((n, M), int); C = np.zeros((n, M), int); Pp = np.zeros((n, M), int); Mk = np.zeros((n, M), bool)
            for cid, (rr, cc, pp) in self.fparr.items():
                k = len(rr); R[cid, :k] = rr; C[cid, :k] = cc; Pp[cid, :k] = pp; Mk[cid, :k] = True
            self._cards = (R, C, Pp, Mk)
        return self._cards

    def kill(self, cid):
        self.status[cid] = "dead"
        for p in list(self.index):
            self.index[p] = [(c, o) for c, o in self.index[p] if c != cid]

    # ---------------- the parse ----------------
    def new_digit(self):
        self.items, self.E, self.blanks, self.owner, self.pred, self.t, self.dropped = [], {}, set(), {}, {}, 0, 0
        self.Egrid = -np.ones((NG, NG), int)

    def leaves(self):
        return self.E

    def candidates(self, strokes):
        """Chunk placements suggested by these strokes: (cid, anchor) with >= half the chunk's
        cells present and matching, none contradicted.  All placements checked in one batch."""
        pairs = set()
        for (i, j), s in strokes:
            for cid, (dr, dc) in self.index[s]:
                if self.status[cid] != "dead":
                    pairs.add((cid, i - dr, j - dc))
        if not pairs:
            return {}
        arr = np.array(sorted(pairs)); cids, ar, ac = arr[:, 0], arr[:, 1], arr[:, 2]
        R0, C0, P0, M0 = self.cards()
        R = R0[cids] + ar[:, None]; C = C0[cids] + ac[:, None]; Pp = P0[cids]; Mk = M0[cids]
        inside = (R >= 0) & (R < NG) & (C >= 0) & (C < NG)
        ok = (inside | ~Mk).all(1)
        got = self.Egrid[np.clip(R, 0, NG - 1), np.clip(C, 0, NG - 1)]
        blank = ((got == -2) & Mk).any(1)
        obs = (got >= 0) & Mk
        n = obs.sum(1); ncell = Mk.sum(1)
        match = self.sim[np.where(Mk, Pp, 0), np.where(obs, got, 0)] | ~obs
        mism = (~match).sum(1)
        strong = ok & ~blank & (n >= 2) & (2 * n >= ncell)
        good = strong & (mism == 0)
        asif = strong & (mism == 1) & (n >= self.cfg["asif_min"]) if self.cfg["asif"] else np.zeros_like(good)
        out = {}
        for idx in np.flatnonzero(good | asif):
            m = Mk[idx]; o = obs[idx]
            cells = {(int(r), int(c)): int(p) for r, c, p in zip(R[idx][m], C[idx][m], Pp[idx][m])}
            seen = {(int(r), int(c)) for r, c in zip(R[idx][o], C[idx][o])}
            subst = None
            if asif[idx]:
                k = int(np.flatnonzero(~match[idx])[0])
                S, P = int(got[idx][k]), int(Pp[idx][k])
                cid = int(cids[idx])
                if S not in self.learned[(cid, k)]:
                    subst = (cid, k, S, P)                  # provisional: "read S as P here"
            out[(int(cids[idx]), (int(ar[idx]), int(ac[idx])))] = (cells, seen, subst)
        return out

    def place(self, cid, a, cells, seen):
        for it in [self.owner[c] for c in seen if c in self.owner and self.owner[c].kind == "c"]:
            self.dissolve(it)
        for c in seen:
            it = self.owner.get(c)
            if it is not None and it.kind == "s": self.items.remove(it)
        item = Item("c", cid, a, cells, set(seen), self.level[cid], self.t); self.items.append(item)
        for c in seen: self.owner[c] = item
        for c, p in cells.items():
            if c not in seen: self.pred.setdefault(c, []).append((cid, p))
        return item

    def dissolve(self, item):
        if item not in self.items: return
        self.items.remove(item)
        for c in item.cells:
            if c in self.pred: self.pred[c] = [(cid, p) for cid, p in self.pred[c] if cid != item.id]
        for c in item.seen:
            if self.owner.get(c) is item:
                s = Item("s", self.E[c], c, {c: self.E[c]}, {c}, 0, self.t); self.items.append(s); self.owner[c] = s

    def saved(self, item): return len(item.seen) - 1 if item.kind == "c" else 0

    def integrate(self, strokes):
        """Revise the parse until stable: place chunks that save more than what they displace."""
        frontier = list(strokes)
        for _ in range(8):
            cands = self.candidates(frontier)
            if not cands: break
            best, best_gain = None, 0
            for (cid, a), (cells, seen, subst) in cands.items():
                displaced = {id(self.owner[c]): self.owner[c] for c in seen if c in self.owner and self.owner[c].kind == "c"}
                if any(d.id == cid and d.anchor == a for d in displaced.values()): continue
                g = (len(seen) - 1) - sum(self.saved(d) for d in displaced.values()) - (1 if subst else 0)
                lv = self.level[cid] - max((d.level for d in displaced.values()), default=0)
                key = (g, lv, len(seen))
                if g > 0 or (g == 0 and lv > 0):
                    if best is None or key > best_gain: best, best_gain = (cid, a, cells, seen, displaced, subst), key
            if best is None: break
            cid, a, cells, seen, displaced, subst = best
            freed = [c for d in displaced.values() for c in d.seen if c not in seen]
            item = self.place(cid, a, cells, seen)
            item.subst = subst
            frontier = [(c, self.E[c]) for c in freed] + [(c, self.E[c]) for c in seen]

    def enforce_cap(self, y, ep):
        while len(self.items) > self.cfg["cap"]:
            best, bc = None, 0
            for x in range(len(self.items)):
                for z in range(x + 1, len(self.items)):
                    a, b = self.items[x], self.items[z]
                    if a.anchor > b.anchor: a, b = b, a
                    off = (b.anchor[0] - a.anchor[0], b.anchor[1] - a.anchor[1])
                    if max(abs(off[0]), abs(off[1])) > 5: continue
                    n = self.pairs.get(((a.kind, a.id), (b.kind, b.id), off), 0)
                    if n > bc: best, bc = (a, b, off), n
            n_prov = sum(1 for s in self.status.values() if s == "provisional")
            if best is not None and bc >= self.cfg["recur"] and n_prov < self.cfg["prov_max"]:
                a, b, off = best
                cid = self.make([((a.kind, a.id), (0, 0)), ((b.kind, b.id), off)], ep, "pressure")
                if cid is not None:
                    cand = self.candidates([(c, self.E[c]) for c in (a.seen | b.seen)])
                    hit = next(((k, v) for k, v in cand.items() if k[0] == cid), None)
                    if hit is not None:
                        (cid2, an), (cells, seen, subst) = hit; self.place(cid2, an, cells, seen); continue
            oldest = min((it for it in self.items if it.kind == "s"), key=lambda it: it.t, default=None)
            if oldest is None: break
            self.items.remove(oldest); del self.owner[oldest.anchor]; del self.E[oldest.anchor]; self.Egrid[oldest.anchor] = -1; self.dropped += 1

    # ---------------- one cell arrives ----------------
    def show(self, cell, s, y, ep, learn):
        self.t += 1; k = CELL_ID[cell]
        if s < 0:
            self.blanks.add(cell); self.Egrid[cell] = -2
            if learn: self.blank[y, k] += 1
            for cid, p in self.pred.pop(cell, []):                     # predicted a stroke, got blank
                if learn:
                    self.trials[cid] += 1
                    base = (self.blank[y, k] + 0.5) / (self.nd[y] + 1.0)
                    self.gain[cid] += math.log2(0.2 / self.P) - math.log2(base)
                for it in [it for it in self.items if it.kind == "c" and it.id == cid and cell in it.cells]: self.dissolve(it)
            return
        if learn: self.R.add(y, k, s)
        for cid, p in self.pred.pop(cell, []):                         # a prediction is tested
            if learn:
                self.trials[cid] += 1
                base = self.R.prob(y, k, s); pc = 0.8 if self.sim[p, s] else 0.2 / self.P
                self.gain[cid] += math.log2(pc) - math.log2(base)
            if not self.sim[p, s]:
                for it in [it for it in self.items if it.kind == "c" and it.id == cid and cell in it.cells]: self.dissolve(it)
        self.E[cell] = s; self.Egrid[cell] = s
        covering = [it for it in self.items if it.kind == "c" and cell in it.cells and cell not in it.seen]
        if covering:                                                   # one owner: the highest-level chunk
            it = max(covering, key=lambda it: (it.level, len(it.seen)))
            it.seen.add(cell); self.owner[cell] = it
        else:
            item = Item("s", s, cell, {cell: s}, {cell}, 0, self.t); self.items.append(item); self.owner[cell] = item
        self.integrate([(cell, s)])
        self.enforce_cap(y, ep)

    # ---------------- end of digit ----------------
    def finish(self, y, ep):
        self.nd[y] += 1
        for it in self.items:
            if it.kind == "c" and it.subst is not None:
                cid, k, S, P = it.subst
                hist = self.used_by_count[cid]
                agrees = (not hist) or (max(hist, key=hist.get) == y)
                if agrees:
                    self.subst_count[(cid, k, S)] += 1
                    if self.subst_count[(cid, k, S)] == self.cfg["asif_recur"]:
                        self.learned[(cid, k)].add(S)
                        self.subst_events.append((ep, cid, k, S, P, y))
                        self.log(f"  ep {ep}: learned to read stroke {S} as {P} in card {cid} slot {k} (class {y})")
                self.used_by_count[cid][y] += 1
        for it in self.items:
            if it.kind == "c":
                for c in it.seen:
                    self.slot_fill[(it.id, (c[0] - it.anchor[0], c[1] - it.anchor[1]))][self.E[c]] += 1
                self.cchunk[y][(it.id, (it.anchor[0] // 2, it.anchor[1] // 2))] += 1; self.uses[it.id] += 1; self.used_by[it.id].add(y)
                if it.subst is None: self.used_by_count[it.id][y] += 1
                post_c = self.posterior([], [(it.id, it.anchor)], []); post_p = self.posterior([(c, self.E[c]) for c in it.seen], [], [])
                self.gain[it.id] += math.log2(post_c[y] + 1e-9) - math.log2(post_p[y] + 1e-9)
        n_new = 0
        units = [((it.kind, it.id), it.anchor) for it in self.items]
        units += [(("s", s), c) for c, s in self.E.items() if self.owner[c].kind == "c"]
        for x in range(len(units)):
            for z in range(x + 1, len(units)):
                (ka, ca), (kb, cb) = units[x], units[z]
                if ca > cb: (ka, ca), (kb, cb) = (kb, cb), (ka, ca)
                off = (cb[0] - ca[0], cb[1] - ca[1])
                if max(abs(off[0]), abs(off[1])) > 3: continue
                key = (ka, kb, off); self.pairs[key] += 1
                if self.pairs[key] == self.cfg["recur"] and sum(1 for s in self.status.values() if s == "provisional") < self.cfg["prov_max"] \
                        and self.make([(ka, (0, 0)), (kb, off)], ep, "pair") is not None: n_new += 1
        surprise = -np.mean([math.log2(self.R.prob(y, CELL_ID[c], s)) for c, s in self.E.items()]) if self.E else 0.0
        searched = 0
        if surprise > self.smean and self.cfg["beam"] > 0 and len(self.recent) > 300:
            n_new += self.search(y, ep); searched = 1
        self.smean += 0.01 * (surprise - self.smean)
        for cid in list(self.parts):
            if self.status[cid] != "provisional": continue
            if self.trials[cid] >= 3 and self.gain[cid] > self.cost(cid):
                self.status[cid] = "consolidated"; self.clicks.append((ep, cid))
            elif (self.trials[cid] >= self.cfg["prune_after"] and self.gain[cid] < 0) or (ep - self.born[cid] > 600 and self.trials[cid] < 3):
                self.kill(cid)
        rid = len(self.recent); self.recent.append((y, dict(self.E)))
        for c, s in self.E.items(): self.inv[(c, int(self.fam[s]))].add(rid)
        return len(self.items), surprise, n_new, searched, self.dropped

    def categorize(self, ep):
        """Strokes that fill the same slot of the same card are interchangeable in that role.
        Merge strokes that are interchangeable in at least two roles; re-express cards; merge
        cards that became identical.  Returns a summary dict."""
        from itertools import combinations
        co = defaultdict(int)
        for slot, cnt in self.slot_fill.items():
            strong = sorted(s for s, n in cnt.items() if n >= self.cfg["fill_min"])
            for a, b in combinations(strong, 2):
                co[(a, b)] += 1
        parent = {s: s for s in range(self.P)}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        size = defaultdict(lambda: 1)
        for (a, b), n in sorted(co.items(), key=lambda x: -x[1]):
            if n < self.cfg["role_min"]: break
            ra, rb = find(a), find(b)
            if ra == rb or size[ra] + size[rb] > self.cfg["cat_max"]: continue
            parent[rb] = ra; size[ra] += size[rb]
        roots = np.array([find(s) for s in range(self.P)])
        groups = defaultdict(list)
        for s in range(self.P): groups[roots[s]].append(s)
        new_canon = np.array([min(groups[roots[s]]) for s in range(self.P)])
        changed = int((new_canon != self.canon).sum())
        cats = [g for g in groups.values() if len(g) >= 2]
        bits_before = sum(self.cost(c) for c, st in self.status.items() if st == "consolidated")
        merged = 0
        if changed:
            self.canon = new_canon
            alive = [c for c, st in self.status.items() if st != "dead"]
            self.by_key = {}
            redirect = {}
            for cid in sorted(alive):
                fp = {off: int(self.canon[p]) for off, p in self.fp[cid].items()}
                self.fp[cid] = fp
                key = frozenset(fp.items())
                if key in self.by_key:
                    surv = self.by_key[key]
                    self.uses[surv] += self.uses[cid]; self.gain[surv] += self.gain[cid]; self.trials[surv] += self.trials[cid]
                    self.used_by[surv] |= self.used_by[cid]
                    if self.status[cid] == "consolidated" and self.status[surv] != "consolidated": self.status[surv] = "consolidated"
                    self.status[cid] = "dead"; redirect[cid] = surv; merged += 1
                else:
                    self.by_key[key] = cid
            for cid in alive:
                if self.status[cid] == "dead": continue
                self.parts[cid] = [((k, redirect.get(i, i) if k == "c" else int(self.canon[i])), off) for (k, i), off in self.parts[cid]]
                fp = self.fp[cid]
                self.fparr[cid] = (np.array([r for r, _ in fp]), np.array([c for _, c in fp]), np.array([p for p in fp.values()]))
            self._cards = None
            newpairs = defaultdict(int)
            for (ka, kb, off), n in self.pairs.items():
                ka2 = (ka[0], redirect.get(ka[1], ka[1]) if ka[0] == "c" else int(self.canon[ka[1]]))
                kb2 = (kb[0], redirect.get(kb[1], kb[1]) if kb[0] == "c" else int(self.canon[kb[1]]))
                newpairs[(ka2, kb2, off)] += n
            self.pairs = newpairs
            same = self.canon[:, None] == self.canon[None, :]
            self.sim = self.sim0 | same
            self.nsim = [np.flatnonzero(self.sim[s])[:8] for s in range(self.P)]
            self.index = defaultdict(list)
            for cid in alive:
                if self.status[cid] == "dead": continue
                for (dr, dc), p in self.fp[cid].items():
                    for q in self.nsim[p]: self.index[int(q)].append((cid, (dr, dc)))
        bits_after = sum(self.cost(c) for c, st in self.status.items() if st == "consolidated")
        ev = dict(ep=ep, categories=len(cats), strokes=sum(len(g) for g in cats), largest=max((len(g) for g in cats), default=0),
                  merged=merged, bits_before=bits_before, bits_after=bits_after)
        self.cat_events.append(ev)
        self.log(f"  ep {ep}: {ev['categories']} categories over {ev['strokes']} strokes (largest {ev['largest']}); "
                 f"{merged} cards merged; library {bits_before:.0f} -> {bits_after:.0f} bits")
        return ev

    def search(self, y, ep):
        leaves = [(c, s) for c, s in self.E.items()]
        if len(leaves) < 3: return 0
        prior = (self.nd[y] + 1) / (self.nd.sum() + 10)
        def score(combo):
            ids = set.intersection(*[self.inv.get((c, int(self.fam[s])), set()) for c, s in combo])
            if len(ids) < 3: return -1.0
            ys = np.array([self.recent[i][0] for i in ids]); p_y = (np.sum(ys == y) + 0.5) / (len(ids) + 5)
            return len(ids) * math.log2(p_y / prior)
        beams = [(l,) for l in leaves]; best, bs = None, 0.0
        for depth in range(self.cfg["depth"]):
            scored = sorted(((score(c), c) for c in beams), key=lambda x: -x[0])
            beams = [c for sc, c in scored[: self.cfg["beam"]]]
            if scored and len(scored[0][1]) >= 2 and scored[0][0] > bs: best, bs = scored[0][1], scored[0][0]
            ext = []
            for combo in beams:
                last = combo[-1]; cands = [l for l in leaves if l not in combo]
                cands.sort(key=lambda l: -self.pairs.get((("s", last[1]), ("s", l[1]), (l[0][0] - last[0][0], l[0][1] - last[0][1])), 0))
                ext += [combo + (l,) for l in cands[: self.cfg["beam"]]]
                if cands: ext.append(combo + (cands[self.rng.integers(len(cands))],))
            beams = ext
        if best is None or bs < self.cfg["hit"]: return 0
        cells = sorted(best); o = cells[0][0]
        cid = self.make([(("s", s), (c[0] - o[0], c[1] - o[1])) for c, s in cells], ep, "search")
        if cid is None: return 0
        self.gain[cid] += bs / 4; self.log(f"  ep {ep}: search hit {bs:.0f} bits -> chunk {cid} ({len(cells)} strokes) for class {y}")
        return 1

    # ---------------- reading out ----------------
    def posterior(self, strokes, chunks, blanks, alpha=0.5):
        lp = np.log(self.nd + 1.0)
        for y in range(10):
            for c, s in strokes: lp[y] += math.log(self.R.prob(y, CELL_ID[c], s))
            for cid, a in chunks: lp[y] += math.log((self.cchunk[y].get((cid, (a[0] // 2, a[1] // 2)), 0.0) + alpha) / (self.nd[y] + alpha * 50))
            for c in blanks: lp[y] += math.log((self.blank[y, CELL_ID[c]] + alpha) / (self.nd[y] + alpha * 2))
        lp -= lp.max(); p = np.exp(lp); return p / p.sum()

    def opinion(self, use_chunks):
        strokes = [(c, self.E[c]) for c in self.E if self.owner[c].kind == "s" or not use_chunks]
        chunks = [(it.id, it.anchor) for it in self.items if it.kind == "c"] if use_chunks else []
        return self.posterior(strokes, chunks, self.blanks)

    def predict_cell(self, cell, y, use_chunks):
        if use_chunks:
            for it in self.items:
                if it.kind == "c" and cell in it.cells and cell not in it.seen: return it.cells[cell], "chunk"
        return int(self.R.probs(y, CELL_ID[cell]).argmax()), "table"


def run_digit(sys_, grid, y, ep, rng, learn, holdout=0, use_chunks=True):
    sys_.new_digit()
    inked = [k for k in range(NC) if grid[k] >= 0]
    held = set(rng.choice(inked, min(holdout, len(inked)), replace=False).tolist()) if holdout else set()
    order = [k for k in rng.permutation(NC) if k not in held]
    for k in order:
        sys_.show(CELLS[k], int(grid[k]), y, ep, learn)
    return held


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=5000); ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--ps", type=int, default=5); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--graded", action="store_true"); ap.add_argument("--match", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=20); ap.add_argument("--recur", type=int, default=4)
    ap.add_argument("--beam", type=int, default=8); ap.add_argument("--depth", type=int, default=4); ap.add_argument("--hit", type=float, default=8.0)
    ap.add_argument("--max-cells", type=int, default=30); ap.add_argument("--prune-after", type=int, default=12)
    ap.add_argument("--prov-max", type=int, default=400)
    ap.add_argument("--categories", action="store_true"); ap.add_argument("--cat-every", type=int, default=500)
    ap.add_argument("--asif", action="store_true", help="allow one as-if substitution per placement, learned when validated")
    ap.add_argument("--asif-min", type=int, default=4); ap.add_argument("--asif-recur", type=int, default=3)
    ap.add_argument("--fill-min", type=int, default=3); ap.add_argument("--role-min", type=int, default=2); ap.add_argument("--cat-max", type=int, default=12)
    ap.add_argument("--holdout", type=int, default=6); ap.add_argument("--report", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="train on only the first N digits (vocabulary still from all)")
    args = ap.parse_args(argv)
    configure(args.ps, args.stride); rng = np.random.default_rng(args.seed); t0 = time.time()
    scale = math.sqrt(args.ps * args.ps / 9.0); tau_new, sim_thr = 0.9 * scale, 0.9 * scale
    X, y = mnist.load("train"); Xt, yt = mnist.load("t10k")
    sel = rng.permutation(len(X))[:args.train]; X, y = X[sel], y[sel]; Xt, yt = Xt[:args.test], yt[:args.test]
    voc = Vocab(tau_new, ink=2)
    for i in rng.permutation(len(X))[:3000]:
        for p in patches(X[i]): voc.learn(p)
    voc.finalize(min_count=20); P = len(voc.means)
    grids = voc.grids(X); grids_t = voc.grids(Xt)
    d = np.sqrt(((voc.means[:, None] - voc.means[None]) ** 2).sum(-1))
    if args.graded:
        bumps = family_bumps(voc, rng)
        nb = bumps / (np.linalg.norm(bumps, axis=2, keepdims=True) + 1e-9)
        agree = np.einsum("pbk,qbk->pq", nb, nb) / bumps.shape[1]
        sim = agree >= args.match; reader = GradedReader(bumps)
    else:
        sim = d < sim_thr; reader = HardReader(P, sim)
    np.fill_diagonal(sim, True)
    fam, _ = skmeans(voc.means - voc.means.mean(1, keepdims=True), 40, rng); fam = fam.argmax(1)
    cfg = dict(cap=args.cap, recur=args.recur, beam=args.beam, depth=args.depth, hit=args.hit, max_cells=args.max_cells,
               prune_after=args.prune_after, prov_max=args.prov_max, fill_min=args.fill_min, role_min=args.role_min, cat_max=args.cat_max,
               asif=args.asif, asif_min=args.asif_min, asif_recur=args.asif_recur)
    log = (lambda s: None) if args.quiet else print
    S = System(P, sim, reader, fam, rng, cfg, log)
    print(f"{P} primitives, {PS}x{PS} patches stride {STRIDE} ({NC} cells), inked/digit {np.mean(grids >= 0) * NC:.0f}; "
          f"{'graded (8 blocks x 10 families, top-3)' if args.graded else 'hard ids'}, similar strokes/stroke {sim.sum(1).mean():.1f}; "
          f"WM cap {args.cap} items; class table {reader.size():,} numbers  [{time.time()-t0:.0f}s]")
    print("   digits   chunks prov/cons/dead   levels (cons)                 items at end  dropped/digit  surprise  searches  search-born cons")
    block = []
    n_train = args.limit or len(grids)
    for ep, (g, lab) in enumerate(zip(grids[:n_train], y[:n_train]), 1):
        run_digit(S, g, int(lab), ep, rng, learn=True)
        block.append(S.finish(int(lab), ep))
        if args.categories and ep % args.cat_every == 0 and ep >= 1000:
            S.categorize(ep)
        if ep % args.report == 0:
            b = np.array(block, float); block = []
            st = defaultdict(int); lv = defaultdict(int)
            for c, s in S.status.items():
                st[s] += 1
                if s == "consolidated": lv[S.level[c]] += 1
            sb = sum(1 for c, s in S.status.items() if s == "consolidated" and S.origin[c] == "search")
            print(f"  {ep:7d}   {st['provisional']:5d}/{st['consolidated']:4d}/{st['dead']:5d}   {' '.join(f'L{l}:{n}' for l, n in sorted(lv.items())):32s} "
                  f"{b[:,0].mean():5.1f}        {b[:,4].mean():5.1f}        {b[:,1].mean():5.2f}    {int(b[:,3].sum()):5d}     {sb:4d}   [{time.time()-t0:.0f}s]")
    cons = [c for c, s in S.status.items() if s == "consolidated"]
    spec = sum(1 for c in cons if len(S.used_by[c]) < 5)
    print(f"\nchunks used by fewer than 5 classes: {spec} of {len(cons)}")
    if args.asif:
        n_learned = sum(len(v) for v in S.learned.values())
        tried = len(S.subst_count)
        by_class = defaultdict(int)
        for ep_, cid, k, s_, p_, yy in S.subst_events: by_class[yy] += 1
        print(f"as-if substitutions: {tried} tried, {n_learned} learned (validated {args.asif_recur}x); by class: {dict(sorted(by_class.items()))}")
        cards_l = {c for (c, k), v in S.learned.items() if v}
        print(f"  learned in {len(cards_l)} cards, of which consolidated: {sum(1 for c in cards_l if S.status[c] == 'consolidated')}")
    for ev in S.cat_events:
        print(f"  categories at ep {ev['ep']}: {ev['categories']} groups over {ev['strokes']} strokes (largest {ev['largest']}), "
              f"{ev['merged']} cards merged, library {ev['bits_before']:.0f} -> {ev['bits_after']:.0f} bits")
    print(f"\nLIBRARY  {len(cons)} consolidated chunks ({sum(1 for s in S.status.values() if s == 'dead')} dropped); origin: "
          f"pairs {sum(1 for c in cons if S.origin[c] == 'pair')}, pressure {sum(1 for c in cons if S.origin[c] == 'pressure')}, search {sum(1 for c in cons if S.origin[c] == 'search')}")
    for c in sorted(cons, key=lambda c: -(S.gain[c] - S.cost(c)))[:8]:
        fp = S.fp[c]; rs = [r for r, _ in fp]; cs = [k for _, k in fp]
        pic = [["." for _ in range(max(cs) + 1)] for _ in range(max(rs) + 1)]
        for (r, k) in fp: pic[r][k] = "#"
        print(f"    chunk {c:5d} L{S.level[c]} {len(fp):2d} cells  gain {S.gain[c]:7.0f} cost {S.cost(c):4.0f}  used {S.uses[c]:5d}x in {sorted(S.used_by[c])}  {S.origin[c]:8s} {' '.join(''.join(r) for r in pic)}")
    print(f"\nTEST  {args.test} digits: hold out {args.holdout} inked cells, show all other cells one at a time, then ask")
    res = defaultdict(list)
    for g, lab in zip(grids_t, yt):
        for name, uc, cap in [("table, all cells", False, 10 ** 6), ("strokes only, WM cap", False, args.cap), ("with library, WM cap", True, args.cap)]:
            S.cfg["cap"] = cap
            if not uc: saved = S.index; S.index = defaultdict(list)
            held = run_digit(S, g, int(lab), 0, np.random.default_rng(int(lab) * 1000 + len(res[name])), learn=False, holdout=args.holdout, use_chunks=uc)
            post = S.opinion(uc); yh = int(post.argmax())
            res[name].append(("class", yh == int(lab)))
            for k in held:
                p, how = S.predict_cell(CELLS[k], yh, uc)
                res[name].append(("comp", bool(S.sim[p, int(g[k])]))); res[name].append(("how", how == "chunk"))
            if not uc: S.index = saved
    S.cfg["cap"] = args.cap
    for name, r in res.items():
        cls = np.mean([v for k, v in r if k == "class"]); comp = np.mean([v for k, v in r if k == "comp"]); how = np.mean([v for k, v in r if k == "how"])
        print(f"   {name:22s} class {cls:6.1%}   held-out cells predicted {comp:6.1%}" + (f"   ({how:.0%} answered by a chunk)" if name.startswith("with") else ""))
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
