"""The glimpse loop on MNIST grids  (DESIGN.md Part XIV).

Same loop as agent.py, on a 4x4 grid of stroke prototypes, with everything the toy lacked:
    noisy sensing   - a stroke matches an expected stroke if their codes overlap enough
    a program       - rows: exemplar digits (with label ⊗ class) and chunks (recurring stroke
                      groups at relative poses); exemplars are re-written in chunk vocabulary
    rotation        - an operator pose: rotate positions, map strokes to their rotated prototypes
    a learned look policy - situations (as bundles) -> the computed VI.5 target, by lookup
    metacognition   - situations at the moment of declaring -> how often that went wrong

Hypotheses are evaluated for all (row, anchor, rotation) at once with numpy; working memory is
still one provenance-tagged bundle read back through the algebra.
"""
import math
from collections import defaultdict
import numpy as np
import mnist

G = mnist.GRID
NC = G * G
CELLS = [(i, j) for i in range(G) for j in range(G)]
NEIGH4 = mnist.NEIGH4
TAGS = ("observed", "inferred", "assumed")


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def normalize(fp):
    """{(i, j): p} -> {(dr, dc): p} with the lexicographically smallest cell at (0, 0)."""
    r0, c0 = min(fp)
    return {(i - r0, j - c0): p for (i, j), p in fp.items()}


def fkey(fp):
    return frozenset(fp.items())


# ====================================================================== working memory
class WM:
    def __init__(self, alg, poses, table, tags):
        self.alg, self.poses, self.table, self.tags = alg, poses, table, tags
        self.W = alg.empty(); self.shadow = {}; self.order = []; self.peak = 0

    def clear(self):
        self.W[:] = 0; self.shadow.clear(); self.order.clear(); self.peak = 0

    def code(self, tag, rid, cell):
        a = self.alg
        return a.bind(a.bind(self.tags[tag], self.table.name(rid)), self.poses(cell))

    def add(self, tag, rid, cell, m=1):
        key = (tag, rid, cell)
        self.alg.add(self.W, self.code(*key), m)
        if key not in self.shadow:
            self.order.append(key)
        self.shadow[key] = self.shadow.get(key, 0) + m
        self.peak = max(self.peak, len(self.shadow))

    def remove(self, tag, rid, cell):
        key = (tag, rid, cell)
        m = self.shadow.pop(key, 0)
        if m:
            self.alg.sub(self.W, self.code(*key), m); self.order.remove(key)
        return m

    def remove_tag(self, tag):
        for key in [k for k in self.order if k[0] == tag]:
            self.remove(*key)

    def read(self, tags=("observed", "inferred")):
        items = []
        names = self.table.names
        for tag in tags:
            keys = self.alg.bind(self.tags[tag][None, :], self.poses.cell_codes)
            pres = self.alg.presence(self.alg.unbind_many(self.W, keys), names)
            for ci, ri in zip(*np.nonzero(pres > 0)):
                items.append((tag, int(ri), self.poses.cells[ci], int(pres[ci, ri])))
        ghosts = sum(1 for tag, rid, cell, _ in items if (tag, rid, cell) not in self.shadow)
        return items, ghosts

    def pressure(self):
        return self.alg.ghost_estimate(self.W, len(TAGS) * NC * self.table.V)

    def load(self):
        return len(self.shadow)


# ====================================================================== the table
class DRow:
    __slots__ = ("id", "name", "label", "parts", "status", "born", "uses", "top_uses", "worked",
                 "last_used", "consolidated_at")

    def __init__(self, id, name, label, parts, status, born):
        self.id, self.name, self.label, self.parts, self.status, self.born = id, name, label, parts, status, born
        self.uses, self.top_uses, self.worked, self.last_used, self.consolidated_at = 0, 0, 0, born, None


class DigitTable:
    """ids 0..P-1 are stroke prototypes, P is EMPTY, rows start at P+1."""

    def __init__(self, alg, C, tau_match=4, blank_overlap=6, prov_max=1500, click_margin=0.0, prune_margin=10.0,
                 require_test=True, log=None):
        self.alg, self.C = alg, C
        self.P = C.EMPTY_ROW
        self.EMPTY = self.P
        O = C.overlap.astype(np.int16).copy()                 # graded stroke similarity, 0..B
        O[self.EMPTY, :] = 0; O[:, self.EMPTY] = 0; O[self.EMPTY, self.EMPTY] = blank_overlap
        self.O, self.tau_match = O, tau_match
        self.sim = O >= tau_match
        np.fill_diagonal(self.sim, True)
        self.rows = []
        self.by_key = {}
        self.prov_max, self.click_margin, self.prune_margin, self.require_test = prov_max, click_margin, prune_margin, require_test
        self.log = log or (lambda s: None)
        self._names = None; self._fps = {}; self._placed = []; self._placed_arr = None; self._ngrid = None
        self.events = []

    # ---------- ids ----------
    def is_prim(self, rid): return rid <= self.P
    def row(self, rid): return self.rows[rid - self.P - 1]
    def name(self, rid): return self.C.codes[rid] if rid <= self.P else self.row(rid).name
    @property
    def V(self): return self.P + 1 + len(self.rows)
    @property
    def names(self):
        if self._names is None:
            self._names = np.vstack([self.C.codes] + [r.name[None, :] for r in self.rows]) if self.rows else self.C.codes
        return self._names

    def label_of(self, rid): return None if rid <= self.P else self.row(rid).label
    def active(self): return [r for r in self.rows if r.status != "forgotten"]
    def chunks(self, status=None): return [r for r in self.active() if r.label is None and (status is None or r.status == status)]
    def exemplars(self, status=None): return [r for r in self.active() if r.label is not None and (status is None or r.status == status)]

    # ---------- footprints ----------
    def footprint(self, rid):
        if rid <= self.P:
            return {(0, 0): rid}
        if rid in self._fps:
            return self._fps[rid]
        fp = {}
        for pid, (dr, dc) in self.row(rid).parts:
            for (r, c), p in self.footprint(pid).items():
                fp[(r + dr, c + dc)] = p
        self._fps[rid] = fp
        return fp

    def _place(self, fp, exemplar):
        """(16 anchors, 16 cells) canvas placements of a normalized footprint, -1 = don't care.
        An exemplar is a whole digit: its cells outside the ink are explicitly EMPTY."""
        out = -np.ones((NC, NC), np.int16)
        for a, (ai, aj) in enumerate(CELLS):
            cells = [((ai + dr, aj + dc), p) for (dr, dc), p in fp.items()]
            if all(0 <= i < G and 0 <= j < G for (i, j), _ in cells):
                if exemplar:
                    out[a, :] = self.EMPTY
                for (i, j), p in cells:
                    out[a, i * G + j] = p
        return out

    def placed(self):
        """(n_rows, 16 anchors, 16 cells) for all rows (forgotten rows are all -1)."""
        if self._placed_arr is None or len(self._placed) != len(self.rows):
            for r in self.rows[len(self._placed):]:
                self._placed.append(self._place(self.footprint(r.id), r.label is not None))
            self._placed_arr = np.stack(self._placed) if self._placed else np.zeros((0, NC, NC), np.int16)
        return self._placed_arr

    def ngrid(self):
        """(n_rows, 4, 7) normalized footprint grids (origin at column 3), for containment tests."""
        if self._ngrid is None or len(self._ngrid) != len(self.rows):
            g = -np.ones((len(self.rows), G, 2 * G - 1), np.int16)
            for k, r in enumerate(self.rows):
                if r.status == "forgotten":
                    continue
                for (dr, dc), p in self.footprint(r.id).items():
                    g[k, dr, dc + G - 1] = p
            self._ngrid = g
        return self._ngrid

    def _dirty(self, fps=True):
        self._names = None
        if fps:
            self._fps = {}; self._ngrid = None

    # ---------- allocation ----------
    def propose(self, fp, ep, label=None, prefix="p"):
        key = fkey(fp)
        if key in self.by_key:
            r = self.row(self.by_key[key])
            if r.status != "forgotten":
                r.last_used = ep
                return r, False
        rid = self.P + 1 + len(self.rows)
        row = DRow(rid, self.alg.random(), label, self.factorize(fp), "provisional", ep)
        row.label = label
        self.rows.append(row); self.by_key[key] = rid
        self._dirty(fps=False); self._fps[rid] = dict(fp)
        self.evict(ep)
        return row, True

    def evict(self, ep):
        act = self.active()
        n_prov = sum(1 for r in act if r.status == "provisional")
        if n_prov <= self.prov_max:
            return
        refd = {pid for r in act for pid, _ in r.parts}
        prov = [r for r in act if r.status == "provisional" and r.id not in refd]
        while n_prov > self.prov_max and prov:
            v = min(prov, key=lambda r: (r.top_uses > 0, r.last_used, r.born))
            self.forget(v); prov.remove(v); n_prov -= 1

    def referenced(self, rid):
        return any(pid == rid for r in self.active() for pid, _ in r.parts)

    def parent_map(self):
        pm = defaultdict(set)
        for q in self.active():
            for pid, _ in q.parts:
                if not self.is_prim(pid):
                    pm[pid].add(q.id)
        return pm

    def ancestors_from(self, rid, pm):
        out, stack = set(), [rid]
        while stack:
            x = stack.pop()
            for q in pm.get(x, ()):
                if q not in out:
                    out.add(q); stack.append(q)
        return out

    def forget(self, r):
        r.status = "forgotten"; r.parts = []
        self._fps.pop(r.id, None); self._placed[r.id - self.P - 1] = -np.ones((NC, NC), np.int16)
        self._placed_arr = None; self._ngrid = None

    # ---------- containment (fuzzy, by cells) ----------
    def contains(self, fp, rows_mask=None):
        """For a normalized footprint: number of rows that contain it at some offset (strokes
        matching by code overlap).  Returns per-row occurrence indicator (n_rows,) bool."""
        g = self.ngrid()
        n = len(g)
        cells = list(fp.items())
        ok = np.zeros(n, bool)
        for oi in range(G):
            for oj in range(-(G - 1), G):
                idx_r = [dr + oi for (dr, dc), _ in cells]
                idx_c = [dc + oj + G - 1 for (dr, dc), _ in cells]
                if min(idx_r) < 0 or max(idx_r) >= G or min(idx_c) < 0 or max(idx_c) >= 2 * G - 1:
                    continue
                got = g[:, idx_r, idx_c]                                          # (n, m)
                want = np.array([p for _, p in cells])[None, :]
                hit = (got >= 0) & self.sim[np.where(got >= 0, got, 0), want]
                ok |= hit.all(1)
        return ok

    def refs_fp(self, fp, exclude=-1):
        ok = self.contains(fp)
        n = 0
        for k, r in enumerate(self.rows):
            if ok[k] and r.id != exclude and r.status != "forgotten" and len(self.footprint(r.id)) > len(fp) \
                    and (r.status == "consolidated" or r.top_uses >= 2):
                n += 1
        return n

    # ---------- description length ----------
    def bits_item(self): return math.log2(self.V) + math.log2(NC)
    def bits_part(self): return math.log2(self.V) + math.log2(NC)
    def bits_name(self): return math.log2(self.V)
    def row_cost(self, r): return self.bits_name() + len(r.parts) * self.bits_part()
    def gain(self, r):
        k = len(r.parts)
        refs = self.refs_fp(self.footprint(r.id), exclude=r.id) if r.label is None else 0
        return r.top_uses * (k - 1) * self.bits_item() + refs * (k - 1) * self.bits_part() - self.row_cost(r)
    def table_bits(self, rows=None):
        return sum(self.row_cost(r) for r in (rows if rows is not None else self.active()) if r.status == "consolidated")

    # ---------- factorization with consolidated chunks (greedy, fuzzy) ----------
    def factorize(self, fp, exclude=()):
        remaining = dict(fp)
        parts = []
        cands = [r for r in self.chunks("consolidated") if r.id not in exclude and len(self.footprint(r.id)) < len(fp)]
        cands.sort(key=lambda r: (-len(self.footprint(r.id)), r.id))
        for cand in cands:
            cfp = self.footprint(cand.id)
            placed = True
            while placed:
                placed = False
                for (r, c) in sorted(remaining):
                    cells = [((r + dr, c + dc), p) for (dr, dc), p in cfp.items()]
                    if all(rc in remaining and self.sim[remaining[rc], p] for rc, p in cells):
                        for rc, _ in cells:
                            del remaining[rc]
                        parts.append((cand.id, (r, c))); placed = True
                        break
        for rc, p in remaining.items():
            parts.append((p, rc))
        parts.sort(key=lambda x: x[1])
        return parts

    def refactor_all(self):
        pm = self.parent_map()
        for r in self.active():
            fp = self.footprint(r.id)
            parts = self.factorize(fp, exclude={r.id} | self.ancestors_from(r.id, pm))
            if len(parts) < len(r.parts):
                r.parts = parts

    def ancestors(self, rid):
        out = set()
        for q in self.active():
            if any(pid == rid for pid, _ in q.parts):
                out.add(q.id); out |= self.ancestors(q.id)
        return out

    # ---------- clicks and their mirror ----------
    def clicks(self, ep, prune=False):
        done = []
        while True:
            cands = [r for r in self.active() if r.status == "provisional" and (r.worked >= 1 or not self.require_test)]
            cands = [(self.gain(r), r) for r in cands]
            cands = [(g, r) for g, r in cands if g > self.click_margin]
            if not cands:
                break
            g, r = max(cands, key=lambda x: (x[0], -x[1].id))
            r.status = "consolidated"; r.consolidated_at = ep
            self.events.append((ep, "click", r.id, g))
            done.append(r)
            if r.label is None:
                self.refactor_with(r)
        if prune:
            bad = [(self.gain(r), r) for r in self.chunks("consolidated")]
            for g, r in sorted(bad, key=lambda x: x[0]):
                if g < -self.prune_margin:
                    for q in self.active():
                        if any(pid == r.id for pid, _ in q.parts):
                            q.parts = self._inline(q.parts, r)
                    self.events.append((ep, "unlearn", r.id, g))
                    self.forget(r)
        return done

    def refactor_with(self, chunk):
        """Re-express the rows that contain the new chunk (library refactoring, IV.4)."""
        ok = self.contains(self.footprint(chunk.id))
        pm = self.parent_map()
        for k in np.flatnonzero(ok):
            r = self.rows[k]
            if r.id == chunk.id or r.status == "forgotten":
                continue
            parts = self.factorize(self.footprint(r.id), exclude={r.id} | self.ancestors_from(r.id, pm))
            if len(parts) < len(r.parts):
                r.parts = parts

    def _inline(self, parts, child):
        out = []
        for pid, (dr, dc) in parts:
            if pid == child.id:
                out += [(p2, (r2 + dr, c2 + dc)) for p2, (r2, c2) in child.parts]
            else:
                out.append((pid, (dr, dc)))
        return sorted(out, key=lambda x: x[1])

    # ---------- pretty ----------
    def label_str(self, rid):
        if rid <= self.P:
            return f"s{rid}" if rid < self.P else "."
        r = self.row(rid)
        return f"{'C' if r.status == 'consolidated' else 'p'}{rid}"

    def describe(self, r):
        return " + ".join(f"{self.label_str(pid)}@({dr},{dc})" for pid, (dr, dc) in r.parts)


# ====================================================================== the agent
class Hyp:
    __slots__ = ("rid", "anchor", "rot", "cells", "matched", "contradicted", "unknown", "weight", "label", "key", "ink")

    def __init__(self, rid, anchor, rot, cells, matched, contradicted, unknown, weight, label, ink=0.0):
        self.rid, self.anchor, self.rot, self.cells = rid, anchor, rot, cells
        self.matched, self.contradicted, self.unknown, self.weight, self.label = matched, contradicted, unknown, weight, label
        self.ink = ink
        self.key = (rid, anchor, rot)


class DigitAgent:
    def __init__(self, alg, table, poses, cfg, log, rot_map=None):
        self.alg, self.table, self.poses, self.cfg, self.log = alg, table, poses, cfg, log
        self.tags = {t: alg.random() for t in TAGS}
        self.wm = WM(alg, poses, table, self.tags)
        self.EMPTY = table.EMPTY
        self.base = cfg["base"]                      # weight = base ** (graded evidence)
        self.rot_map = rot_map                       # prototype -> its 90-degree rotated prototype
        self.roles = {"conc": alg.random(), "n": alg.random(), "unobs": alg.random(), "empty": alg.random()}
        self.group_codes = np.stack([alg.random() for _ in range(64)])
        self.bucket_codes = np.stack([alg.random() for _ in range(8)])
        self.pol_W, self.pol_T, self.pol_arr = [], [], None
        self.meta = defaultdict(lambda: [0, 0])

    # ---------- sensing ----------
    def look(self, grid, cell, held, verified):
        preds = [(h, h.cells[cell]) for h in held if cell in h.cells]
        rid = int(grid[cell[0] * G + cell[1]])
        rid = self.EMPTY if rid < 0 else rid
        self.wm.add("observed", rid, cell)
        tau, B = self.cfg["tau"], self.alg.B
        for h, p in preds:                                   # a prediction made before looking: graded credit
            verified[h.key] += (self.table.O[p, rid] - tau) / (B - tau)
        return rid

    def evidence(self, items):
        E = -np.ones(NC, int)
        ghosts = 0
        for tag, rid, cell, m in items:
            k = cell[0] * G + cell[1]
            if tag == "observed":
                if not self.table.is_prim(rid):
                    ghosts += 1; continue
                E[k] = rid
            elif tag == "inferred":
                if self.table.is_prim(rid) or self.table.row(rid).status == "forgotten":
                    ghosts += 1; continue
                pl = self.table.placed()[rid - self.table.P - 1, k]
                E[pl >= 0] = pl[pl >= 0]
        return E, ghosts

    # ---------- hypotheses, vectorized over (row, anchor, rotation) ----------
    def placed_rot(self, k):
        pl = self.table.placed()
        if k == 0:
            return pl
        img = pl.reshape(len(pl), NC, G, G)
        rot = np.rot90(img, k, axes=(2, 3)).reshape(len(pl), NC, NC)
        rm = np.append(self.rot_map, self.EMPTY)
        for _ in range(k - 1):
            rm = np.append(self.rot_map, self.EMPTY)[rm]
        return np.where(rot >= 0, rm[np.where(rot >= 0, rot, 0)], -1).astype(np.int16)

    def hypotheses(self, E, rotations=(0,), top=40):
        """Graded evidence for every (row, anchor, rotation) at once: each observed cell
        contributes (overlap - tau)/(B - tau); blanks are explicit for exemplars.  Objects are
        built only for the strongest `top` and for complete chunk hypotheses."""
        T = self.table
        if not T.rows:
            return []
        obs = E >= 0
        is_chunk = np.array([r.label is None and r.status != "forgotten" for r in T.rows])
        active = np.array([r.status != "forgotten" for r in T.rows])
        tau, B = self.cfg["tau"], self.alg.B
        hyps = []
        for k in rotations:
            R = self.placed_rot(k)
            valid = R >= 0
            Ec = np.where(obs, E, 0)[None, None, :]
            O = T.O[np.where(valid, R, 0), Ec]
            use = valid & obs[None, None, :]
            contrib = np.where(use, (O - tau) / (B - tau), 0.0)
            eff = contrib.sum(-1)
            ink = np.where(R != self.EMPTY, contrib, 0.0).sum(-1)           # evidence from strokes alone
            m = (use & (O >= T.tau_match) & (R != self.EMPTY)).sum(-1)     # matched ink cells
            c = (use & (O < T.tau_match)).sum(-1)
            u = (valid & ~obs[None, None, :]).sum(-1)
            fits = valid.any(-1) & active[:, None]
            live = fits & (eff > 0)
            complete = fits & is_chunk[:, None] & (u == 0) & (c == 0) & (m >= 2)
            score = np.where(live, eff, -np.inf).reshape(-1)
            n_live = int(live.sum())
            pick = set()
            if n_live:
                pick = set(np.argpartition(-score, min(top, n_live) - 1)[:min(top, n_live)].tolist())
            pick |= set(np.flatnonzero(complete.reshape(-1)).tolist())
            effs, inks = eff.reshape(-1), ink.reshape(-1)
            for idx in pick:
                r_i, a = divmod(int(idx), NC)
                row = T.rows[r_i]
                cells = {CELLS[j]: int(R[r_i, a, j]) for j in range(NC) if R[r_i, a, j] >= 0}
                hyps.append(Hyp(row.id, CELLS[a], k, cells, int(m[r_i, a]), int(c[r_i, a]), int(u[r_i, a]),
                                float(self.base) ** float(effs[idx]), row.label, float(inks[idx])))
        hyps.sort(key=lambda h: (-h.weight, -len(h.cells), h.rid))
        best, out = {}, []                                    # one placement per exemplar row
        for h in hyps:
            if h.label is None:
                out.append(h)
            elif h.rid not in best:
                best[h.rid] = h; out.append(h)
        return out

    def hold(self, hyps, H):
        self.wm.remove_tag("assumed")
        held = hyps[: max(H - 1, 1)]
        T = self.table
        idea = next((h for h in hyps if T.row(h.rid).status == "provisional" and T.row(h.rid).label is None and h.matched >= 2), None)
        if idea is None:
            idea = next((h for h in hyps if T.row(h.rid).status == "provisional"), None)
        if idea is not None and idea not in held:
            held.append(idea)
        elif len(hyps) > len(held):
            held.append(hyps[len(held)])
        for h in held:
            self.wm.add("assumed", h.rid, h.anchor, max(h.matched, 1))
        return held

    def maximal(self, held):
        out = []
        for h in held:
            hs = set(h.cells.items())
            if not any(g is not h and hs <= set(g.cells.items()) for g in held):
                out.append(h)
        return out

    def ring(self, h):
        return {(r + dr, c + dc) for (r, c) in h.cells for dr, dc in NEIGH4
                if (r + dr, c + dc) not in h.cells and 0 <= r + dr < G and 0 <= c + dc < G}

    def posterior(self, hyps):
        """Label mass over the strongest hypotheses (the superposition of their label ⊗ class)."""
        post = defaultdict(float)
        for h in hyps:
            if h.label is not None:
                post[h.label] += h.weight
        return post

    # ---------- recognition of chunks ----------
    def compact(self, hyps, E, ep, stats):
        T = self.table
        complete = [h for h in hyps if h.label is None and h.unknown == 0 and h.contradicted == 0 and h.rot == 0 and h.matched >= 2]
        complete.sort(key=lambda h: (T.row(h.rid).status != "consolidated", -len(h.cells), h.rid))
        for h in complete:
            row = T.row(h.rid)
            hs = set(h.cells.items())
            if row.status != "consolidated" and any(g is not h and g.label is None and hs < set(g.cells.items()) for g in hyps):
                continue
            if ("inferred", h.rid, h.anchor) in self.wm.shadow:
                continue
            if any(("observed", int(E[c[0] * G + c[1]]), c) not in self.wm.shadow for c in h.cells):
                continue
            for cell in h.cells:
                self.wm.remove("observed", int(E[cell[0] * G + cell[1]]), cell)
            self.wm.add("inferred", h.rid, h.anchor)
            row.uses += 1; row.last_used = ep
            stats["compactions"] += 1
            return True
        return False

    # ---------- deciding ----------
    def leader(self, hyps, post):
        lab = max(post, key=post.get)
        return lab, post[lab] / sum(post.values()), next(h for h in hyps if h.label == lab)

    def situation_key(self, hyps, E, glimpses, verified):
        post = self.posterior(hyps)
        if not post:
            return None
        lab, conc, h = self.leader(hyps, post)
        n_ex = len({g.label for g in hyps[:8] if g.label is not None})
        return (lab, 0 if conc >= 0.99 else 1 if conc >= 0.95 else 2, min(glimpses, 12) // 3,
                min(int(max(verified[h.key], 0)), 2), min(h.contradicted, 2), min(n_ex, 3))

    def parts_hit(self, h, E):
        row = self.table.row(h.rid)
        if len(row.parts) < 2:
            return 2
        return sum(1 for pid, (dr, dc) in row.parts
                   if any(0 <= h.anchor[0] + dr + r < G and 0 <= h.anchor[1] + dc + c < G
                          and E[(h.anchor[0] + dr + r) * G + h.anchor[1] + dc + c] >= 0
                          and self.table.sim[p, E[(h.anchor[0] + dr + r) * G + h.anchor[1] + dc + c]]
                          for (r, c), p in self.table.footprint(pid).items()))

    def terminate(self, hyps, E, verified, glimpses, use_meta):
        post = self.posterior(hyps)
        self.why = "no hypotheses"
        if not post:
            return None
        lab, conc, h = self.leader(hyps, post)
        self.why = "concentration" if conc < self.cfg["concentration"] else "ink evidence" if h.ink < self.cfg["min_ink"] else None
        if self.why:
            return None
        if self.parts_hit(h, E) < 2:
            self.why = "parts"; return None
        if verified[h.key] < self.cfg["verify_min"] and h.unknown > 0:
            self.why = "verification"; return None
        if not (E == self.EMPTY).any():
            self.why = "no blank seen"; return None
        self.why = "declared"
        if use_meta:
            key = self.situation_key(hyps, E, glimpses, verified)
            right, wrong = self.meta.get(key, (0, 0))
            if right + wrong >= self.cfg["meta_min"] and wrong / (right + wrong) > self.cfg["meta_veto"]:
                return "veto"
        return ("declare", h)

    def situation_bundle(self, hyps, E, glimpses):
        """The situation as an SDR: cell ⊗ (coarse stroke group | empty | unobserved) + buckets."""
        alg, W = self.alg, self.alg.empty()
        coarse = self.table.C.coarse
        for k in range(NC):
            v = self.roles["unobs"] if E[k] < 0 else self.roles["empty"] if E[k] == self.EMPTY else self.group_codes[coarse[E[k]]]
            alg.add(W, alg.bind(self.poses.cell_codes[k], v))
        post = self.posterior(hyps)
        conc = max(post.values()) / sum(post.values()) if post else 0.0
        alg.add(W, alg.bind(self.roles["conc"], self.bucket_codes[min(int(conc * 4), 3)]))
        alg.add(W, alg.bind(self.roles["n"], self.bucket_codes[4 + min(glimpses // 4, 3)]))
        return W

    def choose(self, hyps, held, E, last, glimpses, policy):
        unobs = [CELLS[k] for k in range(NC) if E[k] < 0]
        if not unobs:
            return None, None
        if policy == "raster":
            return unobs[0], "raster"
        if policy == "random":
            r = np.random.default_rng(glimpses * 7919 + int(E[E >= 0].sum()))
            return unobs[r.integers(len(unobs))], "random"
        if policy == "learned":
            if self.pol_arr is None:
                return unobs[0], "raster"
            Wb = (self.situation_bundle(hyps, E, glimpses) > 0).reshape(-1).astype(np.float32)
            sc = self.pol_arr @ Wb
            votes = defaultdict(float)
            for i in np.argsort(-sc)[: self.cfg["policy_k"]]:
                if self.pol_T[i] in unobs:
                    votes[self.pol_T[i]] += 1
            if votes:
                return max(votes, key=lambda c: (votes[c], -dist(c, last))), "learned"
            return min(unobs, key=lambda c: dist(c, last)), "nearest"
        return self.computed(held, E, last)

    def computed(self, held, E, last):
        """VI.5: look where the held hypotheses disagree; else test the leader where it is exposed."""
        nonempty = {CELLS[k] for k in range(NC) if E[k] >= 0 and E[k] != self.EMPTY}
        unobs = lambda c: E[c[0] * G + c[1]] < 0
        nearest = lambda cs: min(cs, key=lambda c: (dist(c, last), c))
        maximal = self.maximal(held)
        sim = self.table.sim
        if len(maximal) >= 2:
            cands = set()
            for h in maximal:
                cands |= set(h.cells) | self.ring(h)
            total = sum(h.weight for h in maximal)
            best, bestd = None, 0.0
            for c in sorted(cands):
                if not unobs(c):
                    continue
                preds = [(h.cells.get(c, self.EMPTY), h.weight) for h in maximal]
                agree = max(sum(w2 for p2, w2 in preds if sim[p, p2]) for p, _ in preds)
                d = 1 - agree / total
                if d > bestd + 1e-9 or (best is not None and abs(d - bestd) < 1e-9 and dist(c, last) < dist(best, last)):
                    best, bestd = c, d
            if best is not None and bestd >= 0.05:
                return best, "split"
        if maximal:
            h = maximal[0]
            unknown = [c for c in h.cells if unobs(c) and h.cells[c] != self.EMPTY]
            if unknown:
                row, unhit = self.table.row(h.rid), set()
                for pid, (dr, dc) in row.parts:
                    cells_p = [(h.anchor[0] + dr + r, h.anchor[1] + dc + c) for (r, c), p in self.table.footprint(pid).items()]
                    if not any(0 <= i < G and 0 <= j < G and E[i * G + j] >= 0 and sim[h.cells[(i, j)], E[i * G + j]] for (i, j) in cells_p):
                        unhit |= set(cells_p)
                pool = [c for c in unknown if c in unhit] or unknown
                return max(pool, key=lambda c: (min([dist(c, o) for o in nonempty] or [0]), c)), "verify"
            blanks = [c for c in h.cells if unobs(c)]
            if blanks and not (E == self.EMPTY).any():
                return nearest(blanks), "boundary"
        frontier = [(r + dr, c + dc) for (r, c) in nonempty for dr, dc in NEIGH4
                    if 0 <= r + dr < G and 0 <= c + dc < G and unobs((r + dr, c + dc))]
        if frontier:
            return nearest(frontier), "explore"
        return nearest([CELLS[k] for k in range(NC) if E[k] < 0]), "raster"

    # ---------- the episode ----------
    def run_episode(self, grid, label, ep, mode="train", policy="computed", use_meta=False, rotations=(0,), verbose=False):
        T, wm, cfg = self.table, self.wm, self.cfg
        wm.clear()
        verified = defaultdict(float)
        stats = dict(glimpses=0, study=0, compactions=0, ghosts=0, vetoes=0, overwhelmed=0, peak=0, ideas=0, clicks=0)
        cell = (1, 1)
        self.look(grid, cell, [], verified)
        glimpses, last, held, hyps, decision, H = 1, cell, [], [], None, cfg["H"]
        E = -np.ones(NC, int)
        while True:
            items, ghosts = wm.read()
            stats["ghosts"] += ghosts
            E, _ = self.evidence(items)
            hyps = self.hypotheses(E, rotations)
            if self.compact(hyps, E, ep, stats):
                continue
            held = self.hold(hyps, H)
            if wm.pressure() > cfg["eps"]:
                while wm.pressure() > cfg["eps"] and H > 2:
                    H -= 1; held = self.hold(hyps, H)
                if wm.pressure() > cfg["eps"]:
                    stats["overwhelmed"] += 1; break
            decision = self.terminate(hyps, E, verified, glimpses, use_meta)
            if decision == "veto":
                stats["vetoes"] += 1; decision = None
                if stats["vetoes"] > cfg["meta_max"]:
                    decision = self.terminate(hyps, E, verified, glimpses, False)
                    if decision:
                        break
            elif decision:
                break
            if glimpses >= cfg["budget"]:
                break
            if mode == "train" and policy == "computed":
                cell, why = self.computed(held, E, last)
                if cell is not None and len(self.pol_W) < cfg["policy_store"]:
                    self.pol_W.append(self.situation_bundle(hyps, E, glimpses)); self.pol_T.append(cell)
            else:
                cell, why = self.choose(hyps, held, E, last, glimpses, policy)
            if cell is None:
                break
            rid = self.look(grid, cell, held, verified)
            glimpses += 1; last = cell
            if verbose:
                post = self.posterior(hyps)
                tot = sum(post.values()) or 1
                ps = " ".join(f"{l}:{v/tot:.2f}" for l, v in sorted(post.items(), key=lambda x: -x[1])[:3])
                self.log(f"    {why:8s} {cell} -> {T.label_str(rid):4s}  posterior {ps}   held {' '.join(T.label_str(h.rid) for h in held[:3])}")
        stats["glimpses"] = glimpses; stats["peak"] = wm.peak; stats["why"] = getattr(self, "why", "?")
        post = self.posterior(hyps)
        if decision:
            h = decision[1]; label_hat, confident = h.label, True
        else:
            confident = False
            label_hat = max(post, key=post.get) if post else None
            h = next((g for g in hyps if g.label == label_hat), None)
        correct = label_hat == label
        stats.update(label_hat=label_hat, confident=confident, correct=correct, wrong_confident=confident and not correct)
        if mode == "train":
            key = self.situation_key(hyps, E, glimpses, verified) if h is not None else None
            if key is not None:
                self.meta[key][0 if correct else 1] += 1
            for (rid, anchor, rot), v in verified.items():
                if v >= 1.0 and not T.is_prim(rid) and T.row(rid).status != "forgotten":
                    T.row(rid).worked += 1
            if confident and correct:
                row = T.row(h.rid); row.top_uses += 1; row.last_used = ep
            else:
                for k in range(NC):                                   # study the whole digit
                    if E[k] < 0:
                        self.look(grid, CELLS[k], [], verified); stats["study"] += 1
                items, _ = wm.read(); E, _ = self.evidence(items)
                fp = {CELLS[k]: int(E[k]) for k in range(NC) if E[k] >= 0 and E[k] != self.EMPTY}
                if len(fp) >= 2:
                    row, created = T.propose(normalize(fp), ep, label=label)
                    row.top_uses += 1
            stats["ideas"] = self.ideas(E, ep)
            stats["clicks"] = len(T.clicks(ep, prune=(ep % 50 == 0)))
        return stats

    def ideas(self, E, ep, top=6):
        """Shared stroke groups between this digit and remembered rows -> provisional chunks."""
        T = self.table
        if not T.rows:
            return 0
        hyps = [h for h in self.hypotheses(E) if h.label is not None][:top]
        n = 0
        for h in hyps:
            S = {c: p for c, p in h.cells.items() if p != self.EMPTY and E[c[0] * G + c[1]] >= 0 and T.sim[p, E[c[0] * G + c[1]]]}
            n_ink = sum(1 for p in h.cells.values() if p != self.EMPTY)
            if not (3 <= len(S) < n_ink and len(S) <= 6):
                continue
            fp = normalize(S)
            if fkey(fp) in T.by_key or T.refs_fp(fp) < 2:
                continue
            T.propose(fp, ep, label=None, prefix="i"); n += 1
        return n

    def freeze_policy(self):
        if self.pol_W:
            self.pol_arr = np.stack([(W > 0).reshape(-1) for W in self.pol_W]).astype(np.float32)
