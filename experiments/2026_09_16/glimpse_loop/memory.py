"""The table  (DESIGN.md IV.4, V, VIII.4, IX.2, X.2).

One table, two kinds of row, same machinery:
    primitives  - the sensory alphabet, given
    chunks      - (composite -> name): parts = [(row_id, (dr, dc)), ...], recursive

Every non-primitive row is born *provisional* (IX.2: assumed, cheap to write, forgettable).
It is consolidated - the "click" of VIII.4 - when its description-length gain is positive:

    gain(r) = top_uses(r) * (k-1) * bits_item        # episodes it encodes on its own
            + refs(r)     * (k-1) * bits_part        # rows in which it stands for k parts
            - (bits_name  +  k   * bits_part)        # what the row itself costs

where k = len(r.parts).  Nothing is earned for being recognised *inside* a parent that is
itself a row: the parent's row already pays for that.  So a fragment that only ever occurs
inside one chunk can never pay for itself, and a chunk that stops paying (after a refactor
made its parents cheaper another way) is inlined back - the same criterion, both directions.
No tuned threshold (X.3): a row is kept exactly while it shortens the total description.

A second, independent condition (VIII.5, the counterweight): the chunk must have *worked* -
been held as a hypothesis and predicted an unseen cell correctly in at least one episode.
An idea is a compaction the agent tries; the click is the verdict after the test.
"""
import math
import numpy as np


class Row:
    __slots__ = ("id", "name", "label", "parts", "status", "born", "uses",
                 "top_uses", "worked", "last_used", "consolidated_at")

    def __init__(self, id, name, label, parts, status, born):
        self.id, self.name, self.label, self.parts = id, name, label, parts
        self.status, self.born = status, born
        self.uses, self.top_uses, self.worked = 0, 0, 0        # worked: episodes where it predicted right
        self.last_used, self.consolidated_at = born, None

    @property
    def is_prim(self):
        return not self.parts and self.status == "given"


def normalize(cells):
    """cells: iterable of (prim_id, (r, c)) -> frozenset with the lexicographically smallest
    cell moved to (0, 0)."""
    cells = list(cells)
    r0, c0 = min(rc for _, rc in cells)
    return frozenset((p, (r - r0, c - c0)) for p, (r, c) in cells)


class Table:
    def __init__(self, alg, n_cells, n_rel, prov_max=60, click_margin=0.0, require_test=True,
                 prune_margin=10.0, log=None):
        self.alg = alg
        self.require_test = require_test
        self.prune_margin = prune_margin        # hysteresis: unlearn only when clearly not paying
        self.rows = []
        self.n_cells, self.n_rel = n_cells, n_rel
        self.prov_max, self.click_margin = prov_max, click_margin
        self.log = log or (lambda s: None)
        self._names = None
        self._fp = {}
        self._index = None
        self.events = []            # (episode, kind, text)
        self.EMPTY = None

    # ---------------- rows ----------------
    def add_primitive(self, label, code=None):
        row = Row(len(self.rows), self.alg.random() if code is None else code, label, [], "given", 0)
        self.rows.append(row)
        self._dirty()
        return row

    def _dirty(self):
        self._names = None
        self._index = None

    @property
    def names(self):
        if self._names is None:
            self._names = np.stack([r.name for r in self.rows])
        return self._names

    def __len__(self):
        return len(self.rows)

    def prims(self):
        return [r for r in self.rows if r.is_prim]

    def chunks(self, status=None):
        return [r for r in self.rows if r.parts and r.status != "forgotten"
                and (status is None or r.status == status)]

    def footprint(self, rid):
        """Fully expanded {(prim_id, (dr, dc))}, relative to the row's origin."""
        if rid in self._fp:
            return self._fp[rid]
        row = self.rows[rid]
        if row.is_prim:
            fp = frozenset([(rid, (0, 0))])
        else:
            cells = set()
            for pid, (dr, dc) in row.parts:
                for p, (r, c) in self.footprint(pid):
                    cells.add((p, (r + dr, c + dc)))
            fp = frozenset(cells)
        self._fp[rid] = fp
        return fp

    def find(self, fp):
        for r in self.chunks():
            if self.footprint(r.id) == fp:
                return r
        return None

    def index(self):
        """Inverted index  prim_id -> [(row_id, (dr, dc))]  over expanded footprints."""
        if self._index is None:
            idx = {}
            for r in self.chunks():
                for p, off in self.footprint(r.id):
                    idx.setdefault(p, []).append((r.id, off))
            self._index = idx
        return self._index

    def touch(self, r, ep):
        r.last_used = ep

    # ---------------- description length ----------------
    def bits_item(self):  return math.log2(len(self.rows)) + math.log2(self.n_cells)
    def bits_part(self):  return math.log2(len(self.rows)) + math.log2(self.n_rel)
    def bits_name(self):  return math.log2(len(self.rows))
    def row_cost(self, r): return self.bits_name() + len(r.parts) * self.bits_part()
    def table_bits(self):  return sum(self.row_cost(r) for r in self.chunks("consolidated"))

    def occurrences(self, q, fp):
        """Non-overlapping placements of footprint `fp` inside row q as a union of whole
        direct parts of q  (i.e. q could be re-written to use it)."""
        cellmap, partcells = {}, {}
        for i, (pid, (dr, dc)) in enumerate(q.parts):
            cs = [(p, (r + dr, c + dc)) for p, (r, c) in self.footprint(pid)]
            partcells[i] = {rc for _, rc in cs}
            for p, rc in cs:
                cellmap[rc] = (p, i)
        origin_p = next(p for p, rc in fp if rc == (0, 0))
        used, n = set(), 0
        for (r, c), (p, i) in sorted(cellmap.items()):
            if p != origin_p:
                continue
            cells = [((r + dr, c + dc), pp) for pp, (dr, dc) in fp]
            if any(rc in used or cellmap.get(rc, (None,))[0] != pp for rc, pp in cells):
                continue
            covered = {rc for rc, _ in cells}
            parts_hit = {cellmap[rc][1] for rc in covered}
            if all(partcells[i] <= covered for i in parts_hit):
                used |= covered
                n += 1
        return n

    def refs_fp(self, fp, exclude=-1):
        """Independent things this footprint stands inside: consolidated rows, and rows that
        were seen as whole scenes.  Idea rows are derived from those and do not count."""
        return sum(self.occurrences(q, fp) for q in self.chunks()
                   if q.id != exclude and len(self.footprint(q.id)) > len(fp)
                   and (q.status == "consolidated" or q.top_uses >= 1))

    def refs(self, r):
        return self.refs_fp(self.footprint(r.id), exclude=r.id)

    def gain(self, r):
        k = len(r.parts)
        return (r.top_uses * (k - 1) * self.bits_item() + self.refs(r) * (k - 1) * self.bits_part()
                - self.row_cost(r))

    # ---------------- allocation (X.2: gated, provisional first) ----------------
    def factorize(self, fp, exclude=()):
        """Express a footprint as parts, reusing consolidated chunks greedily (largest first)."""
        remaining = dict(((r, c), p) for p, (r, c) in fp)
        parts = []
        cands = [r for r in self.chunks("consolidated")
                 if r.id not in exclude and len(self.footprint(r.id)) < len(fp)]
        cands.sort(key=lambda r: (-len(self.footprint(r.id)), r.id))
        for cand in cands:
            cfp = self.footprint(cand.id)
            placed = True
            while placed:
                placed = False
                for (r, c), p in sorted(remaining.items()):
                    cells = [((r + dr, c + dc), pp) for pp, (dr, dc) in cfp]
                    if all(remaining.get(rc) == pp for rc, pp in cells):
                        for rc, _ in cells:
                            del remaining[rc]
                        parts.append((cand.id, (r, c)))
                        placed = True
                        break
        for (r, c), p in remaining.items():
            parts.append((p, (r, c)))
        parts.sort(key=lambda x: x[1])
        return parts

    def propose(self, fp, ep, label_prefix="p"):
        """Return (row, created). A duplicate footprint returns the existing row."""
        existing = self.find(fp)
        if existing is not None:
            self.touch(existing, ep)
            return existing, False
        row = Row(len(self.rows), self.alg.random(), f"{label_prefix}{len(self.rows)}",
                  self.factorize(fp), "provisional", ep)
        self.rows.append(row)
        self._dirty()
        self.evict(ep)
        return row, True

    def referenced(self, rid):
        return any(rid == pid for r in self.chunks() for pid, _ in r.parts)

    def evict(self, ep):
        prov = [r for r in self.chunks("provisional") if not self.referenced(r.id)]
        n_prov = len(self.chunks("provisional"))
        while n_prov > self.prov_max and prov:
            victim = min(prov, key=lambda r: (r.top_uses > 0, r.last_used, r.born))   # ideas go first
            self.log(f"        forget {victim.label} (last seen ep {victim.last_used})")
            self.forget(victim)
            prov.remove(victim); n_prov -= 1

    def forget(self, r):
        r.status = "forgotten"
        r.parts = []
        self._fp.pop(r.id, None)
        self._dirty()

    # ---------------- the click (VIII.4) and its mirror ----------------
    def clicks(self, ep):
        """Consolidate provisional rows whose gain is positive, largest gain first, re-evaluating
        after every refactor; then inline consolidated rows that no longer pay."""
        done = []
        while True:
            cands = [(self.gain(r), r) for r in self.chunks("provisional")
                     if r.worked >= 1 or not self.require_test]
            cands = [(g, r) for g, r in cands if g > self.click_margin]
            if not cands:
                break
            g, r = max(cands, key=lambda x: (x[0], -x[1].id))
            self.consolidate(r, ep, g)
            done.append(r)
        done += self.prune(ep)
        return done

    def consolidate(self, r, ep, g):
        r.status = "consolidated"
        r.consolidated_at = ep
        r.label = f"C{r.id}"
        self.events.append((ep, "click", f"{r.label} gain {g:+.1f} bits (whole scene {r.top_uses}x, "
                                         f"inside {self.refs(r)} rows, predicted right in {r.worked} ep): {self.describe(r)}"))
        self.log(f"  ** CLICK  {r.label} consolidated (gain {g:+.1f} bits: whole scene {r.top_uses}x, "
                 f"part of {self.refs(r)} rows, predicted right in {r.worked} episodes) = {self.describe(r)}")
        self.refactor_all(ep)

    def prune(self, ep):
        out = []
        while True:
            bad = [(self.gain(r), r) for r in self.chunks("consolidated")]
            bad = [(g, r) for g, r in bad if g < -self.prune_margin]
            if not bad:
                return out
            g, r = min(bad, key=lambda x: (x[0], x[1].id))
            for q in self.chunks():
                if any(pid == r.id for pid, _ in q.parts):
                    q.parts = self._inline(q.parts, r)
                    self._fp.pop(q.id, None)
            self.events.append((ep, "unlearn", f"{r.label} gain {g:+.1f} bits: {self.describe(r)}"))
            self.log(f"  ** UNLEARN {r.label} (gain {g:+.1f} bits) = {self.describe(r)}")
            self.forget(r)
            out.append(r)

    def _inline(self, parts, child):
        out = []
        for pid, (dr, dc) in parts:
            if pid == child.id:
                for p2, (r2, c2) in child.parts:
                    out.append((p2, (r2 + dr, c2 + dc)))
            else:
                out.append((pid, (dr, dc)))
        return sorted(out, key=lambda x: x[1])

    def refactor_all(self, ep):
        """Re-express every row using the current consolidated vocabulary (library refactoring)."""
        changed = []
        for r in self.chunks():
            fp = self.footprint(r.id)
            before = len(r.parts)
            parts = self.factorize(fp, exclude={r.id} | self.ancestors(r.id))
            if len(parts) < before:
                r.parts = parts
                changed.append(f"{r.label} {before}->{len(parts)} parts")
        self._dirty()
        if changed:
            self.log(f"     refactored: {', '.join(changed)}")

    def ancestors(self, rid):
        out = set()
        for q in self.chunks():
            if any(pid == rid for pid, _ in q.parts):
                out.add(q.id)
                out |= self.ancestors(q.id)
        return out

    # ---------------- pretty ----------------
    def describe(self, r):
        return " + ".join(f"{self.rows[pid].label}@({dr},{dc})" for pid, (dr, dc) in r.parts)

    def sketch(self, rid):
        fp = self.footprint(rid)
        rs = [r for _, (r, _) in fp]; cs = [c for _, (_, c) in fp]
        h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        grid = [["." for _ in range(w)] for _ in range(h)]
        for p, (r, c) in fp:
            grid[r - min(rs)][c - min(cs)] = self.rows[p].label
        return ["".join(row) for row in grid]
