"""The perception loop  (DESIGN.md Parts VI, VIII, IX, X).

Working memory is ONE bundle of   tag ⊗ name ⊗ pose   items (VI.3, IX.2), with three
provenance tags:  observed / inferred / assumed.  Everything the agent knows about the
current scene it reads back out of that bundle through unbind + presence, ghosts included.

Per glimpse:
    read WM  ->  evidence  ->  vote for (row, anchor) hypotheses (capsule-style agreement)
    recognize: a complete hypothesis collapses its parts into one `inferred` item (VIII.2)
    hold the top hypotheses in WM as `assumed` items, weight = evidence count (X.4),
        always keeping one *idea* (a provisional row) in mind so that ideas get tested
    under pressure (ghost estimate > eps): hold fewer hypotheses, then try a new compaction
        of what is seen (an idea, tested by later episodes), else stop: overwhelmed (VIII.1)
    terminate (X.6) or choose where to look (VI.5: where held hypotheses disagree)

Per episode end:
    ideas  - part of what I see recurs in something I remember -> its own row (IV.4)
    clicks - rows that pay for themselves AND have predicted correctly are consolidated (VIII.4)
    remember the whole scene as a provisional row, expressed with existing chunks
"""
from collections import defaultdict
import numpy as np
from memory import normalize
from world import NEIGH4

TAGS = ("observed", "inferred", "assumed")


def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class WM:
    """One bundle.  Reads go through the algebra.  The write log (`shadow`) is used only to
    subtract exactly what was written and to count ghosts for the report."""

    def __init__(self, alg, poses, table, tags):
        self.alg, self.poses, self.table, self.tags = alg, poses, table, tags
        self.W = alg.empty()
        self.shadow = {}
        self.order = []
        self.peak = 0

    def clear(self):
        self.W[:] = 0
        self.shadow.clear()
        self.order.clear()
        self.peak = 0

    def code(self, tag, rid, cell):
        a = self.alg
        return a.bind(a.bind(self.tags[tag], self.table.rows[rid].name), self.poses(cell))

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
            self.alg.sub(self.W, self.code(*key), m)
            self.order.remove(key)
        return m

    def remove_tag(self, tag):
        for key in [k for k in self.order if k[0] == tag]:
            self.remove(*key)

    def has(self, tag, rid, cell):
        U = self.alg.unbind(self.W, self.alg.bind(self.tags[tag], self.poses(cell)))
        return int(self.alg.presence(U, self.table.rows[rid].name[None, :])[0])

    def read(self):
        """Enumerate WM by probing every (tag, cell) against every name."""
        items = []
        names = self.table.names
        for tag in TAGS:
            keys = self.alg.bind(self.tags[tag][None, :], self.poses.cell_codes)
            pres = self.alg.presence(self.alg.unbind_many(self.W, keys), names)
            for ci, ri in zip(*np.nonzero(pres > 0)):
                items.append((tag, int(ri), self.poses.cells[ci], int(pres[ci, ri])))
        ghosts = sum(1 for tag, rid, cell, _ in items if (tag, rid, cell) not in self.shadow)
        return items, ghosts

    def n_probes(self):
        return len(TAGS) * len(self.poses.cells) * len(self.table)

    def pressure(self):
        return self.alg.ghost_estimate(self.W, self.n_probes())

    def load(self):
        return len(self.shadow)


class Hyp:
    __slots__ = ("rid", "anchor", "cells", "matched", "unknown", "parts_hit", "key")

    def __init__(self, rid, anchor, cells, matched, parts_hit):
        self.rid, self.anchor, self.cells, self.matched = rid, anchor, cells, matched
        self.unknown = len(cells) - matched
        self.parts_hit = parts_hit
        self.key = (rid, anchor)


class Agent:
    def __init__(self, alg, table, poses, cfg, log):
        self.alg, self.table, self.poses, self.cfg, self.log = alg, table, poses, cfg, log
        self.tags = {t: alg.random() for t in TAGS}
        self.wm = WM(alg, poses, table, self.tags)
        self.EMPTY = table.EMPTY.id
        self.A = len(table.prims())           # alphabet size: a matched cell is ~log2(A) bits of evidence

    def weight(self, h):
        return float(self.A) ** h.matched

    # ------------------------------------------------------------------ sensing
    def sense(self, code):
        """Encode by looking up: sensor code -> primitive row, cleanup over the alphabet."""
        prims = self.table.prims()
        i, _ = self.alg.cleanup(code, np.stack([r.name for r in prims]))
        return prims[i].id

    def look(self, world, cell, held, verified):
        preds_in = [(h, h.cells[cell]) for h in held if cell in h.cells]
        preds_out = [h for h in held if cell in self.ring(h)]
        rid = self.sense(world.look(cell))
        self.wm.add("observed", rid, cell)
        for h, p in preds_in:
            if p == rid:
                verified[h.key][0].add(cell)
        if rid == self.EMPTY:
            for h in preds_out:
                verified[h.key][1].add(cell)
        return rid

    # ------------------------------------------------------------------ reading WM
    def evidence(self, items):
        """cell -> prim/EMPTY according to WM; inferred chunks are expanded.  Items that are
        category errors (an observed chunk, an inferred primitive) can only be ghosts."""
        ev, ghosts = {}, 0
        for tag, rid, cell, m in items:
            row = self.table.rows[rid]
            if tag == "observed":
                if not row.is_prim:
                    ghosts += 1
                    continue
                if cell in ev and ev[cell] != rid:
                    ghosts += 1
                ev[cell] = rid
            elif tag == "inferred":
                if row.is_prim or row.status == "forgotten":
                    ghosts += 1
                    continue
                for p, (dr, dc) in self.table.footprint(rid):
                    ev[(cell[0] + dr, cell[1] + dc)] = p
        return ev, ghosts

    def hypotheses(self, ev):
        """Every evidence cell votes for (row, anchor); keep the uncontradicted ones."""
        idx = self.table.index()
        cands = set()
        for cell, p in ev.items():
            if p == self.EMPTY:
                continue
            for rid, (dr, dc) in idx.get(p, []):
                cands.add((rid, (cell[0] - dr, cell[1] - dc)))
        hyps = []
        for rid, anchor in cands:
            row = self.table.rows[rid]
            if row.status == "forgotten":
                continue
            cells, ok = {}, True
            for p, (dr, dc) in self.table.footprint(rid):
                c = (anchor[0] + dr, anchor[1] + dc)
                if not self.poses.on_canvas(c):
                    ok = False
                    break
                cells[c] = p
            if not ok:
                continue
            matched = sum(1 for c, p in cells.items() if ev.get(c) == p)
            contradicted = sum(1 for c, p in cells.items() if c in ev and ev[c] != p)
            if contradicted or matched == 0:
                continue
            parts_hit = sum(1 for pid, (dr, dc) in row.parts
                            if any(ev.get((anchor[0] + dr + r, anchor[1] + dc + c)) == p
                                   for p, (r, c) in self.table.footprint(pid)))
            hyps.append(Hyp(rid, anchor, cells, matched, parts_hit))
        hyps.sort(key=lambda h: (-h.matched, -len(h.cells), h.rid))
        return hyps

    def hold(self, hyps, H):
        """Top hypotheses live in WM as `assumed` items, multiplicity = evidence (X.4).
        One slot is reserved for the best idea (a provisional row) so that ideas get tested."""
        self.wm.remove_tag("assumed")
        held = hyps[: max(H - 1, 1)]
        idea = next((h for h in hyps if self.table.rows[h.rid].status == "provisional"), None)
        if idea is not None and idea not in held:
            held.append(idea)
        elif len(hyps) > len(held):
            held.append(hyps[len(held)])
        for h in held:
            self.wm.add("assumed", h.rid, h.anchor, h.matched)
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
                if (r + dr, c + dc) not in h.cells and self.poses.on_canvas((r + dr, c + dc))}

    def neighbors(self, cell):
        return [(cell[0] + dr, cell[1] + dc) for dr, dc in NEIGH4
                if self.poses.on_canvas((cell[0] + dr, cell[1] + dc))]

    # ------------------------------------------------------------------ recognition
    def parts_present(self, parts, anchor):
        for pid, (dr, dc) in parts:
            c = (anchor[0] + dr, anchor[1] + dc)
            if self.table.rows[pid].is_prim:
                if not self.wm.has("observed", pid, c):
                    return False
            elif not self.wm.has("inferred", pid, c) and \
                    not self.parts_present(self.table.rows[pid].parts, c):
                return False
        return True

    def collapse(self, rid, anchor, ep):
        """Replace the parts by the whole.  Exact subtraction, one inferred item (VIII.2)."""
        row = self.table.rows[rid]
        for pid, (dr, dc) in row.parts:
            c = (anchor[0] + dr, anchor[1] + dc)
            if self.table.rows[pid].is_prim:
                self.wm.remove("observed", pid, c)
            else:
                if not self.wm.has("inferred", pid, c):
                    self.collapse(pid, c, ep)
                self.wm.remove("inferred", pid, c)
        self.wm.add("inferred", rid, anchor)
        row.uses += 1
        self.table.touch(row, ep)

    def compact(self, hyps, ep, stats):
        """Recognize one complete hypothesis: consolidated rows first, larger first; a
        provisional row only when nothing bigger that is live explains its cells."""
        complete = [h for h in hyps if h.unknown == 0]
        complete.sort(key=lambda h: (self.table.rows[h.rid].status != "consolidated", -len(h.cells), h.rid))
        for h in complete:
            row = self.table.rows[h.rid]
            if row.status != "consolidated":
                hs = set(h.cells.items())
                if any(g is not h and hs < set(g.cells.items()) for g in hyps):
                    continue
            if self.wm.has("inferred", h.rid, h.anchor) or not self.parts_present(row.parts, h.anchor):
                continue
            self.collapse(h.rid, h.anchor, ep)
            stats["compactions"] += 1
            self.log(f"     recognize {row.label} @{h.anchor}: {len(row.parts)} items -> 1   "
                     f"(load {self.wm.load()}, pressure {self.wm.pressure():.2f})")
            return True
        return False

    def try_compaction(self, ep, stats):
        """Nothing stored explains the scene and WM is full: group what is seen into a new
        provisional chunk and collapse it.  An idea; later episodes test it (X.2, VIII.2)."""
        items, _ = self.wm.read()
        top = [(rid, cell) for tag, rid, cell, m in items if rid != self.EMPTY
               and ((tag == "observed" and self.table.rows[rid].is_prim)
                    or (tag == "inferred" and not self.table.rows[rid].is_prim))]
        if len(top) < 2:
            return False
        expanded = [(p, (cell[0] + dr, cell[1] + dc)) for rid, cell in top
                    for p, (dr, dc) in self.table.footprint(rid)]
        fp = normalize(expanded)
        anchor = min(rc for _, rc in expanded)
        existing = self.table.find(fp)
        parts = existing.parts if existing is not None else self.table.factorize(fp)
        if not self.parts_present(parts, anchor):
            return False
        row, created = self.table.propose(fp, ep, label_prefix="t")
        before = self.wm.load()
        self.collapse(row.id, anchor, ep)
        stats["compactions"] += 1
        stats["tries"] += 1
        self.log(f"     pressure: try compaction {row.label} = {self.table.describe(row)}"
                 f"   (load {before} -> {self.wm.load()}, pressure {self.wm.pressure():.2f})")
        return True

    # ------------------------------------------------------------------ deciding
    def terminate(self, held, ev, verified):
        nonempty = {c for c, p in ev.items() if p != self.EMPTY}
        maximal = self.maximal(held)
        if not maximal:
            return None
        h = maximal[0]
        share = self.weight(h) / sum(self.weight(g) for g in maximal)
        if share < self.cfg["concentration"]:
            return None
        if not all(c in h.cells for c in nonempty) or h.matched < 2:
            return None
        row = self.table.rows[h.rid]
        if len(row.parts) >= 2 and h.parts_hit < 2:
            return None                      # evidence inside one shared part proves nothing
        ring_empty = any(ev.get(c) == self.EMPTY for c in self.ring(h))
        verified_in = len(verified[h.key][0]) >= self.cfg["verify_min"] or h.unknown == 0
        if row.top_uses == 0:
            # never seen this as a whole scene: only certain once the frontier is exhausted
            frontier = [c for cell in nonempty for c in self.neighbors(cell) if c not in ev]
            if frontier or h.unknown:
                return None
            return ("declare", h)
        if ring_empty and verified_in:
            return ("declare", h)
        return None

    def policy(self, held, ev, last):
        nonempty = {c for c, p in ev.items() if p != self.EMPTY}
        unobs = lambda c: c not in ev
        nearest = lambda cs: min(cs, key=lambda c: (dist(c, last), c))
        maximal = self.maximal(held)
        if len(maximal) >= 2 and self.cfg["policy"] == "disagree":
            cands = set()
            for h in maximal:
                cands |= set(h.cells) | self.ring(h)
            total = sum(self.weight(h) for h in maximal)
            best, bestd = None, 0.0
            for c in sorted(cands):
                if not unobs(c):
                    continue
                share = defaultdict(float)
                for h in maximal:
                    share[h.cells.get(c, self.EMPTY)] += self.weight(h)
                d = 1 - max(share.values()) / total
                if d > bestd + 1e-9 or (best is not None and abs(d - bestd) < 1e-9
                                         and dist(c, last) < dist(best, last)):
                    best, bestd = c, d
            if best is not None and bestd >= 0.05:
                return best, "split"
        if maximal:
            h = maximal[0]
            unknown = [c for c in h.cells if unobs(c)]
            if unknown:
                # X.6: test the hypothesis where it is exposed - in a part with no evidence yet,
                # as far from what has been seen as possible
                row, unhit = self.table.rows[h.rid], set()
                for pid, (dr, dc) in row.parts:
                    cells_p = [(h.anchor[0] + dr + r, h.anchor[1] + dc + c)
                               for p, (r, c) in self.table.footprint(pid)]
                    if not any(ev.get(c) == h.cells[c] for c in cells_p):
                        unhit |= set(cells_p)
                pool = [c for c in unknown if c in unhit] or unknown
                return max(pool, key=lambda c: (min(dist(c, o) for o in nonempty), c)), "verify"
            ring = self.ring(h)
            if not any(ev.get(c) == self.EMPTY for c in ring):
                ring_unobs = [c for c in ring if unobs(c)]
                if ring_unobs:
                    return nearest(ring_unobs), "boundary"
        frontier = [c for cell in nonempty for c in self.neighbors(cell) if unobs(c)]
        if frontier:
            return nearest(frontier), "explore"
        return None, None

    # ------------------------------------------------------------------ the episode
    def run_episode(self, world, ep, obj_id=None):
        table, wm, cfg = self.table, self.wm, self.cfg
        wm.clear()
        obj = world.start(obj_id)
        known_before = table.find(world.truth_fp())
        verified = defaultdict(lambda: [set(), set()])
        stats = dict(ep=ep, obj=obj, glimpses=0, compactions=0, clicks=0, ideas=0, ghosts=0, tries=0,
                     overwhelmed=0, peak=0, known_before=known_before is not None)
        self.log(f"\nep {ep}  object {obj} at {world.current[1]}  "
                 f"({'known' if known_before is not None else 'never seen'} before)")
        cell = world.cue()
        rid = self.look(world, cell, [], verified)
        self.log(f"  cue  {cell} -> {table.rows[rid].label}")
        last, held, decision, H = cell, [], None, cfg["H"]
        for step in range(cfg["budget"]):
            items, ghosts = wm.read()
            ev, category_ghosts = self.evidence(items)
            stats["ghosts"] += ghosts
            hyps = self.hypotheses(ev)
            if cfg["compact"] == "greedy" or wm.pressure() > cfg["eps"]:
                if self.compact(hyps, ep, stats):
                    continue
            held = self.hold(hyps, H)
            if wm.pressure() > cfg["eps"]:                       # VIII.1-2: over-stimulated
                while wm.pressure() > cfg["eps"] and H > 2:
                    H -= 1
                    held = self.hold(hyps, H)
                if wm.pressure() > cfg["eps"]:
                    if self.try_compaction(ep, stats):
                        continue
                    stats["overwhelmed"] += 1
                    self.log(f"     overwhelmed (load {wm.load()}, pressure {wm.pressure():.2f}): stop looking")
                    break
            decision = self.terminate(held, ev, verified)
            if decision:
                break
            cell, why = self.policy(held, ev, last)
            if cell is None:
                break
            rid = self.look(world, cell, held, verified)
            last = cell
            hs = " ".join(f"{table.rows[h.rid].label}@{h.anchor}x{h.matched}" for h in held[:4])
            self.log(f"  {why:8s} {cell} -> {table.rows[rid].label:2s}  held: {hs}   "
                     f"load {wm.load()}  pressure {wm.pressure():.2f}" + (f"  ghosts {ghosts}" if ghosts else ""))
        stats["glimpses"] = world.glimpses
        stats["peak"] = wm.peak
        self.introspect()
        return self.finish(world, ep, decision, verified, stats)

    def introspect(self):
        """IX.2: unbind WM with each provenance tag -> what do I observe / infer / assume?"""
        items, _ = self.wm.read()
        for tag in TAGS:
            xs = [f"{self.table.rows[rid].label}@{cell}" + (f"x{m}" if m > 1 else "")
                  for t, rid, cell, m in items if t == tag and rid != self.EMPTY]
            self.log(f"  {tag:8s}: {' '.join(xs) if xs else '-'}")

    # ------------------------------------------------------------------ episode end
    def finish(self, world, ep, decision, verified, stats):
        table = self.table
        truth = world.truth()
        items, _ = self.wm.read()
        ev, _ = self.evidence(items)
        declared = decision[1] if decision else None
        tested = {rid for (rid, anchor), (vin, vout) in verified.items() if len(vin) >= 1}
        for rid in tested:
            table.rows[rid].worked += 1
        prov_tested = [table.rows[rid].label for rid in tested if table.rows[rid].status == "provisional"]
        if prov_tested:
            self.log(f"  tested ok: {' '.join(prov_tested)} predicted unseen cells correctly")
        encoding, top = None, []
        if declared is not None:
            row = table.rows[declared.rid]
            correct = declared.cells == truth
            row.top_uses += 1
            table.touch(row, ep)
            encoding = row
            stats["outcome"] = "recognized" if correct else "wrong"
            stats["declared"] = row.label
            self.log(f"  => {row.label} @{declared.anchor}: {'correct' if correct else 'WRONG'}  "
                     f"({world.glimpses} glimpses, verified {len(verified[declared.key][0])} predictions)")
        else:
            stats["outcome"] = "novel" if not stats["known_before"] else "missed"
            stats["declared"] = "-"
            self.log(f"  => no single explanation ({world.glimpses} glimpses); "
                     f"{'a new thing' if not stats['known_before'] else 'MISSED a known object'}")
            while True:                                   # reflection: anything here fully matching a row?
                items, _ = self.wm.read()
                ev, _ = self.evidence(items)
                if not self.compact(self.hypotheses(ev), ep, stats):
                    break
            items, _ = self.wm.read()
            top = [(rid, cell) for tag, rid, cell, m in items if rid != self.EMPTY
                   and ((tag == "observed" and table.rows[rid].is_prim)
                        or (tag == "inferred" and not table.rows[rid].is_prim))]
            if len(top) >= 2:                             # remember the whole scene
                fp = normalize((p, (cell[0] + dr, cell[1] + dc)) for rid, cell in top
                               for p, (dr, dc) in table.footprint(rid))
                row, created = table.propose(fp, ep)
                row.top_uses += 1
                encoding = row
                if created:
                    self.log(f"  + remember scene as {row.label} = {table.describe(row)}")
                else:
                    self.log(f"  = that was {row.label} again")
            elif len(top) == 1:
                encoding = table.rows[top[0][0]]
                encoding.top_uses += 1
        stats["end_items"] = len(top) if declared is None else 1
        stats["ideas"] += self.ideas(ev, ep, exclude={encoding.id} if encoding else set())
        for r in table.clicks(ep):
            stats["clicks"] += 1
        pred = {c: p for c, p in ev.items() if p != self.EMPTY}
        if declared is not None:
            pred.update(declared.cells)
        hit = sum(1 for c, p in pred.items() if truth.get(c) == p)
        stats["completion"] = hit / len(set(pred) | set(truth))
        stats["correct"] = stats["outcome"] == "recognized"
        return stats

    def ideas(self, ev, ep, exclude):
        """Notice that part of what I see now is also part of something I remember, and give
        that shared part a name of its own (provisional) - if it recurs in at least two
        independent things.  Whether it pays is MDL's call, and it must still pass a test."""
        E = {c: p for c, p in ev.items() if p != self.EMPTY}
        if len(E) < 2:
            return 0
        table, n_new = self.table, 0
        for r in list(table.chunks()):
            if r.id in exclude:
                continue
            fp = table.footprint(r.id)
            if len(fp) < 3:
                continue
            votes = defaultdict(int)
            for c, p in E.items():
                for pp, (dr, dc) in fp:
                    if pp == p:
                        votes[(c[0] - dr, c[1] - dc)] += 1
            if not votes:
                continue
            a = max(votes, key=lambda k: (votes[k], k))
            S = [(p, (a[0] + dr, a[1] + dc)) for p, (dr, dc) in fp if E.get((a[0] + dr, a[1] + dc)) == p]
            if len(S) < 2 or len(S) == len(fp):
                continue
            sfp = normalize(S)
            if table.find(sfp) is not None or table.refs_fp(sfp) < 2:
                continue
            row, created = table.propose(sfp, ep, label_prefix="i")
            if created:
                n_new += 1
                self.log(f"  ! idea: {len(S)} cells here are also part of {r.label}  -> {row.label} = {table.describe(row)}")
        return n_new
