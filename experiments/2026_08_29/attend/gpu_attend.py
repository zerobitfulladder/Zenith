"""Rung 1 of attention: the architecture chooses its own cue.

The deep-outer rig (dx, dy, angle channels; present bumps + two-layer
temporal tracks) collapsed to EVAL 0.05 because its cue carried
everything, democratically. Here, BEFORE the top layer learns, the rig
measures each cue block's relevance from its own stream:

    relevance of block b = information its quantised content carries
    about the action label, selected GREEDILY with conditioning on
    already-chosen blocks (so redundant history cannot free-ride on
    the present) and shuffle-bias correction.

Quantisers cost nothing: a present block's bump cell index; a context
block's L2 winner. Shares are set from the surviving blocks' gains;
the top layer and cetele then train on the gated cue.

Ground truth is known: pres-dx and pres-dy carry the outer law; the
angle carries nothing; history is redundant given the present. If the
rig cannot rediscover this by itself, rung 1 fails.

Arms: "learned" (shares from the MI selection) and "handtruth"
(pres-dx = pres-dy = 0.5, rest 0 — plumbing control, should match the
memoryless champion).

Run:  .venv/bin/python experiments/2026_08_29/attend/gpu_attend.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chase"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascade"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deep_outer"))
from gpu_chase import (xp, asnumpy, physics, init_worlds,   # noqa: E402
                       EP_CAP)
import gpu_cascade as C                                     # noqa: E402
from gpu_cascade import (vcmd, code_vc, snap_levels,        # noqa: E402
                         inner_label)
import gpu_deep_outer as D                                  # noqa: E402
from gpu_deep_outer import (DeepOuter, evaluate, NS, K2,    # noqa: E402
                            NB, HIST, BANKS, STG1, STG2, DPRES)


class SlicedStage(C.Stage):
    """Attention as PRUNING: suppressed blocks do not occupy dims.
    Zeroed-but-present columns pollute centering, boot noise and the
    unit renorm (all global over the row) — measured as perfect
    approach with a drowned hold."""

    def __init__(self, k, cols, kb, seed):
        super().__init__(k, len(cols), kb, seed)
        self.cols = xp.asarray(np.asarray(cols))

    def winners(self, cue):
        return super().winners(cue[:, self.cols])

OUT = Path(__file__).resolve().parent / "results"
B = D.B
MI_TICKS = int(os.environ.get("GA_MI", "1000000"))
TOP_TICKS = int(os.environ.get("GA_TOP", "200000"))
NOTCH_TICKS = int(os.environ.get("GA_NOTCH", "300000"))
NBLK = 2 * NS


def run_phase(rig, rng, S, T, ep_tick, ticks, stage, on_outer=None):
    for step in range(ticks // B):
        te = step * B
        lm = (np.arange(B) + step) % 5 == 0
        sv = rig.sense(S, T)
        cue = rig.push(sv, lm, stage)
        lab_vc = xp.stack([vcmd(S[:, 0] - T[:, 0]),
                           vcmd(S[:, 1] - T[:, 1])], axis=1)
        if on_outer is not None and step % 5 == 0:
            on_outer(rig, cue, sv, lab_vc, lm, te)
        li, ri = snap_levels(*inner_label(S, lab_vc))
        S = physics(S, li, ri)
        ep_tick += 1
        over = ep_tick >= EP_CAP
        if over.any():
            n = int(over.sum())
            ns_, nt = init_worlds(rng, n)
            om = xp.asarray(over)
            S[om] = ns_
            T[om] = nt
            rig.reset(over)
            ep_tick[over] = 0
    return S, T, ep_tick


def entropy(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def cond_mi(joint_sel, qb, u, nu=32):
    """I(qb; U | selected), plug-in, via sparse joint indexing."""
    base = np.unique(joint_sel, return_inverse=True)[1]
    hu_given_sel = 0.0
    hu_given_selb = 0.0
    n = len(u)
    for keyed, arr in (("sel", base), ("selb", base * 100000
                                       + np.asarray(qb))):
        k = np.unique(arr, return_inverse=True)[1]
        joint = k * nu + np.asarray(u)
        cj = np.bincount(joint)
        ck = np.bincount(k)
        h = 0.0
        for kk in np.nonzero(ck)[0]:
            seg = cj[kk * nu:(kk + 1) * nu]
            h += ck[kk] / n * entropy(seg)
        if keyed == "sel":
            hu_given_sel = h
        else:
            hu_given_selb = h
    return hu_given_sel - hu_given_selb


def select_blocks(Q, U, rng):
    """Greedy conditional-MI selection with shuffle-bias correction."""
    names = [f"pres-{c}" for c in ("dx", "dy", "ang")] + \
            [f"ctx-{c}" for c in ("dx", "dy", "ang")]
    n = len(U)
    Ushuf = U.copy()
    rng.shuffle(Ushuf)
    sel, gains = [], []
    joint = np.zeros(n, np.int64)
    print("greedy relevance selection:")
    for rnd in range(4):
        best_b, best_g = -1, 0.0
        row = {}
        for b in range(NBLK):
            if b in sel:
                continue
            g = cond_mi(joint, Q[:, b], U)
            g -= cond_mi(joint, Q[:, b], Ushuf)   # bias correction
            row[names[b]] = round(g, 3)
            if g > best_g:
                best_b, best_g = b, g
        print(f"  round {rnd}: {row}")
        thresh = max(0.05, 0.1 * (gains[0] if gains else best_g))
        if best_b < 0 or best_g < thresh:
            break
        sel.append(best_b)
        gains.append(best_g)
        joint = joint * 1000 + np.asarray(Q[:, best_b], np.int64)
        print(f"    -> selected {names[best_b]} (gain {best_g:.3f})")
    shares = np.zeros(NBLK)
    tot = sum(gains)
    for b, g in zip(sel, gains):
        shares[b] = g / tot
    chosen = {names[i]: round(float(shares[i]), 3)
              for i in range(NBLK) if shares[i] > 0}
    print(f"shares: {chosen}")
    return shares


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rig = DeepOuter()
    rng = np.random.default_rng(0)
    erng = np.random.default_rng(99)
    S, T = init_worlds(rng, B)
    ep_tick = np.zeros(B, np.int64)
    t0 = time.time()

    # banks + track layers, exactly as before ---------------------------
    def bank_hook(rig_, cue, sv, lab_vc, lm, te):
        rig_.top.bank.learn_rows(code_vc(lab_vc)[xp.asarray(lm)])
    S, T, ep_tick = run_phase(rig, rng, S, T, ep_tick, BANKS, 0,
                              bank_hook)
    S, T, ep_tick = run_phase(rig, rng, S, T, ep_tick, STG1 - BANKS, 0)
    S, T, ep_tick = run_phase(rig, rng, S, T, ep_tick, STG2 - STG1, 1)
    print(f"tracks trained ({(time.time() - t0) / 60:.1f} min)",
          flush=True)

    # MI stream ---------------------------------------------------------
    stream_q, stream_u = [], []
    VCU = np.array([C.decode_bank(rig.top.bank.W[u], 0.2, 8.0, 0.2, 8.0)
                    for u in range(32)], np.float32)

    def mi_hook(rig_, cue, sv, lab_vc, lm, te):
        qp = np.clip(np.round((asnumpy(sv) + 1) / 2 * (NB - 1)),
                     0, NB - 1).astype(np.int16)
        codes = asnumpy(rig_._last_codes)
        qc = np.where(codes.sum(2) > 0, codes.argmax(2),
                      K2).astype(np.int16)
        d2 = ((asnumpy(lab_vc)[:, None, :] - VCU[None]) ** 2).sum(2)
        u = d2.argmin(1).astype(np.int16)
        ready = asnumpy(rig_.nw >= HIST)
        stream_q.append(np.concatenate([qp, qc], axis=1)[ready])
        stream_u.append(u[ready])
    S, T, ep_tick = run_phase(rig, rng, S, T, ep_tick, MI_TICKS, 2,
                              mi_hook)
    Q = np.concatenate(stream_q)
    U = np.concatenate(stream_u)
    print(f"MI stream: {len(U)} decisions "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    shares = select_blocks(Q, U, np.random.default_rng(5))

    # arms: learned shares vs hand truth --------------------------------
    results = {}
    for arm, sh in (("learned", shares),
                    ("handtruth",
                     np.array([0.5, 0.5, 0, 0, 0, 0]))):
        rig.shares = list(sh)
        cols = []
        for b in range(NBLK):
            if sh[b] > 0:
                if b < NS:
                    cols.extend(range(b * NB, (b + 1) * NB))
                else:
                    c = b - NS
                    cols.extend(range(DPRES + c * K2,
                                      DPRES + (c + 1) * K2))
        bank = rig.top.bank              # keep the trained vocabulary
        rig.top = SlicedStage(256, cols, 32, seed=1234)
        rig.top.bank = bank
        # boot from episode STARTS: a fresh top seeds its partition
        # from the first cues it sees, and mid-stream those are all
        # hover-at-target dwell — the champion booted at spawn time
        S, T = init_worlds(rng, B)
        rig.reset(np.ones(B, bool))
        ep_tick = np.zeros(B, np.int64)

        def top_hook(rig_, cue, sv, lab_vc, lm, te):
            sel = xp.asarray(lm) & (rig_.nw >= 1)
            if bool(sel.any()):
                rig_.top.top.learn_rows(cue[sel][:, rig_.top.cols])
        S, T, ep_tick = run_phase(rig, rng, S, T, ep_tick, TOP_TICKS,
                                  2, top_hook)

        def notch_hook(rig_, cue, sv, lab_vc, lm, te):
            w = rig_.top.winners(cue)
            rig_.top.notch(w, code_vc(lab_vc), xp.ones(B, bool))
        S, T, ep_tick = run_phase(rig, rng, S, T, ep_tick, NOTCH_TICKS,
                                  2, notch_hook)
        ev, near, med = evaluate(rig, VCU, erng)
        results[arm] = {"EVAL": round(ev, 3), "closest": round(near, 2),
                        "median_ticks": med,
                        "shares": [round(float(x), 3) for x in sh]}
        print(f"{arm}: EVAL {ev:.2f}  closest {near:.2f}  "
              f"median {med:.0f}", flush=True)
        np.savez(OUT / f"weights_{arm}.npz",
                 **{f"l1_{c}": asnumpy(rig.l1[c].W) for c in range(NS)},
                 **{f"l2_{c}": asnumpy(rig.l2[c].W) for c in range(NS)},
                 Wo=asnumpy(rig.top.top.W), Bo=asnumpy(rig.top.bank.W),
                 Co=asnumpy(rig.top.C),
                 cols=np.asarray(asnumpy(rig.top.cols)),
                 shares=np.asarray(sh, np.float32))
        json.dump({"kind": "module", "module": "attend_view",
                   "dir": "experiments/2026_08_29/temporal_drone",
                   "arm": arm, "EVAL": round(ev, 3)},
                  open(OUT / f"weights_{arm}.json", "w"))

    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"done ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
