"""The co-adaptive loop (user's design, tried the night it was born).

No phases, no rebuild. From the broken uniform-share configuration,
three things improve together, on three timescales:

    tally (fastest)   leaky cetele — counts decay, so conditionals
                      forget at the rate the partition drifts
    templates (mid)   the top layer NEVER freezes: geodesic rotation
                      is a recency-weighted average and tracks the
                      drifting cue geometry natively
    mask (slowest)    every SHARE_EVERY ticks, greedy conditional-MI
                      selection on a sliding window of recent
                      decisions nudges the shares toward relevance

Watch: do the shares concentrate on pres-dx/dy by themselves, and
does EVAL climb out of 0.05 — and past the rebuild harness's broken
hold?

Run:  .venv/bin/python experiments/2026_08_29/attend/gpu_coadapt.py
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
from gpu_cascade import vcmd, code_vc, snap_levels, inner_label  # noqa
import gpu_deep_outer as D                                  # noqa: E402
from gpu_deep_outer import (DeepOuter, evaluate, NS, K2, NB,  # noqa
                            HIST, BANKS, STG1, STG2)
from gpu_attend import select_blocks, run_phase             # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "coadapt"
B = D.B
TICKS = int(os.environ.get("GC_TICKS", "2000000"))
REPORT = int(os.environ.get("GC_REPORT", "200000"))
SHARE_EVERY = int(os.environ.get("GC_SHARE", "100000"))
WIN = 60000                     # sliding window of decisions
LEAK = 1.0 - B / 15000.0        # cetele memory ~15k decisions
SLEW = 0.3                      # share nudge per update


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rig = DeepOuter()
    rig.shares = [0.5 / NS] * NS + [0.5 / NS] * NS   # the broken start
    rng = np.random.default_rng(0)
    erng = np.random.default_rng(99)
    S, T = init_worlds(rng, B)
    ep_tick = np.zeros(B, np.int64)
    t0 = time.time()

    def bank_hook(rig_, cue, sv, lab_vc, lm, te):
        rig_.top.bank.learn_rows(code_vc(lab_vc)[xp.asarray(lm)])
    S, T, ep_tick = run_phase(rig, rng, S, T, ep_tick, BANKS, 0,
                              bank_hook)
    S, T, ep_tick = run_phase(rig, rng, S, T, ep_tick, STG1 - BANKS, 0)
    S, T, ep_tick = run_phase(rig, rng, S, T, ep_tick, STG2 - STG1, 1)
    VCU = np.array([C.decode_bank(rig.top.bank.W[u], 0.2, 8.0, 0.2, 8.0)
                    for u in range(32)], np.float32)
    # THE VOCABULARY AUDITION (law of 2026-08-30): before anything
    # downstream speaks through the bank, "zero" must decode to zero.
    tests = xp.asarray(np.array([[0.0, 0.0], [1.5, 0.0], [0.0, -1.5]],
                                np.float32))
    u = asnumpy(rig.top.bank.scores(code_vc(tests)).argmax(axis=1))
    for tv, uu in zip(asnumpy(tests), u):
        print(f"  audition: ({tv[0]:+.1f},{tv[1]:+.1f}) -> word {uu} "
              f"= ({VCU[uu][0]:+.2f},{VCU[uu][1]:+.2f})", flush=True)
    z0 = float(np.hypot(*VCU[u[0]]))
    print(f"  zero-word magnitude {z0:.2f} m/s "
          f"({'PASS' if z0 < 0.3 else 'FAIL — vocabulary broken'})",
          flush=True)
    print(f"tracks trained ({(time.time() - t0) / 60:.1f} min)",
          flush=True)

    Qbuf = np.zeros((WIN, 2 * NS), np.int16)
    Ubuf = np.zeros(WIN, np.int16)
    wpos, wfill = 0, 0
    stats = []
    names = ["p-dx", "p-dy", "p-ang", "c-dx", "c-dy", "c-ang"]

    for step in range(TICKS // B):
        te = step * B
        lm = (np.arange(B) + step) % 5 == 0
        sv = rig.sense(S, T)
        cue = rig.push(sv, lm, 2)
        lab_vc = xp.stack([vcmd(S[:, 0] - T[:, 0]),
                           vcmd(S[:, 1] - T[:, 1])], axis=1)
        # top layer: alive forever (rotation = EMA, tracks the mask)
        sel = xp.asarray(lm) & (rig.nw >= 1)
        if bool(sel.any()):
            rig.top.top.learn_rows(cue[sel])
        if step % 5 == 0:
            # leaky cetele: forget at the rate the partition drifts
            rig.top.C *= LEAK
            w = rig.top.winners(cue)
            rig.top.notch(w, code_vc(lab_vc), xp.ones(B, bool))
            # relevance stream into the sliding window
            qp = np.clip(np.round((asnumpy(sv) + 1) / 2 * (NB - 1)),
                         0, NB - 1).astype(np.int16)
            codes = asnumpy(rig._last_codes)
            qc = np.where(codes.sum(2) > 0, codes.argmax(2),
                          K2).astype(np.int16)
            d2 = ((asnumpy(lab_vc)[:, None, :] - VCU[None]) ** 2).sum(2)
            u = d2.argmin(1).astype(np.int16)
            ready = asnumpy(rig.nw >= HIST)
            q = np.concatenate([qp, qc], axis=1)[ready]
            uu = u[ready]
            n = len(uu)
            idxs = (wpos + np.arange(n)) % WIN
            Qbuf[idxs] = q
            Ubuf[idxs] = uu
            wpos = (wpos + n) % WIN
            wfill = min(wfill + n, WIN)
        li, ri = snap_levels(*inner_label(S, lab_vc))
        S = physics(S, li, ri)
        ep_tick += 1
        over = ep_tick >= EP_CAP
        if over.any():
            nn = int(over.sum())
            ns_, nt = init_worlds(rng, nn)
            om = xp.asarray(over)
            S[om] = ns_
            T[om] = nt
            rig.reset(over)
            ep_tick[over] = 0

        if (te + B) % SHARE_EVERY < B and wfill > 20000:
            tgt_sh = select_blocks(Qbuf[:wfill], Ubuf[:wfill],
                                   np.random.default_rng(5))
            cur = np.asarray(rig.shares)
            rig.shares = list(cur + SLEW * (tgt_sh - cur))
            shown = {names[i]: round(rig.shares[i], 3)
                     for i in range(2 * NS)}
            print(f"  shares -> {shown}", flush=True)

        if (step + 1) % max(1, REPORT // B) == 0:
            ev, near, med = evaluate(rig, VCU, erng)
            row = {"exp_ticks": te + B, "EVAL": round(ev, 3),
                   "closest": round(near, 2), "median": med,
                   "shares": [round(float(x), 3) for x in rig.shares],
                   "mins": round((time.time() - t0) / 60, 1)}
            stats.append(row)
            print(json.dumps(row), flush=True)
            np.savez(OUT / "checkpoint.npz",
                     **{f"l1_{c}": asnumpy(rig.l1[c].W)
                        for c in range(NS)},
                     **{f"l2_{c}": asnumpy(rig.l2[c].W)
                        for c in range(NS)},
                     Wo=asnumpy(rig.top.top.W),
                     Bo=asnumpy(rig.top.bank.W),
                     Co=asnumpy(rig.top.C),
                     cols=np.arange(D.D3),
                     shares=np.asarray(rig.shares, np.float32))
            with open(OUT / "metrics.json", "w") as f:
                json.dump(stats, f, indent=2)

    print(f"done ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
