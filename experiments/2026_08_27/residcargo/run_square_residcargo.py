"""Bouncing square: RESIDUAL-AS-CARGO (user's synthesis, evening).

The subtraction becomes a SELF-COMPUTED next-slot. Old next-slot rigs
filled the cargo channel with oracle frame(t+1); here the cargo is
the network's own top-down residual — "the not-yet-elapsed part of my
era" = arc-whole minus explained history, computed from L2's
completion at STORAGE time (against the true firing history). Content
about strictly-later time, written into the vector at storage: the
arrow law satisfied with no oracle and no explicit next pathway.

Phase 1 (unchanged same-rate chassis): b1 = moment dictionary over
frame trails (K1=64), b2 = era arcs over the unit histogram (K2=8),
one-hot speech upward, decay-only timescales.
Phase 2: bank b3 stores [cn(trail) ; GC * cn(residual_t)] jointly,
residual_t = relu(unit(arc_whole) - LAM*unit(hist) - MU*relu(tcorr)).
GC=0.5: cargo is a small-gain minority — KEYS DOMINATE (banked law).

Playback: query b3 with the residual channel EMPTY (lag-advance
form); the winner's stored residual names the upcoming units (the
pool); the trail picks the soonest — backward units were subtracted
at storage, so nearest-in-pool = forward. Emit that unit's frame.
Self-match is now a FEATURE: my own row's cargo is my future.

Run: .venv/bin/python experiments/2026_08_27/residcargo/run_square_residcargo.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_27" / "completion"))

import numpy as np

from run_square_completion import cn1, deblur, decode_x, onehot  # noqa: E402
from run_temporal_square import (  # noqa: E402
    D,
    PERIOD,
    PRIME,
    TRAIN_FRAMES,
    frame_at,
    save_gif,
)
from run_gpu_minibatch import Dict  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "residcargo" / "results"
K1, K2, K3 = 64, 8, 64
G1, G2 = 0.5, 0.9
LAM, MU, TH = 2.0, 0.5, 0.05
GC = 0.5                      # cargo gain: small minority of the row
EPOCHS = 3
FREE = 100


def unit(v):
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


def learn(bank, v):
    z = cn1(v)[None, :]
    if bank.n_boot < bank.k:
        bank.bootstrap(z)
        return -1
    C = (z @ bank.W.T)[0]
    w = int(np.argmax(C))
    bank.update(z, np.array([w]),
                np.array([max(C[w], 0.0)], dtype=np.float32))
    return w


def signed_off(homes, ph, w):
    hs = homes.get(w)
    if not hs:
        return None
    best = None
    for hp in hs:
        d = ((hp - ph + PERIOD // 2) % PERIOD) - PERIOD // 2
        if best is None or abs(d) < abs(best):
            best = d
    return best


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    b1, b2 = Dict(K1, D, 0.05), Dict(K2, K1, 0.05)
    homes1 = {}

    # ---- Phase 1: moment dictionary + era arcs (as validated today) -------
    for ep in range(EPOCHS):
        T1 = np.zeros(D, np.float32)
        T2 = np.zeros(K1, np.float32)
        for t in range(TRAIN_FRAMES):
            T1 = frame_at(t)[0].ravel().astype(np.float32) + G1 * T1
            w1 = learn(b1, T1)
            if w1 < 0:
                continue
            T2 = onehot(w1, K1) + G2 * T2
            learn(b2, T2)
            if ep == EPOCHS - 1:
                homes1.setdefault(w1, set()).add(t % PERIOD)

    def residual(T1, T2):
        """The era's not-yet-elapsed part, per the top-down completion."""
        r2 = int(np.argmax(b2.W @ cn1(T2)))
        whole = np.maximum(b2.W[r2], 0.0)
        tcorr = b1.W @ cn1(T1)
        return np.maximum(unit(whole) - LAM * unit(np.maximum(T2, 0.0))
                          - MU * np.maximum(tcorr, 0.0), 0.0)

    # ---- Phase 2: store [trail ; residual] jointly ------------------------
    b3 = Dict(K3, D + K1, 0.05)
    homes3 = {}
    for ep in range(2):
        T1 = np.zeros(D, np.float32)
        T2 = np.zeros(K1, np.float32)
        for t in range(TRAIN_FRAMES):
            T1 = frame_at(t)[0].ravel().astype(np.float32) + G1 * T1
            w1 = int(np.argmax(b1.W @ cn1(T1)))
            T2 = onehot(w1, K1) + G2 * T2
            z = np.concatenate([cn1(T1), GC * cn1(residual(T1, T2))])
            w3 = learn(b3, z)
            if ep == 1 and w3 >= 0:
                homes3.setdefault(w3, set()).add(t % PERIOD)

    def step(T1):
        """One playback step: lag-advance query, cargo names the pool,
        trail picks the soonest survivor. Returns next unit."""
        q = cn1(np.concatenate([cn1(T1), np.zeros(K1, np.float32)]))
        w3 = int(np.argmax(b3.W @ q))
        cargo = np.maximum(b3.W[w3, D:], 0.0)
        tcorr = b1.W @ cn1(T1)
        pool = np.where(cargo > TH)[0]
        if len(pool):
            return int(pool[np.argmax(tcorr[pool])]), w3
        return int(np.argmax(cargo)), w3

    # ---- Teacher-forced -----------------------------------------------------
    T1 = np.zeros(D, np.float32)
    offs, lead = [], []
    for t in range(PRIME + FREE):
        T1 = frame_at(t)[0].ravel().astype(np.float32) + G1 * T1
        if t < PRIME:
            continue
        wn, _ = step(T1)
        o = signed_off(homes1, t % PERIOD, wn)
        if o is not None:
            offs.append(o)
        gx = decode_x(deblur(np.maximum(b1.W[wn], 0.0)))
        x_now, x_nxt = frame_at(t)[1], frame_at(t + 1)[1]
        lead.append((gx - x_now) * (1 if x_nxt > x_now else -1))
    offs, lead = np.array(offs), np.array(lead)
    tf_lines = [
        f"TF next-unit via stored residual cargo (n={len(offs)}): forward "
        f"{(offs > 0).mean():.2f} / self {(offs == 0).mean():.2f} / backward "
        f"{(offs < 0).mean():.2f}, mean {offs.mean():+.2f}, "
        f"median {np.median(offs):+.1f}",
        f"TF emission lead (+ = ahead): mean {lead.mean():+.2f}, ahead "
        f"{(lead > 0).mean():.2f} / at-present {(lead == 0).mean():.2f} / "
        f"behind {(lead < 0).mean():.2f}",
    ]

    # ---- Free run -----------------------------------------------------------
    T1 = np.zeros(D, np.float32)
    for t in range(PRIME):
        T1 = frame_at(t)[0].ravel().astype(np.float32) + G1 * T1
    emission = frame_at(PRIME - 1)[0]
    gen, gen_x, true_now, foffs = [], [], [], []
    for i in range(FREE):
        T1 = emission.ravel().astype(np.float32) + G1 * T1
        wn, _ = step(T1)
        emission = deblur(np.maximum(b1.W[wn], 0.0))
        gen.append(emission)
        gen_x.append(decode_x(emission))
        true_now.append(frame_at(PRIME + i)[1])
        o = signed_off(homes1, (PRIME + i) % PERIOD, wn)
        if o is not None:
            foffs.append(o)

    save_gif(gen, OUTPUT_DIR / "generated.gif")
    np.savez(OUTPUT_DIR / "gen_frames.npz", gen=np.array(gen))
    gen_x = np.array(gen_x)
    err = np.abs(np.array(true_now) - gen_x)
    foffs = np.array(foffs)
    lines = tf_lines + [
        f"FREE-RUN {FREE} emissions: mean |x err| {err.mean():.2f} "
        f"(max {err.max()})  (refs: arrow forms 0.00; completion family "
        f"floor 4.98)",
        f"free-run offsets: forward {(foffs > 0).mean():.2f} / self "
        f"{(foffs == 0).mean():.2f} / backward {(foffs < 0).mean():.2f}",
        "first 33 (gen_x, true_x): "
        + " ".join(f"({g},{tn})" for g, tn in zip(gen_x[:33], true_now[:33])),
    ]
    report = "\n".join(lines)
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Square, residual-as-cargo (self-computed next-slot)\n\n"
        + report + "\n")


if __name__ == "__main__":
    main()
