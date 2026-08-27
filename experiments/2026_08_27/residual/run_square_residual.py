"""Bouncing square: SAME-RATE chassis + COMPLETION READ (explain-away).

The user's design, assembled from today's session:
  - CHASSIS (user): every layer ticks and speaks EVERY input tick;
    timescale separation lives in decay alone (G1=0.5, G2=0.9).
    Upward speech is one-hot, so slow integration is interference-free
    (orthogonal inputs -> a lossless who-fired-how-recently record).
  - GRAIN: L2 starved (K2=8 rows for a 40-phase cycle) so each row is
    an ARC of the cycle, not a moment. L1 stays at full capacity
    (K1=64) — rendering detail is not taxed; coarseness lives only
    where the arrow lives.
  - ARROW = THE READ (user's "subtract the up-to-now"): retrieve the
    arc whose era matches the running unit-histogram (completion),
    subtract what the histogram already explains; the relu'd residual
    is the era's NOT-YET-ELAPSED part. Used as a GATE: it picks the
    candidate pool of L1 units; the L1 trail picks the soonest of
    them (gate-then-match — the validated hypercolumn recipe). The
    backward neighbor just fired, so the subtraction erases it from
    the pool: the subtraction IS the symmetry breaker.

No next slots, no lag channels, no empty query channels, no
refractory. Storage is plain causal snapshots everywhere; the arrow
exists only in the read.

Instruments: teacher-forced plain-read self fraction (chassis check:
expected ~1.0 present-detector), teacher-forced NEXT-UNIT offsets
(the verdict: forward fraction), emission lead, free-run error/GIF.

Run: .venv/bin/python experiments/2026_08_27/residual/run_square_residual.py
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

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "residual" / "results"
K1 = int(os.environ.get("SR_K1", "64"))
K2 = 8
SEL = os.environ.get("SR_SEL", "trail")   # trail | resid (recovery order)
COMM = os.environ.get("SR_COMM", "onehot")  # onehot | dense upward speech
G1, G2 = 0.5, 0.9
GATE_TH = 0.05
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


def upmsg(b1, T1, w1):
    """Upward speech: committed one-hot, or the graded relu profile."""
    if COMM == "dense":
        return np.maximum(b1.W @ cn1(T1), 0.0)
    return onehot(w1, K1)


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
    homes1, homes2 = {}, {}

    for ep in range(EPOCHS):
        T1 = np.zeros(D, np.float32)
        T2 = np.zeros(K1, np.float32)
        for t in range(TRAIN_FRAMES):
            T1 = frame_at(t)[0].ravel().astype(np.float32) + G1 * T1
            w1 = learn(b1, T1)
            if w1 < 0:
                continue
            T2 = upmsg(b1, T1, w1) + G2 * T2
            w2 = learn(b2, T2)
            if ep == EPOCHS - 1:
                homes1.setdefault(w1, set()).add(t % PERIOD)
                if w2 >= 0:
                    homes2.setdefault(w2, set()).add(t % PERIOD)

    arc_lines = []
    for w in sorted(homes2):
        ph = sorted(homes2[w])
        arc_lines.append(f"    L2 row {w}: {len(ph)} phases {ph}")

    def next_unit(T1, T2, lam, mu, th):
        """Completion read: arc - lam*fired - mu*content-explained; gate,
        then the trail picks the soonest surviving candidate."""
        C2 = b2.W @ cn1(T2)
        tcorr = b1.W @ cn1(T1)
        for r2 in np.argsort(C2)[::-1][:3]:
            whole = np.maximum(b2.W[r2], 0.0)
            resid = np.maximum(
                unit(whole) - lam * unit(np.maximum(T2, 0.0))
                - mu * np.maximum(tcorr, 0.0), 0.0)
            pool = np.where(resid > th)[0]
            if len(pool):
                if SEL == "resid":        # most-recovered = next due
                    return int(pool[np.argmax(resid[pool])])
                return int(pool[np.argmax(tcorr[pool])])
        return int(np.argmax(tcorr))

    def tf_pass(lam, mu, th):
        T1 = np.zeros(D, np.float32)
        T2 = np.zeros(K1, np.float32)
        plain_offs, next_offs, lead = [], [], []
        for t in range(PRIME + FREE):
            T1 = frame_at(t)[0].ravel().astype(np.float32) + G1 * T1
            w1 = int(np.argmax(b1.W @ cn1(T1)))
            if t >= PRIME:
                ph = t % PERIOD
                o = signed_off(homes1, ph, w1)
                if o is not None:
                    plain_offs.append(o)
                wn = next_unit(T1, T2, lam, mu, th)
                o = signed_off(homes1, ph, wn)
                if o is not None:
                    next_offs.append(o)
                gx = decode_x(deblur(np.maximum(b1.W[wn], 0.0)))
                x_now, x_nxt = frame_at(t)[1], frame_at(t + 1)[1]
                lead.append((gx - x_now) * (1 if x_nxt > x_now else -1))
            T2 = upmsg(b1, T1, w1) + G2 * T2
        return (np.array(plain_offs), np.array(next_offs), np.array(lead))

    # ---- Teacher-forced sweep over subtraction strengths ------------------
    po, _, _ = tf_pass(1.0, 0.0, GATE_TH)
    tf_lines = [
        f"K1={K1}",
        f"TF plain read (chassis check, n={len(po)}): self "
        f"{(po == 0).mean():.2f} / fwd {(po > 0).mean():.2f} / "
        f"back {(po < 0).mean():.2f}",
        "TF sweep  lam   mu  | forward  self  backward |  mean  | lead ahead",
    ]
    best, best_score = (1.0, 0.0), -9.0
    for lam in (1.0, 2.0):
        for mu in (0.0, 0.5, 1.0, 2.0):
            _, no, ld = tf_pass(lam, mu, GATE_TH)
            fwd, back = (no > 0).mean(), (no < 0).mean()
            score = fwd - back
            tf_lines.append(
                f"        {lam:4.1f} {mu:4.1f} |   {fwd:.2f}   "
                f"{(no == 0).mean():.2f}    {back:.2f}   | {no.mean():+5.2f} "
                f"|   {(ld > 0).mean():.2f}")
            if score > best_score:
                best, best_score = (lam, mu), score
    LAM, MU = best
    TH = GATE_TH
    tf_lines.append(f"best (fwd-back): lam={LAM}, mu={MU}")

    # ---- Free run ---------------------------------------------------------
    T1 = np.zeros(D, np.float32)
    T2 = np.zeros(K1, np.float32)
    for t in range(PRIME):
        T1 = frame_at(t)[0].ravel().astype(np.float32) + G1 * T1
        w1 = int(np.argmax(b1.W @ cn1(T1)))
        T2 = upmsg(b1, T1, w1) + G2 * T2

    emission = frame_at(PRIME - 1)[0]
    gen, gen_x, true_now, offs = [], [], [], []
    for i in range(FREE):
        T1 = emission.ravel().astype(np.float32) + G1 * T1
        w1p = int(np.argmax(b1.W @ cn1(T1)))       # re-perceive own emission
        T2 = upmsg(b1, T1, w1p) + G2 * T2
        wn = next_unit(T1, T2, LAM, MU, TH)
        emission = deblur(np.maximum(b1.W[wn], 0.0))
        gen.append(emission)
        gen_x.append(decode_x(emission))
        true_now.append(frame_at(PRIME + i)[1])
        o = signed_off(homes1, (PRIME + i) % PERIOD, wn)
        if o is not None:
            offs.append(o)

    save_gif(gen, OUTPUT_DIR / "generated.gif")
    np.savez(OUTPUT_DIR / "gen_frames.npz", gen=np.array(gen))
    gen_x = np.array(gen_x)
    err = np.abs(np.array(true_now) - gen_x)
    offs = np.array(offs)
    lines = tf_lines + [
        f"FREE-RUN {FREE} emissions: mean |x err| {err.mean():.2f} "
        f"(max {err.max()})  (refs: arrow forms 0.00, trail-only 8.12, "
        f"completion-plain 10.02, refractory 7.49)",
        f"free-run next-unit offsets: forward {(offs > 0).mean():.2f} / "
        f"self {(offs == 0).mean():.2f} / backward {(offs < 0).mean():.2f}",
        "first 33 (gen_x, true_x): "
        + " ".join(f"({g},{tn})" for g, tn in zip(gen_x[:33], true_now[:33])),
        "L2 arcs (row: home phases, final epoch):",
    ] + arc_lines
    report = "\n".join(lines)
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Square, same-rate chassis + completion read\n\n" + report + "\n")


if __name__ == "__main__":
    main()
