"""Bouncing square, completion arrow + REFRACTORY at L1 (Arm R).

The user's isolation test of the completion arrow, WITHOUT starving
capacity: keep the full rig from run_square_completion (same training,
same banks, same instrument), and at playback forbid the self-match at
L1 — the winner cannot repeat consecutively (the hypercolumn-day
refractory rule). If the top-down future-tint is real, the best
surviving candidate should sit AHEAD of the present more often than
behind; pure trail-only with refractory measured ~50/50 (the 8.12
symmetry result), so forward-dominance here is attributable to the
descending context.

Two measurements:
  1. TEACHER-FORCED RUNNER-UP (oracle instrument, mechanism untouched):
     drive true frames; at each L1 speak where argmax is a self row
     (home phase == now), exclude ALL self rows and record whether the
     best remaining row sits forward or backward. This is the user's
     claim as a single fraction.
  2. FREE-RUN with the refractory read at L1 (mechanism-level): does
     the loop actually advance instead of freezing?

Run: .venv/bin/python experiments/2026_08_27/refractory/run_square_refractory.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_27" / "completion"))

import numpy as np

from run_square_completion import (  # noqa: E402  (sets sys.path + env)
    EPOCHS,
    FREE_TICKS,
    K1,
    K2,
    K3,
    Layer,
    Recorder,
    deblur,
    decode_x,
    onehot,
    tick,
)
from run_temporal_square import (  # noqa: E402
    D,
    PERIOD,
    PRIME,
    TRAIN_FRAMES,
    frame_at,
    save_gif,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "refractory" / "results"
NSTR = 3


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


def upper_pass(stack, w1, learning=False):
    """L2/L3 part of a tick, identical to the base rig."""
    L1, L2, L3 = stack
    sp2 = L2.integrate(onehot(w1, K1))
    w2 = L2.learn() if (learning and sp2) else L2.read()
    if w2 >= 0:
        L1.down = L2.in_part(w2)
    if not (sp2 and w2 >= 0):
        return
    sp3 = L3.integrate(onehot(w2, K2))
    w3 = L3.learn() if (learning and sp3) else L3.read()
    if w3 >= 0:
        L2.down = L3.in_part(w3)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stack = [Layer(K1, D, K1), Layer(K2, K1, K2), Layer(K3, K2, 0)]
    rec = Recorder()
    for ep in range(EPOCHS):
        for L in stack:
            L.reset_state()
        for t in range(TRAIN_FRAMES):
            rec.phase = t % PERIOD
            tick(stack, frame_at(t)[0].ravel().astype(np.float32),
                 learning=True, rec=rec, record_homes=(ep == EPOCHS - 1))
    homes1 = rec.homes[0]

    # ---- 1. Teacher-forced runner-up: the user's claim as a fraction ------
    for L in stack:
        L.reset_state()
    ru = []
    for t in range(PRIME + FREE_TICKS):
        L1 = stack[0]
        if not L1.integrate(frame_at(t)[0].ravel().astype(np.float32)):
            continue
        C = L1.bank.W @ L1.vec()
        w1 = int(np.argmax(C))
        if t >= PRIME:
            ph = t % PERIOD
            off1 = signed_off(homes1, ph, w1)
            cand = [(C[w], signed_off(homes1, ph, w)) for w in range(K1)
                    if signed_off(homes1, ph, w) not in (None, 0)]
            if cand and off1 == 0:
                ru.append(max(cand)[1])
        upper_pass(stack, w1)
    ru = np.array(ru)
    tf_line = (f"TEACHER-FORCED best NON-SELF candidate (n={len(ru)}): "
               f"forward {(ru > 0).mean():.2f} / backward {(ru < 0).mean():.2f}"
               f", mean {ru.mean():+.2f}, median {np.median(ru):+.1f}, "
               f"|offset| mean {np.abs(ru).mean():.2f}")

    # ---- 2. Free-run with refractory at L1 --------------------------------
    for L in stack:
        L.reset_state()
    for t in range(PRIME):
        tick(stack, frame_at(t)[0].ravel().astype(np.float32), learning=False)

    gen, gen_x, true_now, offs = [], [], [], []
    emission = frame_at(PRIME - 1)[0]
    prev_w = -1
    for i in range(FREE_TICKS):
        L1 = stack[0]
        if not L1.integrate(emission.ravel().astype(np.float32)):
            continue
        C = L1.bank.W @ L1.vec()
        if prev_w >= 0:
            C[prev_w] = -np.inf                 # refractory: no repeat
        w1 = int(np.argmax(C))
        prev_w = w1
        emission = deblur(L1.in_part(w1))
        gen.append(emission)
        gen_x.append(decode_x(emission))
        true_now.append(frame_at(PRIME + i)[1])
        o = signed_off(homes1, (PRIME + i) % PERIOD, w1)
        if o is not None:
            offs.append(o)
        upper_pass(stack, w1)

    save_gif([f for f in gen for _ in range(NSTR)],
             OUTPUT_DIR / "generated.gif")
    gen_x = np.array(gen_x)
    err = np.abs(np.array(true_now) - gen_x)
    offs = np.array(offs)
    lines = [
        tf_line,
        f"FREE-RUN refractory {len(gen)} emissions: mean |x err| "
        f"{err.mean():.2f} (max {err.max()})  "
        f"(refs: arrow forms 0.00, trail-only 8.12, no-refractory 10.02)",
        f"free-run L1 offsets (n={len(offs)}): forward {(offs > 0).mean():.2f}"
        f" / self {(offs == 0).mean():.2f} / backward {(offs < 0).mean():.2f}"
        f", mean {offs.mean():+.2f}",
        "first 33 (gen_x, true_x): "
        + " ".join(f"({g},{tn})" for g, tn in zip(gen_x[:33], true_now[:33])),
    ]
    report = "\n".join(lines)
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Square, completion arrow + refractory at L1 (Arm R)\n\n"
        + report + "\n")


if __name__ == "__main__":
    main()
