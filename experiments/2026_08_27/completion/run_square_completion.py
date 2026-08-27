"""Bouncing square, COMPLETION ARROW (user's design, 2026-08-27).

Uniform unit at every layer. NO next slots, NO lag-advance cargo
channels, NO empty query channels anywhere. Each layer:

  - leakily integrates every arrival from below (T = x + GAMMA*T,
    never reset within a stream),
  - speaks (learns + sends a one-hot winner upward) every NSTR-th
    arrival,
  - stores plain snapshots [own trail ; GD * top-down projection],
    where the projection is the one refreshed at the PREVIOUS tick
    (lagged feedback, learned jointly with the input),
  - the layer above refreshes the downward projection at EVERY
    arrival (reads continuous / event-driven, speech sparse): its
    retrieved row's input-trail segment, relu'd, is handed down.

Claimed arrow (the user's proposal): a mid-era trail is a HALF
pattern to the layer above, whose stored era snapshot was written
LATER and so contains the era's future; retrieval completes the half
to the whole; the projection tints the lower query with that future,
recursively down to L1, whose retrieved row (deblurred) is the
emission. No explicit time-offset channel anywhere — the arrow, if it
exists, comes from storage-time-vs-query-time offset plus completion.

Instrument: for every free-run retrieval at every layer, the signed
circular phase offset between the retrieved row's home phases (the
phases at which it learned, final epoch) and the current intended
phase. forward fraction >> backward fraction = the arrow points ahead.

References on this task: lag-advance / next-slot forms = 0.00 mean
|x| error; trail-only (no arrow) = 8.12.

Run: .venv/bin/python experiments/2026_08_27/completion/run_square_completion.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_square"))

import numpy as np

from run_gpu_minibatch import Dict  # noqa: E402
from run_temporal_square import (  # noqa: E402
    D,
    H,
    PERIOD,
    PRIME,
    SQ,
    TRAIN_FRAMES,
    W,
    frame_at,
    save_gif,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_27" / "completion" / "results"
K1, K2, K3 = 64, 32, 16
GAMMA = 0.5                 # same leak at every layer (per arrival)
NSTR = 3                    # every layer speaks every 3rd arrival
GD = 0.5                    # top-down channel gain
ETA = 0.05
EPOCHS = 3
FREE_TICKS = 300            # input-rate ticks of free run (~100 emissions)


def cn1(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return (v / n if n > 1e-9 else v).astype(np.float32)


class Layer:
    """The uniform unit: leaky trail + snapshot memory + strided speech."""

    def __init__(self, k, in_dim, down_dim):
        self.k, self.in_dim, self.down_dim = k, in_dim, down_dim
        self.bank = Dict(k, in_dim + down_dim, ETA)
        self.reset_state()

    def reset_state(self):
        self.T = np.zeros(self.in_dim, np.float32)
        self.down = np.zeros(self.down_dim, np.float32)
        self.arrivals = 0

    def integrate(self, x):
        self.T = x + GAMMA * self.T
        self.arrivals += 1
        return self.arrivals % NSTR == 0        # is this a speak tick?

    def vec(self):
        parts = [cn1(self.T)]
        if self.down_dim:
            parts.append(GD * cn1(self.down))   # per-pathway cn, then concat
        return cn1(np.concatenate(parts))

    def learn(self):
        z = self.vec()[None, :]
        if self.bank.n_boot < self.k:
            self.bank.bootstrap(z)
            return -1
        C = (z @ self.bank.W.T)[0]
        w = int(np.argmax(C))
        self.bank.update(z, np.array([w]),
                         np.array([max(C[w], 0.0)], dtype=np.float32))
        return w

    def read(self):
        if self.bank.n_boot < self.k:
            return -1
        return int(np.argmax(self.bank.W @ self.vec()))

    def in_part(self, w):
        return np.maximum(self.bank.W[w, : self.in_dim], 0.0)


def onehot(w, k):
    v = np.zeros(k, np.float32)
    v[w] = 1.0
    return v


class Recorder:
    """Home phases (learning) and retrieval offsets (free run)."""

    def __init__(self):
        self.homes = [dict(), dict(), dict()]
        self.offsets = [[], [], []]
        self.phase = 0

    def home(self, li, w):
        self.homes[li].setdefault(w, set()).add(self.phase)

    def offset(self, li, w):
        homes = self.homes[li].get(w)
        if not homes:
            return
        best = None
        for hp in homes:
            d = ((hp - self.phase + PERIOD // 2) % PERIOD) - PERIOD // 2
            if best is None or abs(d) < abs(best):
                best = d
        self.offsets[li].append(best)


def tick(stack, x, learning, rec=None, record_homes=False):
    """One input-rate tick. Returns L1's winner (or -1 if L1 silent)."""
    L1, L2, L3 = stack
    if not L1.integrate(x):
        return -1
    w1 = L1.learn() if learning else L1.read()
    if w1 < 0:
        return w1
    if rec and record_homes:
        rec.home(0, w1)
    if rec and not learning:
        rec.offset(0, w1)

    sp2 = L2.integrate(onehot(w1, K1))
    w2 = L2.learn() if (learning and sp2) else L2.read()
    if w2 >= 0:
        L1.down = L2.in_part(w2)                # consumed NEXT L1 tick (lag)
        if rec and record_homes and sp2:
            rec.home(1, w2)
        if rec and not learning:
            rec.offset(1, w2)
    if not (sp2 and w2 >= 0):
        return w1

    sp3 = L3.integrate(onehot(w2, K2))
    w3 = L3.learn() if (learning and sp3) else L3.read()
    if w3 >= 0:
        L2.down = L3.in_part(w3)                # consumed NEXT L2 tick (lag)
        if rec and record_homes and sp3:
            rec.home(2, w3)
        if rec and not learning:
            rec.offset(2, w3)
    return w1


def deblur(pred):
    """Brightest SQ*SQ pixels of a retrieved frame-trail -> binary frame."""
    f = np.zeros(D, np.float32)
    idx = np.argsort(pred)[-(SQ * SQ):]
    idx = idx[pred[idx] > 1e-6]
    f[idx] = 1.0
    return f.reshape(H, W)


def decode_x(frame):
    col = frame.sum(axis=0)
    return int(np.argmax(np.convolve(col, np.ones(SQ), "valid")))


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

    used = [len(rec.homes[li]) for li in range(3)]
    print(f"rows used in final epoch: L1 {used[0]}/{K1}, "
          f"L2 {used[1]}/{K2}, L3 {used[2]}/{K3}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_DIR / "weights.npz", W1=stack[0].bank.W,
             W2=stack[1].bank.W, W3=stack[2].bank.W)
    import json
    (OUTPUT_DIR / "homes.json").write_text(json.dumps(
        [{str(w): sorted(ph) for w, ph in rec.homes[li].items()}
         for li in range(3)]))

    # ---- Teacher-forced diagnostic: true frames, no self-feedback ---------
    for L in stack:
        L.reset_state()
    tf = Recorder()
    tf.homes = rec.homes
    tf_lead = []
    for t in range(PRIME + FREE_TICKS):
        tf.phase = t % PERIOD
        w1 = tick(stack, frame_at(t)[0].ravel().astype(np.float32),
                  learning=False, rec=(tf if t >= PRIME else None))
        if w1 >= 0 and t >= PRIME:
            # lead of the would-be emission over the present, in frames
            gx = decode_x(deblur(stack[0].in_part(w1)))
            x_now, x_nxt = frame_at(t)[1], frame_at(t + 1)[1]
            tf_lead.append((gx - x_now) * (1 if x_nxt > x_now else -1))
    tf_lines = ["", "TEACHER-FORCED (true frames driven, on-distribution):"]
    for li, name in enumerate(["L1", "L2", "L3"]):
        off = np.array(tf.offsets[li])
        if len(off):
            tf_lines.append(
                f"  {name} offsets (n={len(off)}): "
                f"forward {(off > 0).mean():.2f} / self {(off == 0).mean():.2f}"
                f" / backward {(off < 0).mean():.2f}, mean {off.mean():+.2f}")
    lead = np.array(tf_lead)
    tf_lines.append(
        f"  emission lead vs present (signed, + = ahead in motion dir): "
        f"mean {lead.mean():+.2f}, ahead {(lead > 0).mean():.2f} / "
        f"at-present {(lead == 0).mean():.2f} / behind {(lead < 0).mean():.2f}")

    # ---- Prime with true frames (reads only), then free-run ---------------
    for L in stack:
        L.reset_state()
    for t in range(PRIME):
        tick(stack, frame_at(t)[0].ravel().astype(np.float32), learning=False)

    gen, gen_x, true_now, true_next = [], [], [], []
    emission = frame_at(PRIME - 1)[0]            # held until first speak
    for i in range(FREE_TICKS):
        rec.phase = (PRIME + i) % PERIOD
        w1 = tick(stack, emission.ravel().astype(np.float32),
                  learning=False, rec=rec)
        if w1 >= 0:
            emission = deblur(stack[0].in_part(w1))
            gen.append(emission)
            gen_x.append(decode_x(emission))
            true_now.append(frame_at(PRIME + i)[1])
            true_next.append(frame_at(PRIME + i + 1)[1])

    save_gif([f for f in gen for _ in range(NSTR)],
             OUTPUT_DIR / "generated.gif")

    gen_x = np.array(gen_x)
    err_now = np.abs(np.array(true_now) - gen_x)
    err_next = np.abs(np.array(true_next) - gen_x)
    lines = [
        f"free-run {len(gen)} emissions over {FREE_TICKS} ticks: "
        f"mean |x err| vs t {err_now.mean():.2f} (max {err_now.max()}), "
        f"vs t+1 {err_next.mean():.2f}  "
        f"(refs: arrow forms 0.00, trail-only 8.12)",
    ]
    for li, name in enumerate(["L1", "L2", "L3"]):
        off = np.array(rec.offsets[li])
        if len(off) == 0:
            lines.append(f"{name}: no retrievals recorded")
            continue
        fwd = float((off > 0).mean())
        back = float((off < 0).mean())
        self_ = float((off == 0).mean())
        lines.append(
            f"{name} retrieval offsets (n={len(off)}): "
            f"forward {fwd:.2f} / self {self_:.2f} / backward {back:.2f}, "
            f"mean {off.mean():+.2f}, median {np.median(off):+.1f}")
    lines += tf_lines
    lines.append("")
    lines.append("first 33 free-run (gen_x, true_x): "
                 + " ".join(f"({g},{tn})" for g, tn
                            in zip(gen_x[:33], true_now[:33])))
    report = "\n".join(lines)
    print(report, flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Square, completion arrow (no next, no lag channels)\n\n"
        + report + "\n")


if __name__ == "__main__":
    main()
