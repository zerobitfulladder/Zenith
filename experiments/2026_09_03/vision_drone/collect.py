"""Collect flights from the PID teacher, then render what the drone saw.

flight.npz: states (N,6) = x y vx vy tilt tilt-rate, targets (N,2), levels
(N,2) the teacher's two thrust levels, ep (N,), tick (N,).
frames.npy: (N,48,48) uint8, rendered from the stored states.

Usage:  uv run python collect.py [--episodes 400]
"""
import sys, time
import numpy as np
from pathlib import Path
import box_world as B

HERE = Path(__file__).resolve().parent
EPISODES = int(sys.argv[sys.argv.index("--episodes") + 1]) if "--episodes" in sys.argv else 400


def main():
    rng = np.random.default_rng(0); pid = B.make_pid()
    S, T, L, E, K = [], [], [], [], []
    ok, lengths = 0, []
    t0 = time.time()
    for e in range(EPISODES):
        s, tgt, lv, _ = B.episode(rng, pid) if False else B.episode_no_frames(rng, pid)
        S.append(s); T.append(np.repeat(tgt[None], len(s), 0)); L.append(lv)
        E.append(np.full(len(s), e)); K.append(np.arange(len(s)))
        lengths.append(len(s)); ok += len(s) < B.W.EP_CAP
    S, T, L, E, K = map(np.concatenate, (S, T, L, E, K))
    np.savez(HERE / "results" / "flight.npz", states=S, targets=T, levels=L, ep=E, tick=K)
    F = np.stack([B.render(s, t) for s, t in zip(S, T)])
    np.save(HERE / "results" / "frames.npy", (F * 255).astype(np.uint8))
    hist = np.bincount(L.reshape(-1), minlength=B.W.NLEV)
    print(f"{EPISODES} episodes, {len(S)} ticks, success {ok / EPISODES:.2f}, "
          f"median {int(np.median(lengths))} ticks, {time.time() - t0:.0f}s")
    print("level histogram (both motors):", hist.tolist())
    print(f"frames {F.shape}, mean intensity {F.mean():.3f}")


if __name__ == "__main__":
    main()
