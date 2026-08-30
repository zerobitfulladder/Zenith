"""Train the two-track drone memory from the PID oracle, then fly it.

  1. roll out episodes under fast_oracle.teacher, recording at every tick
     what was sensed (dx, dy, angle) and what was commanded (two thrusts)
  2. train the sensory hypercolumn on the sensed codes and the motor
     hypercolumn on the commanded codes
  3. train layer two on the two codes concatenated -- one template is one
     whole (situation, thrusts) pair
  4. fly with the motor half of every query left empty

Run:  .venv/bin/python experiments/2026_08_30/td_tracks/run_td_tracks.py
Then: VIEW_CKPT=experiments/2026_08_30/td_tracks/results/two_track.npz \
      .venv/bin/python viewer.py
"""

import json, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_28" / "single_layer_drone"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_29" / "temporal_drone"))

import sl_drone as W                                   # noqa: E402
import fast_oracle as F                                # noqa: E402
import two_track as T                                  # noqa: E402

OUT = HERE / "results"
K_SENS, K_MOT, K_TOP, TOPM, RHO = 1024, 169, 4096, 8, 1.0
EPISODES, EPOCHS, SEED = 900, 3, 0


def rollout(n, seed):
    """Fly the oracle; record (sensed, commanded) at every tick."""
    rng = np.random.default_rng(seed)
    S, M = [], []
    for _ in tqdm(range(n), desc="oracle rollouts", ncols=80):
        s, tgt = W.any_init(rng)
        hold = 0
        for _ in range(W.EP_CAP):
            lv = F.teacher(s, tgt)
            S.append(T.sense(s, tgt)); M.append(lv)
            s = W.physics(s, lv)
            hold = hold + 1 if W.at_goal(s, tgt) else 0
            if hold >= 10 or abs(s[0]) > 30 or abs(s[1]) > 30:
                break
    return S, M


def bench(fn, n=300, seed=7):
    """(success, ticks to within 0.5, total ticks) -- fast_oracle's measure."""
    rng = np.random.default_rng(seed)
    ok, done, near = [], [], []
    for _ in range(n):
        s, tgt = W.any_init(rng)
        hold, fin, tn = 0, None, None
        for t in range(W.EP_CAP):
            s = W.physics(s, fn(s, tgt))
            if tn is None and np.hypot(s[0] - tgt[0], s[1] - tgt[1]) < 0.5:
                tn = t
            if W.at_goal(s, tgt):
                hold += 1
                if hold >= 10:
                    fin = t; break
            else:
                hold = 0
            if abs(s[0]) > 30 or abs(s[1]) > 30:
                break
        ok.append(fin is not None)
        if fin is not None:
            done.append(fin); near.append(tn if tn is not None else fin)
    return (float(np.mean(ok)),
            float(np.median(near)) if near else float("nan"),
            float(np.median(done)) if done else float("nan"))


def trace(fn, seed, cap=600):
    rng = np.random.default_rng(seed)
    s, tgt = W.any_init(rng)
    path, cmds, hold = [], [], 0
    for _ in range(cap):
        lv = fn(s, tgt)
        path.append(s[:2].copy()); cmds.append(lv)
        s = W.physics(s, lv)
        hold = hold + 1 if W.at_goal(s, tgt) else 0
        if hold >= 10:
            break
    return np.array(path), np.array(cmds), tgt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cache = Path("/tmp/claude-1000/-home-lavender-Projects-Geodesique/"
                 "b740704d-75db-4cdd-bdff-0ecb9063c424/scratchpad/rollout.npz")
    if cache.exists():
        z = np.load(cache)
        S = [dict(zip(T.SENS_CH, r)) for r in z["S"]]
        M = [tuple(r) for r in z["M"]]
        print(f"reusing cached oracle rollout ({len(S):,} ticks)")
    else:
        S, M = rollout(EPISODES, SEED)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache,
                            S=np.array([[d[c] for c in T.SENS_CH] for d in S]),
                            M=np.array(M))
    print(f"{len(S):,} ticks from {EPISODES} oracle episodes\n")

    m = T.TwoTrack(K_SENS, K_MOT, K_TOP, TOPM, rho=RHO, seed=SEED)
    rng = np.random.default_rng(SEED + 10)

    for ep in range(EPOCHS):                      # the two tracks, separately
        for i in tqdm(rng.permutation(len(S)), desc=f"tracks  epoch {ep+1}",
                      ncols=80):
            m.l1s.learn(*m.sens_track.encode(S[i]))
            m.l1m.learn(*m.mot_track.encode(T.motor(M[i])))
    print(f"sensory track {m.l1s.n_boot} templates, "
          f"motor track {m.l1m.n_boot}")

    codes = [(m.sens_code(S[i]), m.mot_code(M[i])) for i in range(len(S))]
    for ep in range(EPOCHS):                      # the bound memory on top
        for i in tqdm(rng.permutation(len(S)), desc=f"layer 2 epoch {ep+1}",
                      ncols=80):
            m.l2.learn(*m.join(*codes[i]))
    print(f"layer two {m.l2.n_boot} templates, "
          f"{int((m.l2.wins > 0).sum())} of them ever won again\n")

    rows = [("PID oracle (the teacher)", bench(F.teacher))]
    for read in ("graded", "top1"):
        rows.append((f"two-track memory, {read} read",
                     bench(lambda s, t, r=read: m.act(s, t, read=r))))
    rows.append(("hover, no control", bench(
        lambda s, t: (int(np.argmin(np.abs(W.LEVELS - W.HOVER))),) * 2)))
    print(f"  {'':34} {'success':>8} {'to target':>10} {'total':>7}")
    for n_, (o, nr, tt) in rows:
        print(f"  {n_:<34} {o:>8.2f} {nr:>10.0f} {tt:>7.0f}")

    # ---- pictures -------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (name, fn) in zip(axes, [
            ("PID oracle", F.teacher),
            ("two-track memory (graded)", lambda s, t: m.act(s, t)),
            ("two-track memory (top-1)", lambda s, t: m.act(s, t, read="top1"))]):
        for sd in range(6):
            p, _, tgt = trace(fn, 100 + sd)
            ax.plot(p[:, 0], p[:, 1], lw=1.3, alpha=.85)
            ax.plot(*tgt, "*", ms=13, color="0.25")
            ax.plot(p[0, 0], p[0, 1], "o", ms=4, color="0.55")
        ax.set_title(name, fontsize=11); ax.set_aspect("equal")
        ax.grid(alpha=.25, lw=.5); ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    fig.suptitle("six episodes from the same starts — grey dot start, star target",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "flights.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
    _, co, _ = trace(F.teacher, 100)
    _, cm, _ = trace(lambda s, t: m.act(s, t), 100)
    for ax, (name, c) in zip(axes, [("PID oracle", co), ("two-track memory", cm)]):
        ax.plot(W.LEVELS[c[:, 0]], lw=1.2, label="left thrust")
        ax.plot(W.LEVELS[c[:, 1]], lw=1.2, label="right thrust")
        ax.axhline(W.HOVER, ls="--", lw=.8, color="0.5")
        ax.set_ylabel("thrust"); ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.25, lw=.5)
    axes[-1].set_xlabel("tick")
    fig.suptitle("what comes out of the motor half that was never written",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "commands.png", dpi=130); plt.close(fig)

    m.save(OUT / "two_track.npz")
    json.dump({"kind": "module", "module": "two_track",
               "dir": "experiments/2026_08_30/td_tracks",
               "k_sens": K_SENS, "k_mot": K_MOT, "k_top": K_TOP,
               "topm": TOPM, "rho": RHO, "seed": SEED, "read": "graded"},
              open(OUT / "two_track.json", "w"), indent=2)
    json.dump({"config": {"k_sens": K_SENS, "k_mot": K_MOT, "k_top": K_TOP,
                          "topm": TOPM, "rho": RHO, "episodes": EPISODES,
                          "epochs": EPOCHS, "ticks": len(S)},
               "bench": {n_: {"success": o, "to_target": nr, "total": tt}
                         for n_, (o, nr, tt) in rows},
               "seconds": round(time.time() - t0, 1)},
              open(OUT / "metrics.json", "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")
    print(f"\nviewer:\n  VIEW_CKPT=experiments/2026_08_30/td_tracks/"
          f"results/two_track.npz .venv/bin/python viewer.py")


if __name__ == "__main__":
    main()
