"""Reinforcement, the user's rule: rotate the winner toward the moment
when things are improving, away when they are worsening.

No teacher after the bootstrap. The hypercolumn is seeded by watching
the oracle fly (so the minicolumns start from sane states), and then the
oracle is switched OFF entirely and the only learning signal is the
outcome:

    third factor  = (progress - running baseline) / running scale
    progress      = potential(now) - potential(before)
    potential     = -(distance + 0.3*speed + 0.5*|tilt|)   [+ arrival]

Every minicolumn that won recently carries a decaying eligibility tag,
so when the signal arrives the credit reaches the action that SET UP the
outcome, not just the last one (the tilt-before-you-move problem).
Repulsion doubles as exploration: a minicolumn pushed away stops winning
there, so a different one answers next time.

Run:  .venv/bin/python experiments/2026_08_28/single_layer_drone/run_sl_rl.py
Env:  RL_TICKS RL_WARM RL_REPORT RL_K RL_ETA RL_EPS RL_TRACE RL_GAMMA
      RL_GPU RL_TAG
"""

import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sl_drone as W                       # noqa: E402
import sl_viz as V                         # noqa: E402

OUTPUT_DIR = HERE / "results" / f"rl{os.environ.get('RL_TAG', '')}"
TICKS = int(os.environ.get("RL_TICKS", "400000"))
WARM = int(os.environ.get("RL_WARM", "40000"))
# Phase 2: the pupil flies but the oracle still labels its OWN states.
# Without this, switching the teacher off leaves pure behaviour cloning
# from oracle-only states, which fails on covariate shift from the very
# first episode (measured: 100% of RL episodes failed immediately) — RL
# then has nothing to improve. The pole needed exactly this to go from
# 3/100 to 87/100.
DAGGER = int(os.environ.get("RL_DAGGER", "40000"))
FREEZE = os.environ.get("RL_FREEZE", "0") == "1"    # control: no nudges
# Fraction of episodes the ORACLE flies outright, forever. Without it
# the stream is whatever the pupil visits, which is emergencies, so
# the memory fills with emergency responses and starts issuing them in
# calm states too (measured: commanded collective drifts 6.5 -> 8.0 =
# full power, |L-R| 0.54 -> 1.8, success 0.91 -> 0.00). Real DAgger
# keeps every demonstration in its dataset forever; a continual memory
# has to keep re-seeing them.
ORACLE_EP = float(os.environ.get("RL_ORACLE_EP", "0.3"))
REPORT = int(os.environ.get("RL_REPORT", "10000"))
K = int(os.environ.get("RL_K", "32768"))
SIZE = int(os.environ.get("RL_SIZE", "8192"))
ETA = float(os.environ.get("RL_ETA", "0.05"))
EPS = float(os.environ.get("RL_EPS", "0.05"))       # exploration rate
TRACE = int(os.environ.get("RL_TRACE", "8"))        # eligibility depth
GAMMA = float(os.environ.get("RL_GAMMA", "0.65"))   # tag decay
GPU = os.environ.get("RL_GPU", "1") == "1"
# Consolidation threshold is in WINS PER TEMPLATE. With tens of
# thousands of minicolumns each wins only a handful of times in a
# whole run, so a threshold of 300 (my earlier value) never engaged.
CONSOL = float(os.environ.get("RL_CONSOL", "0"))
SCALE_A = 0.001                                      # signal-scale EMA
# Living cost is charged in ADVANTAGE units (a fraction of the signal's
# own natural size), so it can never dominate: as an absolute quantity
# it was ~70x typical progress and punished every tick (measured:
# frac_rewarded 0.10, adv -0.7, dictionary dissolved).
DEAD = float(os.environ.get("RL_DEAD", "0.25"))      # learn only on news
BASE_A = 0.0005                                      # baseline EMA rate
# Repulsion is destructive where attraction is merely corrective: a
# repelled minicolumn abandons a state it legitimately covered. Our own
# LVQ result found repel-at-half-rate is what makes attract safe, and
# with a mostly-negative signal full-rate repulsion collapsed the
# policy inside a few hundred ticks.
REPEL = float(os.environ.get("RL_REPEL", "0.3"))
# NOTE: no running baseline. A baseline that drifts negative makes
# "no change" score positive, so hovering in place is rewarded and the
# drone learns never to arrive (measured: frac_rewarded 0.59 -> 0.67
# while arrivals fell 0.74 -> 0.06). Progress is already symmetric
# about zero — closing is positive, receding is negative — so the
# honest third factor is progress itself, scaled, minus a living cost
# that makes standing still cost something.


def potential(s, tgt):
    # Distance to the target and nothing else — the user's rule,
    # literally. Charging for speed and tilt (my earlier version)
    # punishes the tilt-and-accelerate that CLOSING requires, so the
    # signal was negative on ~100% of ticks and every template got
    # repelled.
    return -np.hypot(s[0] - tgt[0], s[1] - tgt[1])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = W.Encoder(SIZE, seed=0)
    hc = W.Hypercolumn(K, SIZE, eta=ETA, seed=1, gpu=GPU,
                       consol=CONSOL)
    rng = np.random.default_rng(0)

    def save(tag, extra):
        np.savez(OUTPUT_DIR / f"{tag}.npz", Wt=hc.weights_numpy(),
                 n_boot=hc.n_boot, wins=hc.wins)
        with open(OUTPUT_DIR / f"{tag}.json", "w") as f:
            json.dump({"size": SIZE, "k": K, "nb": W.NB, "enc_seed": 0,
                       **extra}, f)

    s, tgt = W.any_init(rng)
    oracle_ep = True
    lv_prev = (W.NLEV // 2, W.NLEV // 2)
    pot = potential(s, tgt)
    trace = deque(maxlen=TRACE)
    base, scale = 0.0, 0.02
    ep_tick = hold = episodes = 0
    recent, advs, progs, pos_frac, sim = [], [], [], [], []
    acts = []
    stats, best = [], {"score": -1.0, "tick": 0}

    for t in range(TICKS):
        sens = W.sense(s, tgt, lv_prev)

        if t < WARM:
            # phase 1: watch the oracle, learn the bound moment
            lv_a = W.teacher(s, tgt)
            hc.learn(*enc.encode_sparse(sens, lv_a))
        elif t < WARM + DAGGER:
            # phase 2: pupil flies, oracle labels the pupil's own states —
            # except on rehearsal episodes, which the oracle flies
            if oracle_ep:
                lv_a = W.teacher(s, tgt)
                hc.learn(*enc.encode_sparse(sens, lv_a))
                qi = None
            else:
                qi, qv = enc.encode_sparse(sens, None)
                i, c = hc.compete(qi, qv)
                lv_a = (enc.read_motors(hc.row(i)) if i >= 0
                        else (W.NLEV // 2, W.NLEV // 2))
                if i >= 0:
                    sim.append(c)
                hc.learn(*enc.encode_sparse(sens, W.teacher(s, tgt)))
        else:
            # act from memory: sensory bits in, motor bits empty
            qi, qv = enc.encode_sparse(sens, None)
            i, c = hc.compete(qi, qv)
            if i < 0:
                lv_a = (W.NLEV // 2, W.NLEV // 2)
            else:
                lv_a = enc.read_motors(hc.row(i))
                sim.append(c)
            if rng.random() < EPS:                  # explore
                lv_a = (int(np.clip(lv_a[0] + rng.integers(-2, 3),
                                    0, W.NLEV - 1)),
                        int(np.clip(lv_a[1] + rng.integers(-2, 3),
                                    0, W.NLEV - 1)))
            # Credit goes to the minicolumn that CAUSED the action (the
            # one the state-only query retrieved), and it is nudged
            # toward/away from the moment as it actually was — the state
            # OR'ed with the action actually taken.
            ai, av = enc.encode_sparse(sens, lv_a)
            trace.appendleft((i, ai, av))

        acts.append(lv_a)
        s = W.physics(s, lv_a)
        lv_prev = lv_a
        ep_tick += 1
        if t < WARM + DAGGER:              # keep the potential current
            pot = potential(s, tgt)        # through both bootstrap phases

        done = False
        arrived = False
        if W.at_goal(s, tgt):
            hold += 1
            if hold >= 10:
                arrived = done = True
        else:
            hold = 0
        lost = abs(s[0]) > 30 or abs(s[1]) > 30 or ep_tick >= W.EP_CAP
        if lost:
            done = True

        if t >= WARM + DAGGER:
            new_pot = potential(s, tgt)
            prog = (new_pot - pot) \
                + (2.0 if arrived else 0.0) - (1.0 if lost else 0.0)
            pot = new_pot
            # Advantage against the pupil's OWN recent progress: "better
            # than I have been doing lately". With a pure-distance
            # potential this cannot reward loitering — when the pupil is
            # flying well the baseline is positive, so standing still
            # scores negative (the earlier exploit came from the speed
            # and tilt terms, which dragged the baseline below zero).
            base += BASE_A * (prog - base)
            scale += SCALE_A * (abs(prog - base) - scale)
            adv = float(np.clip((prog - base) / (scale + 1e-6), -2.0, 2.0))
            if adv < 0:
                adv *= REPEL
            advs.append(adv)
            progs.append(prog)
            pos_frac.append(1.0 if adv > 0 else 0.0)
            # third factor reaches every recently-active minicolumn,
            # weights normalised so total plasticity per tick is one
            # step regardless of trace depth
            if abs(adv) > DEAD and not FREEZE:
                wts = np.array([GAMMA ** k for k in range(len(trace))])
                wts /= wts.sum()
                for k, (wi, ai, av) in enumerate(trace):
                    hc.nudge(wi, ai, av, adv * float(wts[k]))

        if done:
            episodes += 1
            if not oracle_ep:
                recent.append(1 if arrived else 0)
            recent = recent[-100:]
            s, tgt = W.any_init(rng)
            oracle_ep = rng.random() < ORACLE_EP
            lv_prev = (W.NLEV // 2, W.NLEV // 2)
            pot = potential(s, tgt)
            trace.clear()
            ep_tick = hold = 0

        if (t + 1) % REPORT == 0:
            row = {"tick": t + 1,
                   "phase": ("1:oracle-flies" if t < WARM else
                             "2:DAgger" if t < WARM + DAGGER else
                             ("3:FROZEN-control" if FREEZE else "3:RL")),
                   "episodes": episodes,
                   "success_last100": round(float(np.mean(recent))
                                            if recent else 0.0, 3),
                   "mean_adv": round(float(np.mean(advs)), 3)
                   if advs else None,
                   "prog_mean": round(float(np.mean(progs)), 5)
                   if progs else None,
                   "prog_absmean": round(float(np.mean(np.abs(progs))), 5)
                   if progs else None,
                   "frac_rewarded": round(float(np.mean(pos_frac)), 3)
                   if pos_frac else None,
                   "bestmatch": round(float(np.mean(sim)), 3) if sim else None,
                   "minicolumns_used": int(hc.n_boot),
                   "act_spread": round(float(np.std(np.array(acts))), 2),
                   "mean_diff": round(float(np.mean(
                       [abs(a[0] - a[1]) for a in acts])), 2),
                   "mean_coll": round(float(np.mean(
                       [(a[0] + a[1]) / 2 for a in acts])), 2)}
            stats.append(row)
            print(json.dumps(row), flush=True)
            advs, progs, pos_frac, sim, acts = [], [], [], [], []
            save("checkpoint", {"tick": t + 1,
                                "score": row["success_last100"]})
            if t >= WARM + DAGGER and row["success_last100"] > best["score"]:
                best = {"score": row["success_last100"], "tick": t + 1}
                save("weights_best", best)
                print(f"  new best {best['score']} @ {best['tick']}",
                      flush=True)
            V.save_templates(hc, enc,
                             OUTPUT_DIR / f"templates_{t + 1:07d}.png", t + 1)
            with open(OUTPUT_DIR / "metrics.json", "w") as f:
                json.dump({"best": best, "config": {
                    "k": K, "eta": ETA, "eps": EPS, "trace": TRACE,
                    "gamma": GAMMA, "warm": WARM}, "reports": stats},
                    f, indent=2)

    save("weights_final", {"tick": TICKS,
                           "score": stats[-1]["success_last100"]})
    print(f"done — best {best['score']} @ tick {best['tick']}", flush=True)


if __name__ == "__main__":
    main()
