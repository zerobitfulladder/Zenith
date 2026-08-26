"""Swing-up with the FAITHFUL two-track architecture (user's design).

Perception track: bank over [current angle ; fast trail ; slow trail]
-> dense relu code. Motor track: bank over [current force ; force
trail] -> dense relu code. Unification: [perception code ; motor code]
by the standard rule. Inference: angle -> up the perception track ->
query with the motor half empty -> the winner's motor-half is a code
over motor units -> REPROJECT: strongest motor unit's stored template
decodes to the force level that drives the cart.

Same demonstrator, demos, and DAgger harness as run_pole_swingup.
Prediction: if the earlier flat single-bank rig was a legitimate
collapse of this design, peak recovery lands in the same band (~58).

Run:  .venv/bin/python experiments/2026_08_26/pole_tracks/run_pole_tracks.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_angle"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_swingup"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))

import numpy as np

from run_temporal_digits import SeqDict, cn  # noqa: E402
from run_pole_ema import (  # noqa: E402
    LEVELS,
    NB_F,
    THERMO_N,
    force_code,
    physics,
)
from run_pole_swingup import (  # noqa: E402
    EP_CAP,
    EVAL_N,
    N_DEMOS,
    NB_A,
    RECOVER_TDH,
    RECOVER_TH,
    ROLLOUTS,
    ROUNDS,
    X_WIDE,
    angle_code,
    any_init,
    make_expert,
    run_episode,
    wrap,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "pole_tracks" / "results"
A_TRAIL, A_SLOW = 0.4, 0.08
GTR, GSL = 0.6, 0.5
ENC_IN = 3 * NB_A                    # [A ; trail ; slow]
MOT_IN = 2 * NB_F                    # [current force ; force trail]
KE, KM, KU = 512, 32, 512   # KE 128 -> 512: the 128-cell quantization
                            # of the percept manifold coarsened exactly
                            # the phase precision the task needs
                            # (pipeline 0.35 teacher-forced vs flat 0.92)
GM = 0.5
UDIM = KE + KM
EPOCHS = 3


class Percept:
    def __init__(self):
        self.A = np.zeros(NB_A, np.float32)
        self.Fa = np.zeros(NB_A, np.float32)
        self.Sa = np.zeros(NB_A, np.float32)
        self.Ff = np.zeros(NB_F, np.float32)

    def see(self, th):
        a = angle_code(th)
        self.A = a
        self.Fa = A_TRAIL * a + (1 - A_TRAIL) * self.Fa
        self.Sa = A_SLOW * a + (1 - A_SLOW) * self.Sa

    def did(self, level):
        f = force_code(level)
        self.Ff = A_TRAIL * f + (1 - A_TRAIL) * self.Ff

    def enc_input(self):
        return cn(np.concatenate([self.A, GTR * self.Fa, GSL * self.Sa])
                  ).astype(np.float32)

    def mot_input(self, level):
        return cn(np.concatenate([force_code(level), GTR * self.Ff])
                  ).astype(np.float32)


def code_of(bank, x):
    """Dense relu code, MEAN-CENTERED per track before the bridge —
    the standard consumer-side centering (raw relu profiles share a
    large common pedestal across states; measured: uncentered codes
    collapsed unification matching to 0.21 teacher-forced)."""
    c = np.maximum(bank.W @ x, 0.0)
    return cn(c).astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    expert = make_expert(-1)

    demos = []
    while len(demos) < N_DEMOS:
        _, ok, trace = run_episode(lambda s, t: expert(s), any_init(rng))
        if ok:
            demos.append(trace)
    print(f"demos: {len(demos)}", flush=True)

    # ---- Pretrain the two tracks (unsupervised, standard rule) -----------
    enc = SeqDict(KE, ENC_IN, 0.05, seed=1)
    mot = SeqDict(KM, MOT_IN, 0.05, seed=2)
    for _ in range(2):
        for trace in demos:
            p = Percept()
            for s, lvl in trace:
                p.see(s[2])
                enc.step(p.enc_input())
                mot.step(p.mot_input(lvl))
                p.did(lvl)
    print("tracks trained", flush=True)

    def build_z(p, lvl):
        return cn(np.concatenate(
            [code_of(enc, p.enc_input()),
             GM * code_of(mot, p.mot_input(lvl))])).astype(np.float32)

    def train_uni(zs, seed):
        uni = SeqDict(KU, UDIM, 0.05, seed=seed)
        prng = np.random.default_rng(seed + 1)
        for _ in range(EPOCHS):
            for i in prng.permutation(len(zs)):
                uni.step(zs[i])
        return uni

    def decode_action(uni, p):
        q = cn(np.concatenate([code_of(enc, p.enc_input()),
                               np.zeros(KM, np.float32)])).astype(np.float32)
        row = uni.W[int(np.argmax(uni.W @ q))]
        mh = np.maximum(row[KE:], 0.0)
        u_m = int(np.argmax(mh))
        tpl = mot.W[u_m]                       # reprojection to motor track
        f_part = tpl[:NB_F]
        return int(np.argmax(THERMO_N @ (f_part /
                                         (np.linalg.norm(f_part) + 1e-9))))

    def make_policy(uni):
        p = Percept()

        def policy(s, t):
            if t == 0:
                p.__init__()
            p.see(s[2])
            lvl = decode_action(uni, p)
            p.did(lvl)
            return lvl
        return policy

    zs = []
    for trace in demos:
        p = Percept()
        for s, lvl in trace:
            p.see(s[2])
            zs.append(build_z(p, lvl))
            p.did(lvl)
    print(f"round 0 dataset: {len(zs)}", flush=True)

    uni = None
    for rnd in range(ROUNDS):
        uni = train_uni(zs, seed=5 + rnd)
        pol = make_policy(uni)
        vrng = np.random.default_rng(100 + rnd)
        oks = sum(int(run_episode(pol, any_init(vrng))[1])
                  for _ in range(EVAL_N))
        print(f"round {rnd}: recovered {oks}/{EVAL_N} "
              f"(dataset {len(zs)})", flush=True)
        if rnd == ROUNDS - 1:
            break
        crng = np.random.default_rng(200 + rnd)
        for _ in range(ROLLOUTS):
            p = Percept()
            s = any_init(crng)
            for t in range(EP_CAP):
                p.see(s[2])
                lvl = decode_action(uni, p)
                zs.append(build_z(p, expert(s)))
                p.did(lvl)
                s = physics(s, LEVELS[lvl])
                if abs(s[0]) > X_WIDE:
                    break
                if (abs(wrap(s[2])) < RECOVER_TH
                        and abs(s[3]) < RECOVER_TDH):
                    break

    np.savez(OUTPUT_DIR / "weights.npz", We=enc.W, Wm=mot.W, Wu=uni.W)
    print("weights saved", flush=True)
    (OUTPUT_DIR / "report.md").write_text(
        "# Two-track swing-up (faithful architecture)\n\n"
        "See stdout for round-by-round; flat-rig reference peak 58/100.\n")
    print("Report written.")


if __name__ == "__main__":
    main()
