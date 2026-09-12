"""Does a second floor mint the union vocabulary by itself?

`run.py` had to hand-build A+B, B+C, A+C to make the illegal triple refusable.
This asks whether an identical engine, run on floor 0's own output, produces
that vocabulary unprompted -- and whether the two-floor read then inherits the
behaviour.

Two configurations, because they separate two different failures:

  learned   floor 0 mints its own vocabulary (fragments and all)
  clean     floor 0 is handed the eleven atoms; only floor 1 has to learn

The read price for floor 0 is held at 2.5 throughout, which is the one clean
operating point: at 2.5 a single floor is MUTE, so any speculation the stack
does is coming from upstairs and not from floor 0 lowering its own standards.
"""

import json
from pathlib import Path

import numpy as np

from engines import BitTally, PatternEngine
from run import check_probes
from stack import Stack
from world import ATOM_NAMES, N_BITS, make_world, sample

HERE = Path(__file__).parent
N_TRAIN = 8000
MU0_READ = 2.5
MU1_READ = 1.0
SEEDS = [0, 1, 2, 3, 4]
CLEAN_ORDER = ["A", "B", "C", "F1", "F2", "F3", "F4", "CTX1", "CTX2", "G", "H"]


def build(seed, clean):
    rng = np.random.default_rng(1000 + seed)
    atoms = make_world(seed)
    X, _ = sample(rng, atoms, N_TRAIN)
    U = lambda *ix: np.array(sorted(set().union(*[set(atoms[i].tolist()) for i in ix])))

    st = Stack(N_BITS)
    if clean:
        st.L0 = PatternEngine(N_BITS, mu=3.0, use_stats=True, max_patterns=40)
        st.L0.seed_vocab([atoms[i] for i in [0, 1, 2, 3, 4, 5, 6, 9, 10]]
                         + [U(7), U(8)])
        st.L0.fit(X, mint=False)
        names = CLEAN_ORDER
    else:
        st.L0.fit(X, Wbits=BitTally(N_BITS).fit(X).W)
        p0 = st.L0.probs()
        names = []
        for k in range(st.L0.V):
            t = set(np.flatnonzero(p0[k] > 0.5).tolist())
            hit = [ATOM_NAMES[i] for i, a in enumerate(atoms)
                   if len(t & set(a.tolist())) / len(a) >= 0.8]
            names.append("+".join(hit) if hit else f"?{k}")
    st.L0._mint = False

    Z = st._lift(X)
    st.L1 = PatternEngine(st.L0.V, mu=1.0, use_stats=True, max_patterns=40,
                          mint_min=2, mint_max=4, tau=0.6, min_seed=0.5)
    st.L1.fit(Z, Wbits=BitTally(st.L0.V).fit(Z).W)
    st.L1._mint = False
    return st, atoms, names


class TwoFloors:
    """Adapter so check_probes can drive the stack like any other engine."""

    def __init__(self, st):
        self.st = st

    def complete(self, cue, mu=None):
        return self.st.complete(cue, mu=MU0_READ, mu_top=MU1_READ)


def upper_names(st, names):
    return ["{" + ",".join(names[k] for k in g) + "}" for g in st.upper_vocab()]


def main():
    KEYS = ["tri_willing_cue1", "tri_violation_cue1", "tri_violation_cue2",
            "excl_violation", "ambiguity_acc"]
    res = {}
    for clean in [False, True]:
        tag = "clean floor 0" if clean else "learned floor 0"
        res[tag] = {}
        for s in SEEDS:
            st, atoms, names = build(s, clean)
            one = check_probes("patterns", st.L0, atoms,
                               np.random.default_rng(13 + s), mu_read=MU0_READ)
            two = check_probes("patterns", TwoFloors(st), atoms,
                               np.random.default_rng(13 + s), mu_read=None)
            voc = upper_names(st, names)
            unions = sum(1 for v in voc if v in ("{A,B}", "{B,A}", "{A,C}",
                                                 "{C,A}", "{B,C}", "{C,B}"))
            res[tag][str(s)] = dict(one_floor={k: one[k] for k in KEYS},
                                    two_floor={k: two[k] for k in KEYS},
                                    floor0_patterns=int(st.L0.V),
                                    floor1_vocab=voc, unions_found=unions)
            print(f"{tag} seed {s}: unions {unions}/3  {voc}", flush=True)

    (HERE / "results" / "stack.json").write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 96)
    print(f"floor-0 read price {MU0_READ} (a single floor is MUTE here), "
          f"floor-1 price {MU1_READ}, {len(SEEDS)} seeds")
    hdr = f"{'configuration':<18}{'floors':>8}{'unions':>8}{'speculates':>12}" \
          f"{'both B,C':>10}{'A+B->C':>9}{'F1->F2':>9}{'ambiguity':>11}"
    print(hdr)
    for tag, rows in res.items():
        for which in ["one_floor", "two_floor"]:
            g = lambda k: np.mean([rows[str(s)][which][k] for s in SEEDS])
            u = np.mean([rows[str(s)]["unions_found"] for s in SEEDS])
            print(f"{tag:<18}{'one' if which=='one_floor' else 'two':>8}"
                  f"{(u if which=='two_floor' else float('nan')):>8.1f}"
                  f"{g('tri_willing_cue1'):>12.2f}{g('tri_violation_cue1'):>10.2f}"
                  f"{g('tri_violation_cue2'):>9.2f}{g('excl_violation'):>9.2f}"
                  f"{g('ambiguity_acc'):>11.2f}")
    print("=" * 96)

    print("\nper seed, clean floor 0 -- the failure is a MINTING failure, not a read failure")
    for s in SEEDS:
        r = res["clean floor 0"][str(s)]
        print(f"  seed {s}: unions {r['unions_found']}/3  "
              f"spec {r['two_floor']['tri_willing_cue1']:.2f}  "
              f"illegal {r['two_floor']['tri_violation_cue1']:.2f}  {r['floor1_vocab']}")


if __name__ == "__main__":
    main()
