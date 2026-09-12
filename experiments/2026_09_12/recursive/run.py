"""One seed, plain. Does the single-notebook recursive version work, and does
beam search beat greedy?

Same world as ../sparse_patterns (100 lights, eleven hidden groups of 6, the
A/B/C triangle that is never all three, F1/F2 never together).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "sparse_patterns"))
from world import (ATOM_NAMES, N_BITS, TRI, atom_presence,  # noqa: E402
                   make_world, sample)

from engine import Recursive  # noqa: E402

SEED = 0
N_TRAIN = 4000
N_PROBE = 100


def label(eng, k, atoms):
    """Name a pattern by what it turns into once expanded down to lights."""
    v = np.zeros(eng.N, dtype=np.int8)
    v[eng.name_of(k)] = 1
    lights = eng.expand(v)
    hit = [ATOM_NAMES[i] for i, a in enumerate(atoms) if lights[a].mean() >= 0.8]
    kids = [s for s in np.flatnonzero(eng.probs()[k] > 0.5)]
    depth = "lights" if all(s < eng.NL for s in kids) else "patterns"
    return ("{" + ",".join(hit) + "}" if hit else f"?{k}") + f"[{depth}]"


def probes(eng, atoms, rng, mu):
    gu = np.array(sorted(set(atoms[7].tolist()) - set(atoms[8].tolist())))
    hu = np.array(sorted(set(atoms[8].tolist()) - set(atoms[7].tolist())))
    shared = np.array(sorted(set(atoms[7].tolist()) & set(atoms[8].tolist())))
    spec, both, cue2, excl, amb = [], [], [], [], []
    for _ in range(N_PROBE):
        cue = np.zeros(N_BITS, dtype=bool)
        one = int(rng.choice(TRI))
        rest = [t for t in TRI if t != one]
        cue[atoms[one]] = True
        pres = atom_presence(eng.complete(cue, mu=mu)[0], atoms)
        got = sum(1 for t in rest if pres[t])
        spec.append(got >= 1)
        both.append(got == 2)

        pair = rng.choice(TRI, size=2, replace=False)
        third = [t for t in TRI if t not in pair][0]
        cue = np.zeros(N_BITS, dtype=bool)
        for a in pair:
            cue[atoms[a]] = True
        cue2.append(bool(atom_presence(eng.complete(cue, mu=mu)[0], atoms)[third]))

        cue = np.zeros(N_BITS, dtype=bool)
        cue[atoms[3]] = True
        excl.append(bool(atom_presence(eng.complete(cue, mu=mu)[0], atoms)[4]))

        tgt, ctx = (7, 9) if rng.random() < 0.5 else (8, 10)
        cue = np.zeros(N_BITS, dtype=bool)
        cue[shared] = True
        cue[atoms[ctx]] = True
        out = eng.complete(cue, mu=mu)[0]
        sg, sh = out[gu].mean(), out[hu].mean()
        want, other = (sg, sh) if tgt == 7 else (sh, sg)
        amb.append(1.0 if want > other else (0.5 if want == other else 0.0))
    return dict(speculates=np.mean(spec), both_BC=np.mean(both),
                cue2_addsC=np.mean(cue2), F1_addsF2=np.mean(excl),
                ambiguity=np.mean(amb))


def main():
    rng = np.random.default_rng(1000 + SEED)
    atoms = make_world(SEED)
    X, _ = sample(rng, atoms, N_TRAIN)

    for beam in [1, 4]:
        eng = Recursive(N_BITS, beam=beam).fit(X)
        tag = "greedy" if beam == 1 else f"beam {beam}"
        names = [label(eng, k, atoms) for k in range(eng.V)]
        deep = [n for n in names if "[patterns]" in n]
        print(f"\n===== {tag} =====")
        print(f"{eng.V} patterns; {len(deep)} of them are made of other patterns")
        print("  built from lights  :", [n for n in names if "[lights]" in n][:14])
        print("  built from patterns:", deep[:14])

        # one worked example, all the way through
        cue = np.zeros(N_BITS, dtype=bool)
        cue[atoms[0]] = True
        out, top, trace, guess = eng.complete(cue, mu=2.0)
        print("  cue = A's six lights")
        for r, (n, S) in enumerate(trace):
            print(f"    round {r+1}: {n} slots active -> used "
                  f"{[label(eng,k,atoms) for k in S]}")
        print("    proposed:", None if guess is None else label(eng, guess, atoms))
        print("    settled to:", [label(eng, k - eng.NL, atoms)
                                  for k in np.flatnonzero(top) if k >= eng.NL],
              "| expands to atoms:",
              [ATOM_NAMES[i] for i, v_ in enumerate(atom_presence(out, atoms)) if v_])

        print(f"  {'mu':>5}{'speculates':>12}{'both B,C':>10}{'A+B->C':>9}"
              f"{'F1->F2':>9}{'ambiguity':>11}")
        for mu in [3.0, 2.0, 1.0]:
            r = probes(eng, atoms, np.random.default_rng(13), mu)
            print(f"  {mu:>5.1f}{r['speculates']:>12.2f}{r['both_BC']:>10.2f}"
                  f"{r['cue2_addsC']:>9.2f}{r['F1_addsF2']:>9.2f}"
                  f"{r['ambiguity']:>11.2f}")


if __name__ == "__main__":
    main()
