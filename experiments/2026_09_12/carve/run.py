"""One seed, no reruns. Same world as ../sparse_patterns."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "sparse_patterns"))
from world import (ATOM_NAMES, N_BITS, TRI, atom_presence,  # noqa: E402
                   make_world, sample)

from engine import Carve  # noqa: E402

SEED, N_TRAIN, N_PROBE = 0, 5000, 100


def label(eng, t, atoms):
    v = np.zeros(eng.nT, bool)
    v[eng.NL + t] = True
    L = eng.expand(v)
    hit = [ATOM_NAMES[i] for i, a in enumerate(atoms) if L[a].mean() >= 0.8]
    lvl = "L" if eng.members[t][0] < eng.NL else "T"
    return ("{" + ",".join(hit) + "}" if hit else f"?{L.sum()}") + lvl


def probes(eng, atoms, rng):
    gu = np.array(sorted(set(atoms[7].tolist()) - set(atoms[8].tolist())))
    hu = np.array(sorted(set(atoms[8].tolist()) - set(atoms[7].tolist())))
    shared = np.array(sorted(set(atoms[7].tolist()) & set(atoms[8].tolist())))
    spec, both, cue2, excl, amb = [], [], [], [], []
    for _ in range(N_PROBE):
        one = int(rng.choice(TRI))
        rest = [x for x in TRI if x != one]
        cue = np.zeros(N_BITS, bool)
        cue[atoms[one]] = True
        pres = atom_presence(eng.complete(cue)[0], atoms)
        got = sum(1 for x in rest if pres[x])
        spec.append(got >= 1)
        both.append(got == 2)

        pair = rng.choice(TRI, size=2, replace=False)
        third = [x for x in TRI if x not in pair][0]
        cue = np.zeros(N_BITS, bool)
        for a in pair:
            cue[atoms[a]] = True
        cue2.append(bool(atom_presence(eng.complete(cue)[0], atoms)[third]))

        cue = np.zeros(N_BITS, bool)
        cue[atoms[3]] = True
        excl.append(bool(atom_presence(eng.complete(cue)[0], atoms)[4]))

        tgt, ctx = (7, 9) if rng.random() < 0.5 else (8, 10)
        cue = np.zeros(N_BITS, bool)
        cue[shared] = True
        cue[atoms[ctx]] = True
        out = eng.complete(cue)[0]
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

    t0 = time.time()
    eng = Carve(N_BITS).fit(X.astype(bool))
    print(f"trained on {N_TRAIN} vectors in {time.time()-t0:.1f}s")
    names = [label(eng, t, atoms) for t in range(eng.V)]
    lower = [n for n in names if n.endswith("L")]
    upper = [n for n in names if n.endswith("T")]
    print(f"{eng.V} templates carved: {len(lower)} from lights, {len(upper)} from templates")
    print("from lights   :", lower)
    print("from templates:", upper)
    atoms_alone = sorted({n[:-1] for n in lower if n.startswith("{") and "," not in n})
    print(f"atoms represented alone: {len(atoms_alone)}/11  {atoms_alone}")
    sizes = [len(eng.members[t]) for t in range(eng.V) if names[t].endswith("T")]
    print("upper templates by size:", {s: sizes.count(s) for s in sorted(set(sizes))})

    for a_ in [0, 1, 2]:
        cue = np.zeros(N_BITS, bool)
        cue[atoms[a_]] = True
        out, seqs, p = eng.complete(cue)
        print(f"\ncue = {ATOM_NAMES[a_]}'s lights")
        for r, s in enumerate(seqs):
            print(f"  round {r}: used {[names[t] for t in s]}")
        print(f"  proposed: {None if p is None else names[p]}   -> atoms out:",
              [ATOM_NAMES[i] for i, b in enumerate(atom_presence(out, atoms)) if b])

    r = probes(eng, atoms, np.random.default_rng(13))
    print(f"\n{'':<12}{'speculates':>12}{'both B,C':>10}{'A+B->C':>9}"
          f"{'F1->F2':>9}{'ambiguity':>11}")
    print(f"{'carve':<12}{r['speculates']:>12.2f}{r['both_BC']:>10.2f}"
          f"{r['cue2_addsC']:>9.2f}{r['F1_addsF2']:>9.2f}{r['ambiguity']:>11.2f}")
    print(f"{'encoder+tally':<12}{0.61:>12.2f}{0.00:>10.2f}{0.00:>9.2f}"
          f"{0.00:>9.2f}{0.71:>11.2f}   (../encoder, notebook + warm-up)")
    print(f"{'recursive':<12}{1.00:>12.2f}{0.00:>10.2f}{0.00:>9.2f}"
          f"{0.00:>9.2f}{1.00:>11.2f}   (../recursive, beam 4)")


if __name__ == "__main__":
    main()
