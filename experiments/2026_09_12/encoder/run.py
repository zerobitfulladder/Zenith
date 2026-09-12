"""One seed, no reruns. Same world as ../sparse_patterns."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "sparse_patterns"))
from world import (ATOM_NAMES, N_BITS, TRI, atom_presence,  # noqa: E402
                   make_world, sample)

from engine import Encoder, EncoderTally  # noqa: E402

SEED, N_TRAIN, N_PROBE = 0, 5000, 100


def label(eng, t, E, atoms):
    """What a template means, read by expanding its signature to lights."""
    v = np.zeros(eng.N, bool)
    v[eng.sigT[t]] = True
    L = eng.decode(eng.expand(v, E))
    hit = [ATOM_NAMES[i] for i, a in enumerate(atoms) if L[a].mean() >= 0.8]
    on_lights = E[t][eng.sigL].sum()
    on_temps = E[t][eng.sigT[:eng.V]].sum()
    lvl = "L" if on_lights >= on_temps else "T"
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

    which = sys.argv[1] if len(sys.argv) > 1 else "tally"
    t0 = time.time()
    eng = (EncoderTally if which == "tally" else Encoder)(N_BITS).fit(X.astype(bool))
    print(f"[{which}] trained on {N_TRAIN} vectors in {time.time()-t0:.1f}s   "
          f"({eng.M} templates over {eng.N} slots, {eng.k} per signature)")
    E, nE = eng.expect()
    names = [label(eng, t, E, atoms) for t in range(eng.V)]
    print(f"{eng.V} templates hired; {sum(n.endswith('T') for n in names)} "
          f"attend mostly to other templates")
    used = np.argsort(-eng.use[:eng.V])
    print("most used:", [f"{names[t]}x{int(eng.use[t])}" for t in used[:16]])
    atoms_alone = sorted({n[:-1] for n in names
                          if n.startswith("{") and "," not in n})
    print(f"atoms represented alone: {len(atoms_alone)}/11  {atoms_alone}")

    # upper templates: how many template-signatures does each expect? A clean
    # pair like {T_A, T_B} is 2; a whole-scene blob is 4 or 5.
    upper = [t for t in range(eng.V) if names[t].endswith("T")]
    sizes = [int((E[t][eng.sigT[:eng.V]].sum(1) >= 0.6 * eng.k).sum()) for t in upper]
    print("upper templates by size in signatures:",
          {s_: sizes.count(s_) for s_ in sorted(set(sizes))})
    pairs = [names[t] for t, s_ in zip(upper, sizes) if s_ == 2]
    print("the clean pairs:", pairs[:20])

    cue = np.zeros(N_BITS, bool)
    cue[atoms[0]] = True
    out, seqs, p = eng.complete(cue)
    print("\ncue = A's lights")
    for r, s in enumerate(seqs):
        print(f"  round {r}: used {[names[t] for t in s]}")
    print(f"  proposed: {None if p is None else names[p]}   -> atoms out:",
          [ATOM_NAMES[i] for i, b in enumerate(atom_presence(out, atoms)) if b])

    r = probes(eng, atoms, np.random.default_rng(13))
    print(f"\n{'':<12}{'speculates':>12}{'both B,C':>10}{'A+B->C':>9}"
          f"{'F1->F2':>9}{'ambiguity':>11}")
    print(f"{which:<12}{r['speculates']:>12.2f}{r['both_BC']:>10.2f}"
          f"{r['cue2_addsC']:>9.2f}{r['F1_addsF2']:>9.2f}{r['ambiguity']:>11.2f}")
    print(f"{'recursive':<12}{1.00:>12.2f}{0.00:>10.2f}{0.00:>9.2f}"
          f"{0.00:>9.2f}{1.00:>11.2f}   (../recursive, beam 4, has a notebook)")


if __name__ == "__main__":
    main()
