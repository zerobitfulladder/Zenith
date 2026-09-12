"""One seed, fast. Same world as ../sparse_patterns."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "sparse_patterns"))
from world import (ATOM_NAMES, N_BITS, TRI, atom_presence,  # noqa: E402
                   make_world, sample)

from engine import Hashed  # noqa: E402

SEED, N_TRAIN, N_PROBE = 0, 2000, 100


def name(eng, slot, atoms):
    """What a slot means, read by expanding it down to lights."""
    v = np.zeros(eng.N)
    v[slot] = 1.0
    L = eng.expand(v)
    hit = [ATOM_NAMES[i] for i, a in enumerate(atoms) if L[a].mean() >= 0.66]
    return "{" + ",".join(hit) + "}" if hit else f"?{L.sum():.0f}lights"


def probes(eng, atoms, rng, thresh):
    gu = np.array(sorted(set(atoms[7].tolist()) - set(atoms[8].tolist())))
    hu = np.array(sorted(set(atoms[8].tolist()) - set(atoms[7].tolist())))
    shared = np.array(sorted(set(atoms[7].tolist()) & set(atoms[8].tolist())))
    spec, both, cue2, excl, amb = [], [], [], [], []
    for _ in range(N_PROBE):
        one = int(rng.choice(TRI))
        rest = [t for t in TRI if t != one]
        cue = np.zeros(N_BITS, dtype=bool)
        cue[atoms[one]] = True
        pres = atom_presence(eng.complete(cue, thresh)[0], atoms)
        got = sum(1 for t in rest if pres[t])
        spec.append(got >= 1)
        both.append(got == 2)

        pair = rng.choice(TRI, size=2, replace=False)
        third = [t for t in TRI if t not in pair][0]
        cue = np.zeros(N_BITS, dtype=bool)
        for a in pair:
            cue[atoms[a]] = True
        cue2.append(bool(atom_presence(eng.complete(cue, thresh)[0], atoms)[third]))

        cue = np.zeros(N_BITS, dtype=bool)
        cue[atoms[3]] = True
        excl.append(bool(atom_presence(eng.complete(cue, thresh)[0], atoms)[4]))

        tgt, ctx = (7, 9) if rng.random() < 0.5 else (8, 10)
        cue = np.zeros(N_BITS, dtype=bool)
        cue[shared] = True
        cue[atoms[ctx]] = True
        out = eng.complete(cue, thresh)[0]
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
    eng = Hashed(N_BITS).fit(X)
    print(f"trained on {N_TRAIN} vectors in {time.time()-t0:.1f}s")
    used = [s for s in eng.meaning if eng.slot[s] > 1.0]
    print(f"{len(eng.meaning)} friendship slots addressed, {len(used)} of them "
          f"with real weight; {eng.NH} available")

    # what the strongest slots at each height turn out to mean
    cue = np.zeros(N_BITS, dtype=bool)
    cue[atoms[0]] = True
    cue[atoms[1]] = True
    out, levels, guesses = eng.complete(cue, thresh=1.0)
    print("\ncue = A's and B's lights")
    for r, v in enumerate(levels):
        act = np.flatnonzero(v > eng.floor)
        top = act[np.argsort(-v[act])][:4]
        print(f"  round {r}: {len(act):>3} slots active   strongest: "
              + ", ".join(f"{name(eng,int(s),atoms)}@{v[s]:.2f}" for s in top))
    print("  guessed:", [name(eng, g, atoms) for g in guesses] or None,
          "-> atoms out:",
          [ATOM_NAMES[i] for i, b in enumerate(atom_presence(out, atoms)) if b])

    print(f"\n{'thresh':>7}{'speculates':>12}{'both B,C':>10}{'A+B->C':>9}"
          f"{'F1->F2':>9}{'ambiguity':>11}")
    for th in [4.0, 2.0, 1.0, 0.0]:
        r = probes(eng, atoms, np.random.default_rng(13), th)
        print(f"{th:>7.1f}{r['speculates']:>12.2f}{r['both_BC']:>10.2f}"
              f"{r['cue2_addsC']:>9.2f}{r['F1_addsF2']:>9.2f}{r['ambiguity']:>11.2f}")


if __name__ == "__main__":
    main()
