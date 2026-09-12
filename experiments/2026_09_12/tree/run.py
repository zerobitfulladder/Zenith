"""Two questions.

1  the probes -- does a top-down category tree hold the constraints the
   bottom-up engines needed a stored rule for?

2  the scaling -- does it pay the PRODUCT of independent choices? A world with
   `f` independent blocks, each with 4 legal options, has 4^f legal scenes. If
   the tree needs one leaf per scene, leaves should quadruple with each block
   while a pairwise notebook would only add a handful of counts.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "sparse_patterns"))
from world import (ATOM_NAMES, N_BITS, TRI, atom_presence,  # noqa: E402
                   make_world, sample)

from tree import CatTree  # noqa: E402

SEED, N_TRAIN, N_PROBE = 0, 4000, 200


def probes(t, atoms, rng):
    gu = np.array(sorted(set(atoms[7].tolist()) - set(atoms[8].tolist())))
    hu = np.array(sorted(set(atoms[8].tolist()) - set(atoms[7].tolist())))
    shared = np.array(sorted(set(atoms[7].tolist()) & set(atoms[8].tolist())))
    spec, both, cue2, excl, amb = [], [], [], [], []
    for _ in range(N_PROBE):
        one = int(rng.choice(TRI))
        rest = [x for x in TRI if x != one]
        cue = np.zeros(N_BITS, dtype=bool)
        cue[atoms[one]] = True
        pres = atom_presence(t.complete(cue)[0], atoms)
        got = sum(1 for x in rest if pres[x])
        spec.append(got >= 1)
        both.append(got == 2)

        pair = rng.choice(TRI, size=2, replace=False)
        third = [x for x in TRI if x not in pair][0]
        cue = np.zeros(N_BITS, dtype=bool)
        for a in pair:
            cue[atoms[a]] = True
        cue2.append(bool(atom_presence(t.complete(cue)[0], atoms)[third]))

        cue = np.zeros(N_BITS, dtype=bool)
        cue[atoms[3]] = True
        excl.append(bool(atom_presence(t.complete(cue)[0], atoms)[4]))

        tgt, ctx = (7, 9) if rng.random() < 0.5 else (8, 10)
        cue = np.zeros(N_BITS, dtype=bool)
        cue[shared] = True
        cue[atoms[ctx]] = True
        out = t.complete(cue)[0]
        sg, sh = out[gu].mean(), out[hu].mean()
        want, other = (sg, sh) if tgt == 7 else (sh, sg)
        amb.append(1.0 if want > other else (0.5 if want == other else 0.0))
    return dict(speculates=np.mean(spec), both_BC=np.mean(both),
                cue2_addsC=np.mean(cue2), F1_addsF2=np.mean(excl),
                ambiguity=np.mean(amb))


def free_world(rng, n_free, n, block=6, noise=2, drop=0.05):
    """`n_free` independent blocks. Each block has 4 sub-groups and exactly one
    of them fires. Nothing constrains one block given another, so the legal
    scenes number 4^n_free."""
    n_bits = n_free * 4 * block + 20
    X = np.zeros((n, n_bits), dtype=np.int8)
    for r in range(n):
        for f in range(n_free):
            k = rng.integers(0, 4)
            s = (f * 4 + k) * block
            bits = np.arange(s, s + block)
            X[r, bits[rng.random(block) > drop]] = 1
        X[r, rng.integers(0, n_bits, size=noise)] = 1
    return X


def main():
    rng = np.random.default_rng(1000 + SEED)
    atoms = make_world(SEED)
    X, _ = sample(rng, atoms, N_TRAIN)

    t0 = time.time()
    t = CatTree().fit(X)
    lv = t.leaves()
    print(f"trained on {N_TRAIN} vectors in {time.time()-t0:.1f}s")
    print(f"{len(t.nodes)} nodes, {len(lv)} leaves "
          f"(45 legal scenes in this world: 3 triangle x 5 filler x 3 ambiguity)")

    # what do the leaves mean?
    print("\nfirst eight leaves, read by which atoms they predict:")
    for i in lv[:8]:
        p = t.probs(i)
        hit = [ATOM_NAMES[k] for k, a in enumerate(atoms) if p[a].mean() > 0.5]
        print(f"  leaf {i:>3} n={t.nodes[i]['n']:>4}  {{{','.join(hit)}}}")

    # does any leaf contain the illegal triple?
    bad = 0
    for i in lv:
        p = t.probs(i)
        if all(p[atoms[k]].mean() > 0.5 for k in TRI):
            bad += 1
    bad_excl = sum(1 for i in lv
                   if t.probs(i)[atoms[3]].mean() > 0.5
                   and t.probs(i)[atoms[4]].mean() > 0.5)
    print(f"\nleaves containing A,B,C all three: {bad}   "
          f"leaves containing F1 and F2: {bad_excl}")

    r = probes(t, atoms, np.random.default_rng(13))
    print(f"\n{'':<14}{'speculates':>12}{'both B,C':>10}{'A+B->C':>9}"
          f"{'F1->F2':>9}{'ambiguity':>11}")
    print(f"{'tree':<14}{r['speculates']:>12.2f}{r['both_BC']:>10.2f}"
          f"{r['cue2_addsC']:>9.2f}{r['F1_addsF2']:>9.2f}{r['ambiguity']:>11.2f}")
    print(f"{'recursive':<14}{1.00:>12.2f}{0.00:>10.2f}{0.00:>9.2f}"
          f"{0.00:>9.2f}{1.00:>11.2f}     (../recursive, beam 4)")

    # ---- the cost of independence ---------------------------------------
    print("\nscaling: f independent blocks, 4 options each -> 4^f legal scenes")
    print(f"{'blocks':>7}{'lights':>8}{'legal scenes':>14}{'leaves':>9}"
          f"{'pair counts a notebook would need':>36}")
    for f in [1, 2, 3, 4]:
        Xf = free_world(np.random.default_rng(7), f, 6000)
        tf = CatTree(min_size=20).fit(Xf)
        nb = f * 4 * 6
        print(f"{f:>7}{Xf.shape[1]:>8}{4**f:>14}{len(tf.leaves()):>9}"
              f"{nb*(nb-1)//2:>36}")


if __name__ == "__main__":
    main()
