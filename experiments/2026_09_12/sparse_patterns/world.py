"""The world: a sparse binary vector with known hidden structure, and no geometry.

100 bits. Eleven hidden atoms of 6 bits each, scattered at random through the
vector -- no adjacency, no image, no convolution. A sample is a union of some
atoms plus noise. Three pieces of structure are built in, chosen so that each
one tests a different claim:

  triangle    A, B, C      present as a PAIR or not at all, never all three.
              Each pair is POSITIVELY correlated (they arrive together), so a
              pair count over bits is actively tempted to add the third.
              -> third-order illegality.

  exclusion   F1, F2       never together.
              -> second-order illegality, which pair counts should catch.

  ambiguity   G, H         share four of their six bits. G arrives with CTX1,
              CTX2         H arrives with CTX2. The shared bits alone cannot
              CTX1         say which. Only the company it keeps can.
              -> the case where pattern-level statistics have to break a tie.
"""

import numpy as np

N_BITS = 100
ATOM_SIZE = 6
SHARED = 4               # bits G and H have in common

A, B, C = 0, 1, 2
F1, F2, F3, F4 = 3, 4, 5, 6
G, H = 7, 8
CTX1, CTX2 = 9, 10
ATOM_NAMES = ["A", "B", "C", "F1", "F2", "F3", "F4", "G", "H", "CTX1", "CTX2"]
TRI = [A, B, C]
EXCL = (F1, F2)
FILLERS = [F1, F2, F3, F4]

P_TRI = 0.25             # how often the triangle block appears at all
P_AMB = 0.50             # how often the G/H block appears at all
P_DROP = 0.05            # an atom bit fails to fire
N_ADD = 2                # spurious bits per sample


def make_world(seed):
    """Atom -> bit indices. G and H overlap; everything else is disjoint."""
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(N_BITS))
    atoms = [None] * len(ATOM_NAMES)
    take = lambda n: np.array([perm.pop() for _ in range(n)])

    for a in [A, B, C, F1, F2, F3, F4, CTX1, CTX2]:
        atoms[a] = take(ATOM_SIZE)
    shared = take(SHARED)
    atoms[G] = np.concatenate([shared, take(ATOM_SIZE - SHARED)])
    atoms[H] = np.concatenate([shared, take(ATOM_SIZE - SHARED)])
    return atoms


def draw_atoms(rng):
    picked = []
    if rng.random() < P_TRI:                       # two of A,B,C -- never three
        picked += list(rng.choice(TRI, size=2, replace=False))
    while True:                                    # two fillers -- never F1 & F2
        fil = rng.choice(FILLERS, size=2, replace=False)
        if not (F1 in fil and F2 in fil):
            break
    picked += list(fil)
    if rng.random() < P_AMB:                       # G with CTX1, or H with CTX2
        picked += [G, CTX1] if rng.random() < 0.5 else [H, CTX2]
    return np.array(picked)


def sample(rng, atoms, n):
    X = np.zeros((n, N_BITS), dtype=np.int8)
    used = []
    for r in range(n):
        picked = draw_atoms(rng)
        for a in picked:
            bits = atoms[a]
            keep = rng.random(len(bits)) > P_DROP
            X[r, bits[keep]] = 1
        if N_ADD:
            X[r, rng.integers(0, N_BITS, size=N_ADD)] = 1
        used.append(picked)
    return X, used


def atom_presence(x, atoms, thresh=0.75):
    return np.array([x[a].mean() >= thresh for a in atoms])


def unique_bits(atoms, a, other):
    """Bits of atom `a` that atom `other` does not have -- for telling G from H."""
    return np.array(sorted(set(atoms[a].tolist()) - set(atoms[other].tolist())))
