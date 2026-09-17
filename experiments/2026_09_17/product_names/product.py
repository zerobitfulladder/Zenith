"""Product names: a card's fingerprint is its parts walked together.   python product.py

    name(card) = ⊗_i ( part_i ⊗ offset_i )       commutative, associative, order-free

Tests, in the exact toy setting (random primitives, random offsets):
  1. order:      integrating the same parts in different orders gives the same name
  2. subtract:   name ⊗ (observed part at offset)^-1 = the rest; does it clean up to the
                 right remaining part(s) among a vocabulary of distractors?
  3. depth:      how many parts can be subtracted in a row before the residual stops
                 cleaning up (it should never stop: binding is exact; the limit is only in
                 what the cleanup vocabulary contains)
  4. hierarchy:  whole = sub1 ⊗ sub2 (each a product); subtracting a whole sub-card yields
                 the other sub-card's name directly, without expanding to leaves
  5. similarity: wholes sharing all but one part overlap only where the differing parts do
                 (the price), versus a bundle of the same parts (graded)
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "2026_09_16" / "glimpse_loop"))   # sdr.py
import sdr

alg = sdr.Algebra(16, 41, 0)
rng = np.random.default_rng(0)
P = 60                                   # primitive vocabulary
prims = np.stack([alg.random() for _ in range(P)])
poses = sdr.Poses(alg, 12, 12)


def name(parts):
    """parts: list of (prim id, (dr, dc)) -> product fingerprint."""
    out = np.zeros(alg.B, int)
    for p, off in parts:
        out = alg.bind(out, alg.bind(prims[p], poses(off)))
    return out


def rand_card(k):
    ids = rng.choice(P, k, replace=False)
    offs = [(int(rng.integers(0, 4)), int(rng.integers(0, 4))) for _ in range(k)]
    return list(zip(ids.tolist(), offs))


print("1. ORDER")
card = rand_card(5)
names = [name(list(np.array(card, dtype=object)[perm])) for perm in [rng.permutation(5) for _ in range(20)]]
print(f"   20 integration orders of the same 5 parts -> {len({tuple(n) for n in names})} distinct name(s)\n")

print("2. SUBTRACT one part, clean up the rest against every (prim, offset) pair in a 4x4 window")
vocab_pairs = [(p, (r, c)) for p in range(P) for r in range(4) for c in range(4)]
vocab_codes = np.stack([alg.bind(prims[p], poses(off)) for p, off in vocab_pairs])
ok = 0; trials = 300
for _ in range(trials):
    a, b = rand_card(2)
    W = name([a, b])
    rest = alg.bind(W, alg.inverse(alg.bind(prims[a[0]], poses(a[1]))))
    i, ov = alg.cleanup(rest, vocab_codes)
    ok += vocab_pairs[i] == b and ov == alg.B
print(f"   2-part cards: the residual after subtracting one part cleans up to the other part and its offset: {ok/trials:.0%}  ({len(vocab_pairs):,} candidates)\n")

print("3. DEPTH: subtract parts one at a time from a k-part card; is the final residual the last part?")
for k in [3, 5, 8, 12, 20]:
    ok = 0
    for _ in range(100):
        card = rand_card(k)
        W = name(card)
        for p, off in card[:-1]:
            W = alg.bind(W, alg.inverse(alg.bind(prims[p], poses(off))))
        i, ov = alg.cleanup(W, vocab_codes)
        ok += vocab_pairs[i] == card[-1] and ov == alg.B
    print(f"   k={k:2d}: {ok:3d}% exact")
print("   (exact every time: binding loses nothing; the only limit is what the cleanup table knows)\n")

print("4. HIERARCHY: whole = sub1 ⊗ sub2, a k-leaf sub-card placed at offset o is walked by o k times")
def place(N, k, off):
    return alg.bind(N, alg.power(poses(off), k))
ok = 0
for _ in range(200):
    s1, s2 = rand_card(3), rand_card(3)
    o1, o2 = (0, 0), (2, 1)
    N1, N2 = name(s1), name(s2)
    whole = alg.bind(place(N1, 3, o1), place(N2, 3, o2))
    # also equal to the product over all six leaves at their absolute offsets?
    leaves = [(p, (r + o1[0], c + o1[1])) for p, (r, c) in s1] + [(p, (r + o2[0], c + o2[1])) for p, (r, c) in s2]
    same = np.array_equal(whole, name(leaves))
    rest = alg.bind(whole, alg.inverse(place(N1, 3, o1)))
    sub_codes = np.stack([place(N2, 3, o2), place(N1, 3, o1)] + [place(alg.random(), 3, o2) for _ in range(500)])
    i, ov = alg.cleanup(rest, sub_codes)
    ok += same and i == 0 and ov == alg.B
print(f"   whole built from sub-card names equals whole built from leaves, and subtracting sub1 yields sub2's name: {ok/200:.0%}")
# translating a whole of K leaves by t = walking by t K times; 41 is prime so K*t is a distinct shift for K < 41
ok = 0
for _ in range(200):
    card = rand_card(6); t = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
    moved = name([(p, (r + t[0], c + t[1])) for p, (r, c) in card])
    ok += np.array_equal(moved, place(name(card), 6, t))
print(f"   moving a 6-leaf whole by t equals walking its name by t six times: {ok/200:.0%}\n")

print("5. SIMILARITY: two wholes sharing 4 of 5 parts")
ovp, ovb, ovr = [], [], []
for _ in range(300):
    card = rand_card(5)
    other = card[:4] + [(int(rng.integers(P)), card[4][1])]
    ovp.append(alg.overlap(name(card), name(other)))
    Bc = alg.empty(); Bo = alg.empty()
    for p, off in card: alg.add(Bc, alg.bind(prims[p], poses(off)))
    for p, off in other: alg.add(Bo, alg.bind(prims[p], poses(off)))
    ovb.append(int(((Bc > 0) & (Bo > 0)).sum()))
    ovr.append(alg.overlap(name(card), name(rand_card(5))))
print(f"   product names: overlap {np.mean(ovp):4.2f} of 16 blocks  (two unrelated names: {np.mean(ovr):4.2f})")
print(f"   bundles of the same parts: {np.mean(ovb):4.1f} shared lit slots of 80  -> graded: 4 of 5 parts shared shows")
print("   the price of product names: sharing parts is invisible to overlap; recognition must go through the residual")
