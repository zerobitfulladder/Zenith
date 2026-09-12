"""The four checks.

1  vocabulary   does the engine recover the eleven hidden atoms?
2  completion   given half the on-bits, does it produce the other half?
3  exclusion    cue F1 -- does it ever add F2?          second-order illegal
4  triangle     cue A+B -- does it ever add C?          third-order illegal
   ambiguity    cue the bits G and H share, plus a context -- which one wins?

Three arms:
  bits            pair counts over BITS                       the baseline
  patterns        pattern vocabulary, no pattern statistics
  patterns+stats  pattern vocabulary and pattern pair counts  the proposal

Plus a price sweep on the baseline, to check whether any threshold lets it both
complete and refuse.
"""

import json
import time
from pathlib import Path

import numpy as np

from engines import BitTally, PatternEngine
from world import (ATOM_NAMES, CTX1, CTX2, EXCL, F1, G, H, N_BITS, TRI,
                   atom_presence, make_world, sample, unique_bits)

HERE = Path(__file__).parent
OUT = HERE / "results"
N_TRAIN = 8000
N_TEST = 600
N_PROBE = 200
MU = 3.0
ARMS = ["bits", "patterns", "patterns+stats"]


# ---------------------------------------------------------------- helpers
def supports(eng, thresh=0.5):
    p = eng.probs()
    return [set(np.flatnonzero(p[k] > thresh).tolist()) for k in range(eng.V)]


def vocab_scores(eng, atoms):
    """Two different questions, and they come apart:

    f1        is this atom represented ALONE by some pattern?
    coverage  is this atom represented AT ALL -- possibly inside a bigger lump?
    """
    sup = supports(eng)
    best, cov = [], []
    for a in atoms:
        s = set(a.tolist())
        f1 = c = 0.0
        for t in sup:
            inter = len(s & t)
            if inter:
                f1 = max(f1, 2 * inter / (len(s) + len(t)))
                c = max(c, inter / len(s))
        best.append(f1)
        cov.append(c)
    return np.array(best), np.array(cov), sup


def pattern_kind(sup, atoms):
    kinds = {"atom": 0, "union": 0, "fragment": 0}
    for t in sup:
        hit = [len(t & set(a.tolist())) / len(a) for a in atoms]
        full = sum(1 for h in hit if h >= 0.8)
        cover = sum(len(t & set(a.tolist())) for a in atoms) / max(len(t), 1)
        if full == 1 and cover > 0.8:
            kinds["atom"] += 1
        elif full >= 2 and cover > 0.8:
            kinds["union"] += 1
        else:
            kinds["fragment"] += 1
    return kinds


def complete(arm, eng, cue, mu_read=None):
    return eng.complete(cue) if arm == "bits" else eng.complete(cue, mu=mu_read)[0]


# ---------------------------------------------------------------- checks
def check_completion(arm, eng, X, rng, mu_read=None):
    rec, prec = [], []
    for x in X:
        on = np.flatnonzero(x)
        if len(on) < 6:
            continue
        on = on.copy()
        rng.shuffle(on)
        half = len(on) // 2
        cue = np.zeros(N_BITS, dtype=bool)
        cue[on[:half]] = True
        hidden = on[half:]
        out = complete(arm, eng, cue, mu_read)
        added = np.flatnonzero(out & ~cue)
        rec.append(float(out[hidden].mean()))
        prec.append(float((x[added] == 1).mean()) if len(added) else 1.0)
    return float(np.mean(rec)), float(np.mean(prec))


def check_probes(arm, eng, atoms, rng, mu_read=None):
    excl, tri2, tri1_both, tri1_any, amb = [], [], [], [], []
    gu, hu = unique_bits(atoms, G, H), unique_bits(atoms, H, G)
    shared = np.array(sorted(set(atoms[G].tolist()) & set(atoms[H].tolist())))

    for _ in range(N_PROBE):
        # 3. exclusion -- cue F1, look for F2
        cue = np.zeros(N_BITS, dtype=bool)
        cue[atoms[F1]] = True
        excl.append(bool(atom_presence(complete(arm, eng, cue, mu_read), atoms)[EXCL[1]]))

        # 4. triangle, two given -- cue two of A,B,C, look for the third
        pair = rng.choice(TRI, size=2, replace=False)
        third = [t for t in TRI if t not in pair][0]
        cue = np.zeros(N_BITS, dtype=bool)
        for a in pair:
            cue[atoms[a]] = True
        tri2.append(bool(atom_presence(complete(arm, eng, cue, mu_read), atoms)[third]))

        # 4b. triangle, one given -- legal is exactly one of the other two
        one = int(rng.choice(TRI))
        rest = [t for t in TRI if t != one]
        cue = np.zeros(N_BITS, dtype=bool)
        cue[atoms[one]] = True
        pres = atom_presence(complete(arm, eng, cue, mu_read), atoms)
        got = sum(1 for t in rest if pres[t])
        tri1_both.append(got == 2)
        tri1_any.append(got >= 1)

        # ambiguity -- shared bits plus a context; which of G/H comes back?
        tgt, ctx = (G, CTX1) if rng.random() < 0.5 else (H, CTX2)
        cue = np.zeros(N_BITS, dtype=bool)
        cue[shared] = True
        cue[atoms[ctx]] = True
        out = complete(arm, eng, cue, mu_read)
        sg, sh = out[gu].mean(), out[hu].mean()
        want, other = (sg, sh) if tgt == G else (sh, sg)
        amb.append(1.0 if want > other else (0.5 if want == other else 0.0))

    return dict(excl_violation=float(np.mean(excl)),
                tri_violation_cue2=float(np.mean(tri2)),
                tri_violation_cue1=float(np.mean(tri1_both)),
                tri_willing_cue1=float(np.mean(tri1_any)),
                ambiguity_acc=float(np.mean(amb)))


# ---------------------------------------------------------------- driver
def build(arm, Xtr, mu, Wbits):
    if arm == "bits":
        return BitTally(N_BITS).fit(Xtr)
    return PatternEngine(N_BITS, mu=mu, use_stats=(arm == "patterns+stats")
                         ).fit(Xtr, Wbits=Wbits)


def run_seed(seed, mu=MU, arms=ARMS):
    rng = np.random.default_rng(1000 + seed)
    atoms = make_world(seed)
    Xtr, _ = sample(rng, atoms, N_TRAIN)
    Xte, _ = sample(rng, atoms, N_TEST)
    Wbits = BitTally(N_BITS).fit(Xtr).W

    out = {}
    for arm in arms:
        t0 = time.time()
        eng = build(arm, Xtr, mu, Wbits)
        if arm == "bits":
            voc = dict(best_f1=None, recovered=None, coverage=None, covered=None,
                   n_patterns=None, kinds=None)
        else:
            f1, cov, sup = vocab_scores(eng, atoms)
            voc = dict(best_f1=float(f1.mean()), recovered=int((f1 >= 0.8).sum()),
                       coverage=float(cov.mean()), covered=int((cov >= 0.8).sum()),
                       n_patterns=int(eng.V), kinds=pattern_kind(sup, atoms),
                       per_atom={ATOM_NAMES[i]: [round(float(f1[i]), 2), round(float(cov[i]), 2)]
                                 for i in range(len(atoms))},
                       supports=[sorted(t) for t in sup])
        r, p = check_completion(arm, eng, Xte, np.random.default_rng(7 + seed))
        out[arm] = dict(vocab=voc, comp_recall=r, comp_precision=p,
                        secs=round(time.time() - t0, 1),
                        **check_probes(arm, eng, atoms, np.random.default_rng(13 + seed)))
    return out


def read_price_sweep(seed=0):
    """Is there a read price at which the pattern engine SPECULATES -- adds a
    pattern on company alone -- and still refuses the illegal triple? That is
    where pattern-level statistics would have to earn their place."""
    rng = np.random.default_rng(1000 + seed)
    atoms = make_world(seed)
    Xtr, _ = sample(rng, atoms, N_TRAIN)
    Wbits = BitTally(N_BITS).fit(Xtr).W
    rows = {}
    for arm in ["patterns", "patterns+stats"]:
        eng = PatternEngine(N_BITS, mu=MU, use_stats=(arm == "patterns+stats")
                            ).fit(Xtr, Wbits=Wbits)
        rows[arm] = {}
        for mr in [3.0, 1.0, 0.0, -1.0, -2.0, -4.0, -8.0]:
            pr = check_probes(arm, eng, atoms, np.random.default_rng(13 + seed), mu_read=mr)
            rows[arm][str(mr)] = {k: pr[k] for k in
                                  ["tri_willing_cue1", "tri_violation_cue1",
                                   "tri_violation_cue2", "excl_violation"]}
    return rows


def carving_test(seed=0):
    """The decisive one. Hand the engine a vocabulary instead of letting it
    find one, and vary only WHERE IT CARVES:

      atoms   A, B, C as three separate patterns
      unions  A+B, B+C, A+C as three patterns -- the legal combinations

    The triangle is third-order over atoms and second-order over unions. If
    level-raising is what refuses the illegal triple, only `unions` can do it.
    """
    rng = np.random.default_rng(1000 + seed)
    atoms = make_world(seed)
    Xtr, _ = sample(rng, atoms, N_TRAIN)
    U = lambda *ix: np.array(sorted(set().union(*[set(atoms[i].tolist()) for i in ix])))
    vocabs = {
        "atoms": [atoms[i] for i in [0, 1, 2, 3, 4, 5, 6, 9, 10]] + [U(7), U(8)],
        "unions": [U(0, 1), U(1, 2), U(0, 2)] + [atoms[i] for i in [3, 4, 5, 6]]
                  + [U(7, 9), U(8, 10)],
    }
    rows = {}
    for name, voc in vocabs.items():
        for stats in [False, True]:
            eng = PatternEngine(N_BITS, mu=MU, use_stats=stats, max_patterns=40)
            eng.seed_vocab(voc)
            eng.fit(Xtr, mint=False)
            key = f"{name}{'+stats' if stats else ''}"
            rows[key] = {}
            for mr in [3.0, 1.0, 0.0, -2.0]:
                pr = check_probes("patterns", eng, atoms,
                                  np.random.default_rng(13 + seed), mu_read=mr)
                rows[key][str(mr)] = {k: pr[k] for k in
                                      ["tri_willing_cue1", "tri_violation_cue1",
                                       "tri_violation_cue2", "excl_violation"]}
    return rows


def price_sweep(seed=0):
    """Is there any threshold at which bit-counting both completes and refuses?"""
    rng = np.random.default_rng(1000 + seed)
    atoms = make_world(seed)
    Xtr, _ = sample(rng, atoms, N_TRAIN)
    Xte, _ = sample(rng, atoms, N_TEST)
    rows = {}
    for price in [0.0, 2.0, 4.0, 6.0, 9.0, 13.0, 18.0]:
        eng = BitTally(N_BITS, price=price).fit(Xtr)
        r, p = check_completion("bits", eng, Xte, np.random.default_rng(7 + seed))
        pr = check_probes("bits", eng, atoms, np.random.default_rng(13 + seed))
        rows[str(price)] = dict(comp_recall=r, comp_precision=p, **pr)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    seeds = list(range(5))
    res = {}
    for s in seeds:
        print(f"seed {s} ...", flush=True)
        res[str(s)] = run_seed(s)

    print("bit price sweep ...", flush=True)
    sweep = price_sweep(0)
    print("read price sweep ...", flush=True)
    rsweep = read_price_sweep(0)
    print("carving test ...", flush=True)
    carve = {str(s_): carving_test(s_) for s_ in [0, 1, 2]}

    print("mu sweep ...", flush=True)
    sens = {}
    for mu in [1.0, 2.0, 3.0, 5.0, 8.0]:
        r = run_seed(0, mu=mu, arms=["patterns+stats"])["patterns+stats"]
        sens[str(mu)] = {k: r[k] for k in
                         ["comp_recall", "comp_precision", "tri_violation_cue2",
                          "excl_violation", "ambiguity_acc"]}
        sens[str(mu)].update(best_f1=r["vocab"]["best_f1"],
                             coverage=r["vocab"]["coverage"],
                             n_patterns=r["vocab"]["n_patterns"],
                             recovered=r["vocab"]["recovered"],
                             kinds=r["vocab"]["kinds"])

    (OUT / "results.json").write_text(json.dumps(
        dict(seeds=res, bit_price_sweep=sweep, read_price_sweep=rsweep,
             carving=carve, mu_sweep=sens,
             config=dict(n_bits=N_BITS, n_train=N_TRAIN, mu=MU, seeds=seeds)),
        indent=2))

    def col(arm, k):
        return np.mean([res[str(s)][arm][k] for s in seeds])

    def vcol(arm, k):
        v = [res[str(s)][arm]["vocab"][k] for s in seeds]
        return None if v[0] is None else np.mean(v)

    print("\n" + "=" * 100)
    print(f"{'arm':<16}{'alone F1':>10}{'alone/11':>10}{'coverage':>10}{'cov/11':>8}"
          f"{'npat':>7}{'recall':>10}{'precision':>11}")
    for arm in ARMS:
        f1, rec, npat = vcol(arm, "best_f1"), vcol(arm, "recovered"), vcol(arm, "n_patterns")
        cv, cvn = vcol(arm, "coverage"), vcol(arm, "covered")
        s = (f"{'--':>10}{'--':>10}{'--':>10}{'--':>8}{'--':>7}" if f1 is None
             else f"{f1:>10.3f}{rec:>10.1f}{cv:>10.3f}{cvn:>8.1f}{npat:>7.1f}")
        print(f"{arm:<16}{s}{col(arm,'comp_recall'):>10.3f}{col(arm,'comp_precision'):>11.3f}")
    print()
    ks = ["excl_violation", "tri_violation_cue2", "tri_violation_cue1",
          "tri_willing_cue1", "ambiguity_acc"]
    print(f"{'arm':<16}" + "".join(f"{k.replace('_',' '):>20}" for k in ks))
    for arm in ARMS:
        print(f"{arm:<16}" + "".join(f"{col(arm,k):>20.3f}" for k in ks))
    print("=" * 100)
    print("\nread price: does the pattern engine ever speculate, and is it legal?")
    print(f"{'arm':<16}{'mu_read':>9}{'willing':>10}{'both B,C':>10}{'A+B->C':>9}{'F1->F2':>9}")
    for arm, rows in rsweep.items():
        for k, v in rows.items():
            print(f"{arm:<16}{float(k):>9.1f}{v['tri_willing_cue1']:>10.3f}"
                  f"{v['tri_violation_cue1']:>10.3f}{v['tri_violation_cue2']:>9.3f}"
                  f"{v['excl_violation']:>9.3f}")

    print("\nCARVING: same engine, vocabulary handed to it, 3 seeds")
    print(f"{'vocabulary':<16}{'mu_read':>9}{'willing':>10}{'both B,C':>10}{'A+B->C':>9}{'F1->F2':>9}")
    for key in ["atoms", "atoms+stats", "unions", "unions+stats"]:
        for mr in ["3.0", "1.0", "0.0", "-2.0"]:
            g = lambda k: np.mean([carve[s_][key][mr][k] for s_ in carve])
            print(f"{key:<16}{float(mr):>9.1f}{g('tri_willing_cue1'):>10.2f}"
                  f"{g('tri_violation_cue1'):>10.2f}{g('tri_violation_cue2'):>9.2f}"
                  f"{g('excl_violation'):>9.2f}")

    print("\nbit-counting price sweep (seed 0): recall vs the illegal triple")
    print(f"{'price':>7}{'recall':>10}{'precision':>11}{'tri viol':>11}{'excl viol':>11}")
    for k, v in sweep.items():
        print(f"{float(k):>7.1f}{v['comp_recall']:>10.3f}{v['comp_precision']:>11.3f}"
              f"{v['tri_violation_cue2']:>11.3f}{v['excl_violation']:>11.3f}")


if __name__ == "__main__":
    main()
