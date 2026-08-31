"""Forgetting experiment: repulsion, but experts earn resistance to it.

Everything is the counts and protocol of ../continual_fixed: 40 hypercolumns
x 36 minicolumns, pre-allocated, split MNIST (phase 1 = digits 0-4, phase 2 =
5-9), 6 epochs per phase, batch 128, same seed, same fit gate at test.

The only change is HOW the learner is chosen and corrected:

    winner   = argmin over hypercolumns of
                 image error + LAM*(1 - confidence in the true label)
                 - GAMMA*(1/H - win_fraction)          <- conscience
    correct  = the winner learns the joined vector, AND whoever would win the
               unlabelled fit gate and is wrong is rotated away from it
    read     = fit gate, optionally offset by a reluctance learned per phase

The reluctance is the interesting risk here: it is trained by the teacher's
yes/no, and in phase 2 the only data is 5-9, so it can push the 0-4 experts
into silence even though their content is untouched. Reported both ways.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "competitive"))
sys.path.insert(0, str(HERE.parent / "fashion"))
from compete import load, join, geo_step, EPS                      # noqa: E402
from dopamine import geo_step_neg                                  # noqa: E402

OUT = HERE / "results"
H, K, ETA = 40, 36, 0.5
EPOCHS, BATCH, MIN_S, SEED = 6, 128, 2, 0
LAM, GAMMA = 4.0, 0.3
ETA_NEG, CAP_NEG, WINDOW = 0.25, np.pi / 16, 0.8
CAL_EPOCHS, CAL_LR = 6, 0.002
REPEL = "--no-repel" not in sys.argv
CONSOL = "--no-consolidate" not in sys.argv
TAG = ("batchconsol" if CONSOL else "plain") + ("" if REPEL else "_norepel")
PH1, PH2 = list(range(5)), list(range(5, 10))
BASE = {"logistic (SGD)": (0.9487, 0.0094), "MLP 256 (SGD)": (0.9765, 0.0000),
        "old rule (label-picked)": (0.9757, 0.9526)}


def rebuild(W, Q, n_img):
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
    R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)
    e = np.linalg.norm(Q[:, None, :n_img] - R[:, :, :n_img], axis=2) / np.maximum(
        np.linalg.norm(Q[:, :n_img], axis=1)[:, None], EPS)
    return e, R


def errors_of(W, X, n_img, chunk=2000):
    return np.concatenate([rebuild(W, join(X[s:s + chunk]), n_img)[0]
                           for s in range(0, len(X), chunk)])


def calibrate(E, y, claim, alive, b, rng, plastic):
    """Update the reluctances from the teacher's yes/no on THIS phase's data."""
    E = np.where(alive[None], E, np.inf)
    for _ in range(CAL_EPOCHS):
        for i in rng.permutation(len(y)):
            s = E[i] + b
            w = s.argmin()
            if claim[w] == y[i]:
                continue
            same = np.where(claim == y[i])[0]
            if not len(same):
                continue
            c = same[s[same].argmin()]
            b[w] += CAL_LR * plastic[w]; b[c] -= CAL_LR * plastic[c]
    return b


def evaluate(W, wins, Xte, yte, n_img, b):
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.where(alive[None], errors_of(W, Xte, n_img), np.inf)
    pred = claim[(E + b[None]).argmin(1)]
    mA, mB = np.isin(yte, PH1), np.isin(yte, PH2)
    return (float((pred[mA] == yte[mA]).mean()), float((pred[mB] == yte[mB]).mean()),
            float((pred == yte).mean()))


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load()
    n_img = Xtr.shape[1]
    rng = np.random.default_rng(SEED + 1)
    W = rng.standard_normal((H, K, n_img + 10))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H, 10), dtype=np.int64)
    f = np.full(H, 1.0 / H)
    b = np.zeros(H)
    res, snaps, seen = {}, {}, 0

    for ph, digits in (("1", PH1), ("2", PH2)):
        idx = np.nonzero(np.isin(ytr, digits))[0]
        before = wins.copy()
        phase_wins = np.zeros(H, dtype=np.int64)
        n_rep = 0
        for ep in range(EPOCHS):
            o = rng.permutation(idx)
            for s in range(0, len(o), BATCH):
                bt = o[s:s + BATCH]
                J, Q = join(Xtr[bt], ytr[bt]), join(Xtr[bt])
                err, R = rebuild(W, Q, n_img)
                L = R[:, :, n_img:]
                conf = L[np.arange(len(bt)), :, ytr[bt]] / np.maximum(
                    np.linalg.norm(L, axis=2), EPS)
                score = err + LAM * (1.0 - conf) - GAMMA * (1.0 / H - f)[None]
                win = score.argmin(1)
                cnt = np.bincount(win, minlength=H)
                f = 0.99 * f + 0.01 * (cnt / max(cnt.sum(), 1))
                for h in range(H):
                    m = win == h
                    if m.sum() >= MIN_S:
                        W[h] = geo_step(W[h], J[m], ETA)
                    np.add.at(wins[h], ytr[bt][m], 1)
                    phase_wins[h] += int(m.sum())
                if REPEL and seen > 0:             # no repulsion in the 1st epoch ever
                    claim = np.where(wins.sum(1) > 0, wins.argmax(1), -1)
                    fw = err.argmin(1)
                    same = claim[None, :] == ytr[bt][:, None]
                    e_right = np.where(same, err, np.inf).min(1)
                    bad = ((claim[fw] != ytr[bt]) & np.isfinite(e_right) &
                           (err[np.arange(len(bt)), fw] /
                            np.maximum(e_right, EPS) > WINDOW))
                    plastic = (cnt >= MIN_S) if CONSOL else np.ones(H, bool)
                    for h in np.unique(fw[bad]):
                        p = bad & (fw == h)
                        if p.sum() >= MIN_S and plastic[h]:
                            W[h] = geo_step_neg(W[h], Q[p], ETA_NEG, CAP_NEG)
                            n_rep += int(p.sum())
            seen += 1
            print(f"  phase {ph} epoch {ep+1}/{EPOCHS}   repelled {n_rep}", flush=True)

        raw = evaluate(W, wins, Xte, yte, n_img, np.zeros(H))
        alive = wins.sum(1) > 0
        claim = np.where(alive, wins.argmax(1), -1)
        Etr = errors_of(W, Xtr[idx], n_img)
        # at the gate: only hypercolumns that won something this phase
        plastic = (phase_wins > 0).astype(float) if CONSOL else np.ones(H)
        b = calibrate(Etr, ytr[idx], claim, alive, b, np.random.default_rng(0),
                      plastic)
        withA = evaluate(W, wins, Xte, yte, n_img, b)
        liveB = before.sum(1) > before.sum() * 0.002 if before.sum() else np.zeros(H, bool)
        res[f"phase {ph}"] = {
            "raw": {"acc_0_4": raw[0], "acc_5_9": raw[1], "acc_all": raw[2]},
            "with_reluctance": {"acc_0_4": withA[0], "acc_5_9": withA[1],
                                "acc_all": withA[2]},
            "live": int(alive.sum()), "repelled": n_rep,
            "phase_wins_to_previously_live":
                float(phase_wins[liveB].sum() / max(phase_wins.sum(), 1)),
            "phase_wins_to_previously_dead":
                float(phase_wins[~liveB].sum() / max(phase_wins.sum(), 1))}
        snaps[ph] = (wins.copy(), phase_wins.copy())
        print(f"[after phase {ph}]  raw  0-4 {raw[0]:.4f}  5-9 {raw[1]:.4f}  "
              f"all {raw[2]:.4f}   |  +reluctance  0-4 {withA[0]:.4f}  "
              f"5-9 {withA[1]:.4f}  all {withA[2]:.4f}   ({int(alive.sum())} live)")

    w1, _ = snaps["1"]; w2, pw2 = snaps["2"]
    live1 = w1.sum(1) > w1.sum() * 0.002
    poached = [(int(h), int(w1[h].argmax()), int(w2[h].argmax()))
               for h in np.nonzero(live1)[0] if w1[h].argmax() != w2[h].argmax()]
    print(f"\nafter phase 1: {live1.sum()} live, {(~live1).sum()} dead")
    print(f"phase-2 samples claimed by previously LIVE: "
          f"{pw2[live1].sum()/pw2.sum()*100:.1f}%   previously DEAD: "
          f"{pw2[~live1].sum()/pw2.sum()*100:.1f}%")
    print(f"phase-1 hypercolumns whose claimed digit CHANGED: "
          f"{len(poached)}/{live1.sum()}" + (f"  {poached[:8]}" if poached else ""))
    res["mechanism"] = {"dead_after_phase1": int((~live1).sum()),
                        "phase2_to_previously_live": float(pw2[live1].sum() / pw2.sum()),
                        "phase2_to_previously_dead": float(pw2[~live1].sum() / pw2.sum()),
                        "changed_claim": len(poached), "changes": poached}

    print(f"\nrepulsion: {'ON' if REPEL else 'OFF'}   "
          f"consolidation: {'ON' if CONSOL else 'OFF'}")
    print(f"correctable at the gate in phase 2: "
          f"{int((snaps['2'][1] > 0).sum())} of {H} hypercolumns")
    print("forgetting on 0-4:")
    for k, (p1, p2) in BASE.items():
        print(f"  {k:<24} {p1:.4f} -> {p2:.4f}   (forgot {p1-p2:+.4f})")
    for tag in ("raw", "with_reluctance"):
        p1 = res["phase 1"][tag]["acc_0_4"]; p2 = res["phase 2"][tag]["acc_0_4"]
        print(f"  {'new rule, ' + tag:<24} {p1:.4f} -> {p2:.4f}   "
              f"(forgot {p1-p2:+.4f})")
    res["baselines"] = {k: list(v) for k, v in BASE.items()}
    (OUT / f"metrics_{TAG}.json").write_text(json.dumps(res, indent=2))
    np.savez_compressed(OUT / "state.npz", W=W.astype(np.float32), wins1=w1,
                        wins2=w2, phase2_wins=pw2, b=b)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
