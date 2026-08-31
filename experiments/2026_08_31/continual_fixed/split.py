"""The 0.9390 architecture, unchanged, on a split stream. Forgetting only.

40 experts x 36 templates, pre-allocated, label-confidence competition,
fit gate at test -- exactly `../competitive40`. No hiring, no threshold, no
conscience. Phase 1 is digits 0-4, phase 2 is 5-9.

The question is who claims the 5-9 samples. Every sample must go to SOME
expert (argmax has no none-of-the-above branch), so either:

  (a) the 0-4 specialists get poached and drift  -> forgetting, or
  (b) the experts left dead after phase 1 win them, because a committed
      "3" expert outputs a NEGATIVE score for label 7 while an uncommitted
      one merely outputs noise -> specialisation protects itself.

Measured directly below: what fraction of phase-2 wins go to experts that
were live vs dead after phase 1, and how many live experts change their
claimed digit.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "competitive"))
sys.path.insert(0, str(HERE.parents[0] / "competitive40"))
from compete import load, join, geo_step, EPS                     # noqa: E402
from compete40 import predict, H, K, ETA                          # noqa: E402

OUT = HERE / "results"
EPOCHS, BATCH, MIN_S, SEED = 6, 128, 2, 0
A, B = list(range(5)), list(range(5, 10))
BASE = {"logistic (SGD)": (0.9487, 0.0094, 0.8949),
        "MLP 256 (SGD)": (0.9765, 0.0000, 0.9673)}


def evaluate(W, Xte, yte, n_img, wins):
    Q = join(Xte)
    P, R = predict(W, Q, n_img)
    err = np.linalg.norm(Q[None, :, :n_img] - R, axis=2) / np.maximum(
        np.linalg.norm(Q[None, :, :n_img], axis=2), EPS)
    pick = err.argmin(0)
    lab = wins.argmax(1)
    live = wins.sum(1) > 0
    pred = np.where(live[pick], lab[pick], -1)
    mA, mB = np.isin(yte, A), np.isin(yte, B)
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
    res, snaps = {}, {}

    for ph, digits in (("1", A), ("2", B)):
        m = np.isin(ytr, digits)
        idx = np.nonzero(m)[0]
        before = wins.copy()
        phase_wins = np.zeros(H, dtype=np.int64)
        for _ in range(EPOCHS):
            o = rng.permutation(idx)
            for s in range(0, len(o), BATCH):
                b = o[s:s + BATCH]
                J = join(Xtr[b], ytr[b])
                P, _ = predict(W, join(Xtr[b]), n_img)
                conf = P[:, np.arange(len(b)), ytr[b]] / np.maximum(
                    np.linalg.norm(P, axis=2), EPS)
                win = conf.argmax(0)
                for h in range(H):
                    sel = win == h
                    if sel.sum() >= MIN_S:
                        W[h] = geo_step(W[h], J[sel], ETA)
                    np.add.at(wins[h], ytr[b][sel], 1)
                    phase_wins[h] += sel.sum()
        a, bb, allacc = evaluate(W, Xte, yte, n_img, wins)
        liveB = before.sum(1) > before.sum() * 0.002 if before.sum() else np.zeros(H, bool)
        res[f"phase {ph}"] = {
            "acc_0_4": a, "acc_5_9": bb, "acc_all": allacc,
            "live_experts": int((wins.sum(1) > wins.sum() * 0.002).sum()),
            "phase_wins_to_previously_live": float(
                phase_wins[liveB].sum() / max(phase_wins.sum(), 1)) if ph == "2" else None,
            "phase_wins_to_previously_dead": float(
                phase_wins[~liveB].sum() / max(phase_wins.sum(), 1)) if ph == "2" else None,
        }
        snaps[ph] = (wins.copy(), phase_wins.copy(), liveB.copy())
        print(f"[after phase {ph}]  0-4 {a:.4f}   5-9 {bb:.4f}   all {allacc:.4f}   "
              f"{res[f'phase {ph}']['live_experts']} live experts")

    w1, pw1, _ = snaps["1"]
    w2, pw2, liveB = snaps["2"]
    live1 = w1.sum(1) > w1.sum() * 0.002
    print(f"\nafter phase 1: {live1.sum()} live, {(~live1).sum()} dead (never won)")
    print(f"phase-2 samples claimed by experts that were LIVE after phase 1: "
          f"{pw2[live1].sum()/pw2.sum()*100:.1f}%")
    print(f"phase-2 samples claimed by experts that were DEAD after phase 1: "
          f"{pw2[~live1].sum()/pw2.sum()*100:.1f}%")
    poached = [(h, int(w1[h].argmax()), int(w2[h].argmax())) for h in np.nonzero(live1)[0]
               if w1[h].argmax() != w2[h].argmax()]
    print(f"phase-1 experts whose claimed digit CHANGED: {len(poached)}/{live1.sum()}"
          + (f"  {poached[:8]}" if poached else ""))
    res["mechanism"] = {
        "dead_after_phase1": int((~live1).sum()),
        "phase2_to_previously_live": float(pw2[live1].sum() / pw2.sum()),
        "phase2_to_previously_dead": float(pw2[~live1].sum() / pw2.sum()),
        "experts_that_changed_claim": len(poached), "changes": poached}
    print("\nbaselines on the same split:")
    for k, (p1, p2, b59) in BASE.items():
        print(f"  {k:<16} 0-4 {p1:.4f} -> {p2:.4f}   (forgot {p1-p2:+.4f})")
    print(f"  {'this run':<16} 0-4 {res['phase 1']['acc_0_4']:.4f} -> "
          f"{res['phase 2']['acc_0_4']:.4f}   "
          f"(forgot {res['phase 1']['acc_0_4']-res['phase 2']['acc_0_4']:+.4f})")
    res["baselines"] = {k: {"p1": v[0], "p2": v[1], "5_9": v[2]} for k, v in BASE.items()}
    np.savez_compressed(OUT / "state.npz", W=W.astype(np.float32),
                        wins1=w1, wins2=w2, phase2_wins=pw2)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
