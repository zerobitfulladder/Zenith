"""Reinforce what contributed, weaken what did not — and price the damage.

The current rule only has the positive half: the expert most confident in the
true label takes a step toward the sample. Nothing is ever pushed away from
anything, so an expert's subspace can grow to cover other classes for free.

Arms (content differs only in the last one):
  none    the rule as it stands
  B       + repulsion: whoever WOULD have won the unlabelled fit competition,
          if its claimed class is wrong, is rotated AWAY from that image.
          Kohonen's window: only when it is genuinely close to the right
          expert, else the references drift.
  A       none + a learned reluctance per expert (selector only, content frozen)
  A+B     both

For every arm we measure not just accuracy but what it cost the model: the
reconstruction error of an image by an expert of its OWN class. If B buys
discrimination by wrecking the generative content, that number goes up.
"""

import json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, join, EPS, geo_step, CLASSES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
H, K, ETA = 40, 36, 0.5
EPOCHS, BATCH, MIN_S = 6, 128, 2
ETA_NEG, CAP_NEG, WINDOW, WARMUP = 0.25, np.pi / 16, 0.8, 1
CAL_EPOCHS, CAL_LR = 6, 0.002
SEED = 0
N_IMG = 784


def geo_step_neg(Wh, B, eta, cap):
    """The same geodesic step, run backwards: rotate AWAY from these vectors."""
    S = B @ Wh.T
    E = B - S @ Wh
    M = -(S.T @ E) / len(B)
    tau = M - (M * Wh).sum(1, keepdims=True) * Wh
    tn = np.linalg.norm(tau, axis=1)
    th = np.clip(eta * tn, 0.0, cap)
    hat = np.zeros_like(tau); live = tn > EPS
    hat[live] = tau[live] / tn[live, None]
    Wh = Wh * np.cos(th)[:, None] + hat * np.sin(th)[:, None]
    return Wh / (np.linalg.norm(Wh, axis=1, keepdims=True) + EPS)


def rebuild(W, Q):
    """(n, H) relative error of the image half, and the label completions."""
    h, k, d = W.shape
    S = (Q @ W.reshape(h * k, d).T).reshape(len(Q), h, k)
    R = np.matmul(S.transpose(1, 0, 2), W).transpose(1, 0, 2)          # (n, h, d)
    e = np.linalg.norm(Q[:, None, :N_IMG] - R[:, :, :N_IMG], axis=2) / np.maximum(
        np.linalg.norm(Q[:, :N_IMG], axis=1)[:, None], EPS)
    return e, R[:, :, N_IMG:]


def train(Xtr, ytr, repel, rng):
    d = N_IMG + 10
    W = rng.standard_normal((H, K, d))
    W -= W.mean(axis=2, keepdims=True)
    W /= np.linalg.norm(W, axis=2, keepdims=True) + EPS
    wins = np.zeros((H, 10), dtype=np.int64)
    n_rep = 0
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            J, Q = join(Xtr[b], ytr[b]), join(Xtr[b])
            err, P = rebuild(W, Q)
            conf = P[:, :, :].transpose(1, 0, 2)                        # (H,n,10)
            c = conf[:, np.arange(len(b)), ytr[b]] / np.maximum(
                np.linalg.norm(conf, axis=2), EPS)
            win = c.argmax(0)
            for h in range(H):
                m = win == h
                if m.sum() >= MIN_S:
                    W[h] = geo_step(W[h], J[m], ETA)
                np.add.at(wins[h], ytr[b][m], 1)
            if not repel or ep < WARMUP:
                continue
            # --- the negative half -------------------------------------------
            alive = wins.sum(1) > 0
            claim = np.where(alive, wins.argmax(1), -1)
            e = np.where(alive[None], err, np.inf)
            fit_win = e.argmin(1)
            wrong = claim[fit_win] != ytr[b]
            if not wrong.any():
                continue
            # best expert that DOES claim the true label, for the window test
            same = (claim[None, :] == ytr[b][:, None])
            e_right = np.where(same, e, np.inf).min(1)
            close = e[np.arange(len(b)), fit_win] / np.maximum(e_right, EPS) > WINDOW
            hit = wrong & close & np.isfinite(e_right)
            for h in np.unique(fit_win[hit]):
                m = hit & (fit_win == h)
                if m.sum() >= MIN_S:
                    W[h] = geo_step_neg(W[h], Q[m], ETA_NEG, CAP_NEG)
                    n_rep += int(m.sum())
        print(f"  {'B (repel)' if repel else 'none':<10} epoch {ep+1}/{EPOCHS}"
              f"   repelled {n_rep}", flush=True)
    return W, wins, n_rep


def calibrate(W, wins, Xtr, ytr, rng):
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.concatenate([rebuild(W, join(Xtr[s:s + 2000]))[0]
                        for s in range(0, len(Xtr), 2000)])
    E = np.where(alive[None], E, np.inf)
    b = np.zeros(H)
    for _ in range(CAL_EPOCHS):
        for i in rng.permutation(len(ytr)):
            s = E[i] + b
            w = s.argmin()
            if claim[w] == ytr[i]:
                continue
            same = np.where(claim == ytr[i])[0]
            if not len(same):
                continue
            b[w] += CAL_LR; b[same[s[same].argmin()]] -= CAL_LR
    return b


def report(W, wins, Xte, yte, b, tag):
    alive = wins.sum(1) > 0
    claim = np.where(alive, wins.argmax(1), -1)
    E = np.concatenate([rebuild(W, join(Xte[s:s + 2000]))[0]
                        for s in range(0, len(Xte), 2000)])
    E = np.where(alive[None], E, np.inf)
    pick = (E + b[None]).argmin(1)
    got = claim[pick]
    acc = float((got == yte).mean())
    own = (claim[None, :] == yte[:, None]) & alive[None]
    own_e = float(np.where(own, E, np.nan)[own.any(1)].min(1, initial=np.inf).mean()
                  if own.any() else np.nan)
    own_e = float(np.nanmean(np.where(own, E, np.nan).min(1)))
    oth_e = float(np.nanmean(np.where(~own & alive[None], E, np.nan).min(1)))
    per = {CLASSES[c]: float((got[yte == c] == c).mean()) for c in range(10)}
    print(f"  {tag:<8} acc {acc:.4f}   own-class rebuild err {own_e:.4f}   "
          f"best other-class {oth_e:.4f}   gap {oth_e-own_e:+.4f}   "
          f"alive {int(alive.sum())}")
    return {"acc": acc, "own_class_err": own_e, "other_class_err": oth_e,
            "gap": oth_e - own_e, "alive": int(alive.sum()), "per_class": per,
            "claim": claim.tolist()}, claim, E


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = load("fashion_mnist")
    res = {}
    print("training", flush=True)
    W0, wins0, _ = train(Xtr, ytr, False, np.random.default_rng(SEED + 1))
    WB, winsB, nrep = train(Xtr, ytr, True, np.random.default_rng(SEED + 1))
    print(f"\n(B repelled {nrep} sample-steps)\n\nreading:", flush=True)
    z = np.zeros(H)
    res["none"], claim0, E0 = report(W0, wins0, Xte, yte, z, "none")
    bA = calibrate(W0, wins0, Xtr, ytr, np.random.default_rng(0))
    res["A"], _, _ = report(W0, wins0, Xte, yte, bA, "A")
    res["B"], claimB, _ = report(WB, winsB, Xte, yte, z, "B")
    bAB = calibrate(WB, winsB, Xtr, ytr, np.random.default_rng(0))
    res["A+B"], _, _ = report(WB, winsB, Xte, yte, bAB, "A+B")

    # --- why a sneaker is called a sandal ---------------------------------
    sn, sa = CLASSES.index("Sneaker"), CLASSES.index("Sandal")
    m = yte == sn
    w = E0[m].argmin(1)
    top = np.bincount(w, minlength=H).argsort()[::-1][:3]
    print(f"\non SNEAKER test images, the fit gate picks:")
    for h in top:
        share = float((w == h).mean())
        if share == 0:
            continue
        print(f"  expert {h:>2}  wins {share*100:4.1f}% of sneakers   "
              f"claims {CLASSES[claim0[h]]:<12} "
              f"trained on {wins0[h]/max(wins0[h].sum(),1)}")
    print(f"  during TRAINING no expert won both: sneaker experts "
          f"{np.where(claim0==sn)[0].tolist()}, sandal experts "
          f"{np.where(claim0==sa)[0].tolist()}")

    (OUT / "dopamine_metrics.json").write_text(json.dumps(
        {"config": {"H": H, "K": K, "eta": ETA, "eta_neg": ETA_NEG,
                    "cap_neg": CAP_NEG, "window": WINDOW, "warmup": WARMUP,
                    "epochs": EPOCHS, "cal_epochs": CAL_EPOCHS, "cal_lr": CAL_LR},
         "results": res, "repelled_steps": nrep,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    np.savez_compressed(OUT / "dopamine_weights.npz",
                        W_none=W0.astype(np.float32), wins_none=wins0,
                        W_B=WB.astype(np.float32), wins_B=winsB, b_A=bA, b_AB=bAB)

    arms = ["none", "A", "B", "A+B"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax[0].bar(arms, [res[a]["acc"] for a in arms],
              color=["#999", "#2f8f4e", "#1b6ca8", "#6a3d9a"])
    for i, a in enumerate(arms):
        ax[0].text(i, res[a]["acc"], f"{res[a]['acc']:.4f}", ha="center",
                   va="bottom", fontsize=8)
    ax[0].set_ylim(0, 1); ax[0].grid(alpha=.25, axis="y")
    ax[0].set_title("accuracy", fontsize=9)
    w_ = .38
    ax[1].bar(np.arange(4) - w_/2, [res[a]["own_class_err"] for a in arms],
              width=w_, color="#2f8f4e", label="own class (lower = intact)")
    ax[1].bar(np.arange(4) + w_/2, [res[a]["other_class_err"] for a in arms],
              width=w_, color="#c1462d", label="best other class")
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(arms)
    ax[1].set_ylabel("relative rebuild error"); ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.25, axis="y"); ax[1].set_title("what it cost the model", fontsize=9)
    o = np.argsort([res["none"]["per_class"][c] for c in CLASSES])
    for j, a in enumerate(arms):
        ax[2].barh(np.arange(10) + j * .22, [res[a]["per_class"][CLASSES[c]] for c in o],
                   height=.21, label=a)
    ax[2].set_yticks(np.arange(10) + .33)
    ax[2].set_yticklabels([CLASSES[c] for c in o], fontsize=7)
    ax[2].set_xlim(0, 1); ax[2].legend(fontsize=7); ax[2].grid(alpha=.25, axis="x")
    ax[2].set_title("per class", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "40_dopamine.png", dpi=140); plt.close(fig)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}/40_dopamine.png")


if __name__ == "__main__":
    main()
