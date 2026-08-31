"""Same experts, same gate — credit assigned by episode return instead.

`drone_rl.py` failed because it consolidated an action whenever the
single-step TD error was positive, which mostly means "the value estimate was
pessimistic", not "the action was good". It fired on half of all steps and
random-walked the policy away from its own hover seed.

Here, nothing is learned during an episode. A batch of episodes is run, each
step's **return-to-go** is computed, and then each expert consolidates only
the top slice of its own experience -- ranked by **advantage**, the
return-to-go minus that expert's own average. Raw return would just rank the
expert's easy states above its hard ones; the advantage asks "did this action
beat what I normally manage from here".

    per batch:  roll out, record (expert, [situation ; action], return-to-go)
                V[h] <- running mean of the returns h collected  (Monte Carlo,
                        which also sidesteps the TD divergence from before)
                each h rotates toward its top-q experiences by G - V[h]

Value is still only at the selector: fit says who is competent here, V picks
among them, and reward never edits template content -- it only chooses which
experiences are worth keeping.
"""

import json, time
from pathlib import Path
import numpy as np

from drone_rl import (Bank, encode, sense, geo_step, reward, dead, evaluate,
                      baselines, H, K, DIM, EPS, OUT, TMAX, N_SIT, N_MOT,
                      SIT_LO, SIT_HI, HOVER, MAX_STEPS)
from sl_drone import physics, any_init, at_goal, LEVELS

GAMMA, BETA, TOPQ, ETA_C = 0.99, 0.30, 0.20, 0.15
BATCHES, PER_BATCH, EVAL_EVERY, SEED = 240, 25, 20, 0


def rollout(bank, rng, eps, sigma, hard, m_cand=5):
    """One episode. Records nothing but what the batch update needs."""
    s, tgt = any_init(rng)
    s = s * hard; tgt = tgt * hard
    hs, xs, rs = [], [], []
    hit = False
    for _ in range(MAX_STEPS):
        sit = sense(s, tgt)
        err, mot = bank.read(encode(sit))
        cand = np.argsort(err)[:m_cand]
        h = int(rng.choice(cand)) if rng.random() < eps else int(cand[np.argmax(bank.V[cand])])
        a = np.clip(mot[h] + rng.normal(0, sigma, 2), 0.0, TMAX)
        lv = (int(np.argmin(np.abs(LEVELS - a[0]))), int(np.argmin(np.abs(LEVELS - a[1]))))
        s2 = physics(s, lv)
        r = reward(s2, tgt)
        done = at_goal(s2, tgt) or dead(s2)
        if at_goal(s2, tgt):
            r += 8.0; hit = True
        elif dead(s2):
            r -= 8.0
        hs.append(h); xs.append(encode(sit, a)); rs.append(r)
        bank.uses[h] += 1
        s = s2
        if done:
            break
    g, G = 0.0, np.empty(len(rs))                       # return-to-go
    for t in range(len(rs) - 1, -1, -1):
        g = rs[t] + GAMMA * g
        G[t] = g
    return np.array(hs), np.array(xs), G, sum(rs), hit


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    bank = Bank(np.random.default_rng(SEED + 1))
    base = baselines(99)
    print(f"{H} experts x {K} templates, dim {DIM}   "
          f"credit = episode return, top {TOPQ:.0%} by advantage")
    print(f"baselines: teacher success {base['teacher']['success']:.3f}   "
          f"random {base['random']['success']:.3f}\n", flush=True)

    curve = []
    for it in range(BATCHES):
        frac = it / BATCHES
        eps = 0.40 * (1 - frac) + 0.05
        sig = 1.2 * (1 - frac) + 0.10
        hard = min(1.0, 0.35 + 1.3 * frac)
        HS, XS, GS = [], [], []
        for _ in range(PER_BATCH):
            h, x, g, _, _ = rollout(bank, rng, eps, sig, hard)
            HS.append(h); XS.append(x); GS.append(g)
        HS, XS, GS = np.concatenate(HS), np.concatenate(XS), np.concatenate(GS)

        for h in range(H):                              # Monte-Carlo baseline
            m = HS == h
            if m.sum() >= 4:
                bank.V[h] = (1 - BETA) * bank.V[h] + BETA * GS[m].mean()
        for h in range(H):                              # keep the best slice
            m = np.nonzero(HS == h)[0]
            if len(m) < 8:
                continue
            adv = GS[m] - bank.V[h]
            keep = m[np.argsort(-adv)[:max(2, int(TOPQ * len(m)))]]
            if (adv[np.argsort(-adv)[:len(keep)]] > 0).any():
                bank.W[h] = geo_step(bank.W[h], XS[keep].mean(0) /
                                     max(np.linalg.norm(XS[keep].mean(0)), EPS), ETA_C)

        if (it + 1) % EVAL_EVERY == 0:
            r, sc, st = evaluate(bank, 7)
            curve.append({"episodes": (it + 1) * PER_BATCH, "reward": r,
                          "success": sc, "steps": st})
            print(f"  ep {(it+1)*PER_BATCH:>6}   reward {r:8.1f}   success {sc:.3f}"
                  f"   mean steps {st:6.1f}   [{time.time()-t0:.0f}s]", flush=True)

    r, sc, st = evaluate(bank, 7, n=300)
    print(f"\nfinal (300 episodes): reward {r:.1f}   success {sc:.3f}   "
          f"(teacher {base['teacher']['success']:.3f})")
    np.savez_compressed(OUT / "cem.npz", W=bank.W.astype(np.float32),
                        V=bank.V, uses=bank.uses)
    (OUT / "cem.json").write_text(json.dumps(
        {"kind": "module", "module": "drone_cem",
         "dir": "experiments/2026_08_31/drone_rl", "m_cand": 5}))
    (OUT / "cem_metrics.json").write_text(json.dumps(
        {"config": {"H": H, "K": K, "gamma": GAMMA, "topq": TOPQ,
                    "eta": ETA_C, "batches": BATCHES, "per_batch": PER_BATCH},
         "final": {"reward": r, "success": sc, "steps": st},
         "baselines": base, "curve": curve,
         "seconds": round(time.time() - t0, 1)}, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    e = [c["episodes"] for c in curve]
    ax[0].plot(e, [c["success"] for c in curve], "o-", color="#1b6ca8")
    ax[0].axhline(base["teacher"]["success"], ls="--", color="#2f8f4e", label="teacher")
    ax[0].set_ylim(-.02, 1.02); ax[0].set_title("reached the goal"); ax[0].legend(fontsize=7)
    ax[1].plot(e, [c["steps"] for c in curve], "o-", color="#7a4fa3")
    ax[1].set_title("episode length (survival)")
    for a in ax:
        a.set_xlabel("episodes"); a.grid(alpha=.3)
    fig.suptitle("credit by episode return, top 20% by advantage", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "02_cem.png", dpi=130); plt.close(fig)
    print(f"done in {time.time()-t0:.0f}s")


def load_policy(npz, cfg):
    from drone_rl import load_policy as lp
    return lp(npz, cfg)


if __name__ == "__main__":
    main()
