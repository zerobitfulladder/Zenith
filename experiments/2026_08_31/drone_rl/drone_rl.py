"""Competitive experts flying a drone, trained by reward alone. No teacher.

One layer. An expert is a subspace over [situation ; motor], so completing
the blank motor half from a written situation IS the policy -- the same
partial-cue read that produced a label from an image.

    situation   dx, dy, vx, vy, tilt, gyro     (place-coded, 6 channels)
    motor       left thrust, right thrust      (place-coded, 2 channels)

Selection follows the project's rule -- **value only at the selector**:
FIT says which experts are competent in this state, and a scalar V per
expert picks among them. Reward never edits template content; it only
decides who acts, and whether what they did is worth consolidating.

    candidates = top-m experts by fit to the situation
    act        = argmax V among them (epsilon-greedy), + exploration noise
    TD(lambda) = delta = r + gamma V[h'] - V[h];  V += alpha * delta * e
    learning   = the acting expert absorbs [situation ; what it actually did]
                 only when delta > 0 -- consolidate what beat expectation
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "place_code"))
sys.path.insert(0, str(HERE.parents[1] / "2026_08_28" / "single_layer_drone"))
from place_code import PlaceCode                                    # noqa: E402
from sl_drone import (physics, any_init, at_goal, wrap, teacher,    # noqa: E402
                      LEVELS, TMAX, HOVER, DT)

OUT = HERE / "results"
EPS = 1e-12
H, K, NB, RHO, ETA = 48, 8, 12, 1.0, 0.30
GAMMA, LAM, ALPHA, VCLIP = 0.99, 0.90, 0.02, 400.0
EPISODES, MAX_STEPS, EVAL_EVERY, N_EVAL = 8000, 300, 500, 120
SEED = 0

SIT_LO = np.array([-8., -8., -5., -5., -np.pi, -8.])
SIT_HI = np.array([8., 8., 5., 5., np.pi, 8.])
N_SIT, N_MOT = 6, 2
pc_sit = PlaceCode(N_SIT, nb=NB, lo=SIT_LO, hi=SIT_HI, halfw=1.5)
pc_mot = PlaceCode(N_MOT, nb=NB, lo=0.0, hi=TMAX, halfw=1.5)
D_SIT, D_MOT = pc_sit.size, pc_mot.size
DIM = D_SIT + D_MOT


def sense(s, tgt):
    return np.array([s[0] - tgt[0], s[1] - tgt[1], s[2], s[3], wrap(s[4]), s[5]])


def encode(sit, mot=None):
    """[situation ; motor], the motor block scaled to RHO x the situation's."""
    v = np.zeros(DIM)
    a = pc_sit.encode(sit[None])[0]
    v[:D_SIT] = a
    if mot is not None:
        b = pc_mot.encode(np.clip(mot, 0, TMAX)[None])[0]
        v[D_SIT:] = b * (RHO * np.linalg.norm(a) / max(np.linalg.norm(b), EPS))
    return v / max(np.linalg.norm(v), EPS)


def geo_step(Wh, x, eta):
    s = Wh @ x
    e = x - s @ Wh
    m = np.outer(s, e)
    tau = m - (m * Wh).sum(1, keepdims=True) * Wh
    tn = np.linalg.norm(tau, axis=1)
    th = np.clip(eta * tn, 0.0, np.pi / 4)
    hat = np.zeros_like(tau); live = tn > EPS
    hat[live] = tau[live] / tn[live, None]
    Wh = Wh * np.cos(th)[:, None] + hat * np.sin(th)[:, None]
    return Wh / (np.linalg.norm(Wh, axis=1, keepdims=True) + EPS)


class Bank:
    """Seeded so the starting policy HOVERS rather than tumbles.

    Each expert gets a random pocket of state space and K jittered
    [situation ; near-hover thrust] samples through it. That is a prior on
    the action, not a teacher: hovering is the obvious thing to babble
    around, and without it every episode is a tumble inside 0.3 s and there
    is no experience to learn from.
    """

    def __init__(self, rng):
        self.W = np.zeros((H, K, DIM))
        for h in range(H):
            c = rng.uniform(SIT_LO * 0.6, SIT_HI * 0.6)
            rows = np.stack([encode(c + rng.normal(0, 0.15, N_SIT) * (SIT_HI - SIT_LO),
                                    HOVER + rng.normal(0, 0.25, N_MOT))
                             for _ in range(K)])
            self.W[h] = np.linalg.svd(rows, full_matrices=False)[2][:K]
        self.W /= np.linalg.norm(self.W, axis=2, keepdims=True) + EPS
        self.V = np.zeros(H)
        self.n_boot = H
        self.uses = np.zeros(H, dtype=np.int64)

    def read(self, q):
        """(fit error per expert, the motors each would command)."""
        S = (q @ self.W.reshape(H * K, DIM).T).reshape(H, K)
        R = np.einsum('hk,hkd->hd', S, self.W)
        err = (np.linalg.norm(q[None, :D_SIT] - R[:, :D_SIT], axis=1) /
               max(np.linalg.norm(q[:D_SIT]), EPS))
        mot = pc_mot.decode(np.maximum(R[:, D_SIT:], 0.0))
        return err, np.clip(mot, 0.0, TMAX)


def reward(s, tgt):
    d = np.hypot(s[0] - tgt[0], s[1] - tgt[1])
    return -(0.10 * d + 0.05 * np.hypot(s[2], s[3])
             + 0.30 * abs(wrap(s[4])) + 0.02 * abs(s[5]))


def dead(s):
    return abs(s[0]) > 12 or abs(s[1]) > 12 or abs(wrap(s[4])) > 1.4


def run_episode(bank, rng, eps, sigma, learn=True, m_cand=5, hard=1.0):
    s, tgt = any_init(rng)
    s = s * hard; tgt = tgt * hard        # curriculum: start easy, widen
    e_tr = np.zeros(H)
    total, steps, hit = 0.0, 0, False
    for _ in range(MAX_STEPS):
        q = encode(sense(s, tgt))
        err, mot = bank.read(q)
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
        bank.uses[h] += 1
        total += r; steps += 1
        if learn:
            q2 = encode(sense(s2, tgt))
            h2 = int(np.argsort(bank.read(q2)[0])[0])
            delta = r + (0.0 if done else GAMMA * bank.V[h2]) - bank.V[h]
            e_tr *= GAMMA * LAM
            e_tr[h] = 1.0                             # REPLACING traces: accumulating
            bank.V += ALPHA * delta * e_tr            # ones diverged (V hit -4e64)
            np.clip(bank.V, -VCLIP, VCLIP, out=bank.V)
            if delta > 0:                             # consolidate what beat expectation
                bank.W[h] = geo_step(bank.W[h], encode(sense(s, tgt), a), ETA)
        s = s2
        if done:
            break
    return total, steps, hit


def evaluate(bank, seed, n=N_EVAL):
    rng = np.random.default_rng(seed)
    R, hits, st = [], 0, []
    for _ in range(n):
        t, k, h = run_episode(bank, rng, 0.0, 0.0, learn=False)
        R.append(t); hits += h; st.append(k)
    return float(np.mean(R)), hits / n, float(np.mean(st))


def baselines(seed, n=N_EVAL):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("teacher", "random"):
        R, hits = [], 0
        for _ in range(n):
            s, tgt = any_init(rng); tot = 0.0
            for _ in range(MAX_STEPS):
                lv = teacher(s, tgt) if name == "teacher" else (
                    int(rng.integers(13)), int(rng.integers(13)))
                s = physics(s, lv); r = reward(s, tgt)
                if at_goal(s, tgt): r += 8.0; tot += r; hits += 1; break
                if dead(s): r -= 8.0; tot += r; break
                tot += r
            R.append(tot)
        out[name] = {"reward": float(np.mean(R)), "success": hits / n}
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    bank = Bank(np.random.default_rng(SEED + 1))
    base = baselines(99)
    print(f"{H} experts x {K} templates, dim {DIM} "
          f"({N_SIT} situation + {N_MOT} motor channels x {NB} cells)")
    print(f"baselines over {N_EVAL} episodes:  "
          f"teacher reward {base['teacher']['reward']:7.1f} success "
          f"{base['teacher']['success']:.3f}   |   random reward "
          f"{base['random']['reward']:7.1f} success {base['random']['success']:.3f}\n",
          flush=True)
    curve = []
    for ep in range(EPISODES):
        frac = ep / EPISODES
        run_episode(bank, rng, eps=0.45 * (1 - frac) + 0.05,
                    sigma=1.2 * (1 - frac) + 0.10,
                    hard=min(1.0, 0.3 + 1.4 * frac))
        if (ep + 1) % EVAL_EVERY == 0:
            r, sc, st = evaluate(bank, 7)
            curve.append({"episode": ep + 1, "reward": r, "success": sc,
                          "steps": st, "live": int((bank.uses > 0).sum())})
            print(f"  ep {ep+1:>5}   reward {r:8.1f}   success {sc:.3f}   "
                  f"mean steps {st:5.1f}   experts used {curve[-1]['live']}/{H}   "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    r, sc, st = evaluate(bank, 7)
    res = {"config": {"H": H, "K": K, "nb": NB, "dim": DIM, "gamma": GAMMA,
                      "lambda": LAM, "alpha": ALPHA, "episodes": EPISODES},
           "final": {"reward": r, "success": sc, "steps": st},
           "baselines": base, "curve": curve,
           "expert_uses": bank.uses.tolist(), "V": bank.V.tolist(),
           "seconds": round(time.time() - t0, 1)}
    np.savez_compressed(OUT / "bank.npz", W=bank.W.astype(np.float32),
                        V=bank.V, uses=bank.uses)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
    print(f"\nfinal: reward {r:.1f}  success {sc:.3f}   "
          f"(teacher {base['teacher']['success']:.3f}, "
          f"random {base['random']['success']:.3f})")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    e = [c["episode"] for c in curve]
    ax[0].plot(e, [c["reward"] for c in curve], "o-", color="#1b6ca8")
    ax[0].axhline(base["teacher"]["reward"], ls="--", color="#2f8f4e", label="teacher")
    ax[0].axhline(base["random"]["reward"], ls=":", color="#c1462d", label="random")
    ax[0].set_title("episode reward"); ax[0].legend(fontsize=7)
    ax[1].plot(e, [c["success"] for c in curve], "o-", color="#1b6ca8")
    ax[1].axhline(base["teacher"]["success"], ls="--", color="#2f8f4e")
    ax[1].axhline(base["random"]["success"], ls=":", color="#c1462d")
    ax[1].set_title("reached the goal"); ax[1].set_ylim(-0.02, 1.02)
    ax[2].bar(range(H), np.sort(bank.uses)[::-1], color="#7a4fa3")
    ax[2].set_title("how often each expert acted"); ax[2].set_xlabel("expert (sorted)")
    for a in ax[:2]:
        a.set_xlabel("episodes"); a.grid(alpha=.3)
    fig.suptitle("competitive experts flying by reward alone", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "01_learning.png", dpi=130); plt.close(fig)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------- viewer ---
def load_policy(npz, cfg):
    """viewer.py hook: greedy, no exploration. Fit picks who is competent
    here, V picks the best of those, and that expert's completed motor
    block is the command."""
    bank = Bank(np.random.default_rng(0))
    bank.W = np.asarray(npz["W"], dtype=np.float64)
    bank.V = np.asarray(npz["V"], dtype=np.float64)
    m = int(cfg.get("m_cand", 5))

    def act(s, tgt, lv_prev=None):
        err, mot = bank.read(encode(sense(s, tgt)))
        cand = np.argsort(err)[:m]
        a = np.clip(mot[int(cand[np.argmax(bank.V[cand])])], 0.0, TMAX)
        return (int(np.argmin(np.abs(LEVELS - a[0]))),
                int(np.argmin(np.abs(LEVELS - a[1]))))
    return act
