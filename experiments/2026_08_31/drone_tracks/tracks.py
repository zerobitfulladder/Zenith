"""Sensory track, motor track, and a table between them learned by TD.

The failures in ../drone_rl all came from ONE subspace doing three jobs at
once -- territory, code, and policy -- so improving the policy moved the
territory out from under it. Split them:

    sensory track   N experts over the 6 sensor channels, trained UNSUPERVISED
                    by fit competition. Emits a sparse vector: which expert
                    won, valued by its confidence. Reward never touches it.
    motor track     M experts over the 2 thrust channels, same rule on sampled
                    commands. A discrete vocabulary of actions.
    the picker      Q[n, m], a plain table, learned by TD. This is the only
                    thing reward is allowed to edit.

The policy is then tabular Q-learning over a learned discretisation of state
and action -- which converges, unlike fitting a controller inside a rotating
subspace.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "place_code"))
sys.path.insert(0, str(HERE.parents[0] / "competitive"))
sys.path.insert(0, str(HERE.parents[1] / "2026_08_28" / "single_layer_drone"))
from place_code import PlaceCode                                   # noqa: E402
from compete import geo_step                                       # noqa: E402
from sl_drone import physics, any_init, at_goal, wrap, teacher, LEVELS, TMAX, HOVER  # noqa

OUT = HERE / "results"
EPS = 1e-12
N_SENS, K_SENS, M_MOT, K_MOT, NB = 256, 6, 25, 3, 12
ALPHA, GAMMA, MAX_STEPS = 0.15, 0.99, 300
PRETRAIN_EP, EPISODES, EVAL_EVERY, N_EVAL = 400, 24000, 2000, 150
SEED = 0

SIT_LO = np.array([-8., -8., -5., -5., -np.pi, -8.])
SIT_HI = np.array([8., 8., 5., 5., np.pi, 8.])
pc_s = PlaceCode(6, nb=NB, lo=SIT_LO, hi=SIT_HI, halfw=1.5)
pc_m = PlaceCode(2, nb=NB, lo=0.0, hi=TMAX, halfw=1.5)
D_S, D_M = pc_s.size, pc_m.size


def sense(s, tgt):
    return np.array([s[0] - tgt[0], s[1] - tgt[1], s[2], s[3], wrap(s[4]), s[5]])


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def fit_track(X, n_exp, k, rng, epochs=6, batch=256, eta=0.4):
    """Competitive experts, fit-only, with a conscience. No labels, no reward."""
    d = X.shape[1]
    W = rng.standard_normal((n_exp, k, d))
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    f = np.full(n_exp, 1.0 / n_exp)
    for _ in range(epochs):
        for s in range(0, len(X), batch):
            B = X[rng.permutation(len(X))[:batch]] if s == 0 else X[s:s + batch]
            if len(B) < 8:
                continue
            S = (B @ W.reshape(n_exp * k, d).T).reshape(len(B), n_exp, k).transpose(1, 0, 2)
            e = np.linalg.norm(B[None] - np.matmul(S, W), axis=2) + 1.0 * (f - 1.0 / n_exp)[:, None]
            win = e.argmin(0)
            c = np.bincount(win, minlength=n_exp)
            f = 0.99 * f + 0.01 * (c / max(c.sum(), 1))
            for h in range(n_exp):
                m = win == h
                if m.sum() >= 2:
                    W[h] = geo_step(W[h], B[m], eta)
    return W


def which(W, q):
    """(winner, confidence) for one already-unit-normalised vector."""
    n, k, d = W.shape
    S = (q @ W.reshape(n * k, d).T).reshape(n, k)
    e = np.linalg.norm(q[None] - np.einsum('hk,hkd->hd', S, W), axis=1)
    i = int(e.argmin())
    return i, float(1.0 - e[i])


def reward(s, tgt):
    return -(0.10 * np.hypot(s[0] - tgt[0], s[1] - tgt[1])
             + 0.05 * np.hypot(s[2], s[3]) + 0.30 * abs(wrap(s[4])) + 0.02 * abs(s[5]))


def dead(s):
    return abs(s[0]) > 12 or abs(s[1]) > 12 or abs(wrap(s[4])) > 1.4


def episode(Ws, acts, Q, rng, eps, learn=True, hard=1.0):
    s, tgt = any_init(rng)
    s = s * hard; tgt = tgt * hard
    tot, steps, hit = 0.0, 0, False
    n, _ = which(Ws, unit(pc_s.encode(sense(s, tgt)[None]))[0])
    for _ in range(MAX_STEPS):
        m = int(rng.integers(len(acts))) if rng.random() < eps else int(Q[n].argmax())
        a = acts[m]
        lv = (int(np.argmin(np.abs(LEVELS - a[0]))), int(np.argmin(np.abs(LEVELS - a[1]))))
        s2 = physics(s, lv)
        r = reward(s2, tgt)
        done = at_goal(s2, tgt) or dead(s2)
        if at_goal(s2, tgt):
            r += 8.0; hit = True
        elif dead(s2):
            r -= 8.0
        n2, _ = which(Ws, unit(pc_s.encode(sense(s2, tgt)[None]))[0])
        if learn:
            tgtq = r + (0.0 if done else GAMMA * Q[n2].max())
            Q[n, m] += ALPHA * (tgtq - Q[n, m])
        tot += r; steps += 1; s, n = s2, n2
        if done:
            break
    return tot, steps, hit


def evaluate(Ws, acts, Q, seed, n=N_EVAL):
    rng = np.random.default_rng(seed)
    R = [episode(Ws, acts, Q, rng, 0.0, learn=False) for _ in range(n)]
    return (float(np.mean([x[0] for x in R])), sum(x[2] for x in R) / n,
            float(np.mean([x[1] for x in R])))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ---- sensory track: states from random flying, no reward involved ----
    S = []
    for _ in range(PRETRAIN_EP):
        s, tgt = any_init(rng)
        for _ in range(120):
            S.append(sense(s, tgt))
            s = physics(s, (int(rng.integers(13)), int(rng.integers(13))))
            if dead(s):
                break
    S = unit(pc_s.encode(np.array(S)))
    Ws = fit_track(S, N_SENS, K_SENS, np.random.default_rng(1))
    print(f"sensory track: {N_SENS} experts x {K_SENS} on {len(S)} states "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- motor track: a vocabulary of commands ---------------------------
    Mraw = rng.uniform(0, TMAX, (20000, 2))
    Wm = fit_track(unit(pc_m.encode(Mraw)), M_MOT, K_MOT, np.random.default_rng(2))
    acts = np.clip(pc_m.decode(np.array([
        np.einsum('k,kd->d', Wm[i] @ Wm[i].mean(0), Wm[i]) for i in range(M_MOT)])), 0, TMAX)
    acts = np.clip(pc_m.decode(np.maximum(np.array(
        [Wm[i].mean(0) for i in range(M_MOT)]), 0)), 0, TMAX)
    print(f"motor track: {M_MOT} commands, thrust range "
          f"[{acts.min():.2f}, {acts.max():.2f}], hover is {HOVER:.2f}", flush=True)

    # ---- the picker ------------------------------------------------------
    Q = np.zeros((N_SENS, M_MOT))
    curve = []
    for ep in range(EPISODES):
        frac = ep / EPISODES
        episode(Ws, acts, Q, rng, eps=0.5 * (1 - frac) + 0.03,
                hard=min(1.0, 0.35 + 1.3 * frac))
        if (ep + 1) % EVAL_EVERY == 0:
            r, sc, st = evaluate(Ws, acts, Q, 7)
            curve.append({"episode": ep + 1, "reward": r, "success": sc, "steps": st})
            print(f"  ep {ep+1:>6}   reward {r:8.1f}   success {sc:.3f}   "
                  f"steps {st:6.1f}   visited {int((Q!=0).any(1).sum())}/{N_SENS} states"
                  f"   [{time.time()-t0:.0f}s]", flush=True)

    r, sc, st = evaluate(Ws, acts, Q, 7, n=300)
    print(f"\nfinal: reward {r:.1f}   success {sc:.3f}   steps {st:.1f}")
    np.savez_compressed(OUT / "tracks.npz", Ws=Ws.astype(np.float32),
                        acts=acts, Q=Q)
    (OUT / "tracks.json").write_text(json.dumps(
        {"kind": "module", "module": "tracks",
         "dir": "experiments/2026_08_31/drone_tracks"}))
    (OUT / "metrics.json").write_text(json.dumps(
        {"config": {"n_sens": N_SENS, "k_sens": K_SENS, "m_mot": M_MOT,
                    "alpha": ALPHA, "gamma": GAMMA, "episodes": EPISODES},
         "final": {"reward": r, "success": sc, "steps": st}, "curve": curve,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    e = [c["episode"] for c in curve]
    ax[0].plot(e, [c["success"] for c in curve], "o-", color="#1b6ca8")
    ax[0].axhline(0.942, ls="--", color="#2f8f4e", label="teacher"); ax[0].legend(fontsize=7)
    ax[0].set_title("reached the goal"); ax[0].set_ylim(-.02, 1.02)
    ax[1].plot(e, [c["steps"] for c in curve], "o-", color="#7a4fa3")
    ax[1].set_title("episode length")
    im = ax[2].imshow(Q, aspect="auto", cmap="magma")
    ax[2].set_title("Q[sensory, motor]"); ax[2].set_xlabel("motor command")
    ax[2].set_ylabel("sensory category"); fig.colorbar(im, ax=ax[2], fraction=.046)
    for a in ax[:2]:
        a.set_xlabel("episodes"); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "01_tracks.png", dpi=130); plt.close(fig)
    print(f"done in {time.time()-t0:.0f}s")


def load_policy(npz, cfg):
    Ws, acts, Q = npz["Ws"].astype(np.float64), npz["acts"], npz["Q"]

    def act(s, tgt, lv_prev=None):
        n, _ = which(Ws, unit(pc_s.encode(sense(s, tgt)[None]))[0])
        a = acts[int(Q[n].argmax())]
        return (int(np.argmin(np.abs(LEVELS - a[0]))), int(np.argmin(np.abs(LEVELS - a[1]))))
    return act


if __name__ == "__main__":
    main()
