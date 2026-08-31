"""Can the tiling express the PID controller at all? Imitation, not reward.

The four RL failures could not separate "the representation cannot express a
good policy" from "the learner cannot find one". This separates them: hand it
the answer and see whether it can hold it.

    1. fly the PID teacher, record (situation, its thrusts)
    2. tile those states with competitive experts -- and note these are now
       states a COMPETENT controller visits, which is the allocation the
       unsupervised run got wrong
    3. per cell, store the teacher's answer, two ways:
         constant   the mean thrust the teacher used in that cell
         linear     a least-squares map from the 6 raw sensor values
    4. fly it

Sweeping the number of experts gives the resolution curve directly. If
success climbs with N, the tiling is the bottleneck and adaptive splitting is
the fix. If it plateaus low, constant-per-cell is the bottleneck and the cell
needs a policy inside it, not more cells.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "place_code"))
sys.path.insert(0, str(HERE.parents[0] / "competitive"))
sys.path.insert(0, str(HERE.parents[1] / "2026_08_28" / "single_layer_drone"))
from place_code import PlaceCode                                    # noqa: E402
from compete import geo_step                                        # noqa: E402
from sl_drone import (physics, any_init, at_goal, wrap, teacher,    # noqa: E402
                      LEVELS, TMAX, HOVER)

OUT = HERE / "results"
EPS = 1e-12
NS = [64, 256, 512]
M_MOT = 25          # motor-track categories the tally votes over
SUB = 150000        # cap on states used for tiling (all are used for the tally)
K, NB, EPOCHS, BATCH = 6, 12, 4, 2048
N_DEMO, MAX_STEPS, N_EVAL, SEED = 1500, 300, 300, 0
SIT_LO = np.array([-8., -8., -5., -5., -np.pi, -8.])
SIT_HI = np.array([8., 8., 5., 5., np.pi, 8.])
pc = PlaceCode(6, nb=NB, lo=SIT_LO, hi=SIT_HI, halfw=1.5)


def sense(s, tgt):
    return np.array([s[0] - tgt[0], s[1] - tgt[1], s[2], s[3], wrap(s[4]), s[5]])


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def dead(s):
    return abs(s[0]) > 12 or abs(s[1]) > 12 or abs(wrap(s[4])) > 1.4


def demos(rng, n_ep=N_DEMO):
    Sit, Act = [], []
    for _ in range(n_ep):
        s, tgt = any_init(rng)
        for _ in range(MAX_STEPS):
            lv = teacher(s, tgt)
            Sit.append(sense(s, tgt)); Act.append([LEVELS[lv[0]], LEVELS[lv[1]]])
            s = physics(s, lv)
            if at_goal(s, tgt) or dead(s):
                break
    return np.array(Sit), np.array(Act)


def tile(X, n_exp, rng):
    d = X.shape[1]
    W = rng.standard_normal((n_exp, K, d))
    W /= np.linalg.norm(W, axis=2, keepdims=True)
    f = np.full(n_exp, 1.0 / n_exp)
    for _ in range(EPOCHS):
        o = rng.permutation(len(X))
        for s in range(0, len(o), BATCH):
            B = X[o[s:s + BATCH]]
            if len(B) < 16:
                continue
            S = (B @ W.reshape(n_exp * K, d).T).reshape(len(B), n_exp, K).transpose(1, 0, 2)
            e = np.linalg.norm(B[None] - np.matmul(S, W), axis=2) \
                + 0.5 * (f - 1.0 / n_exp)[:, None]
            win = e.argmin(0)
            c = np.bincount(win, minlength=n_exp)
            f = 0.98 * f + 0.02 * (c / max(c.sum(), 1))
            for h in np.nonzero(c >= 3)[0]:
                W[h] = geo_step(W[h], B[win == h], 0.4)
    return W


def assign(W, Q):
    n_exp, k, d = W.shape
    out = np.empty(len(Q), dtype=np.int32)
    for s in range(0, len(Q), 4096):
        B = Q[s:s + 4096]
        S = (B @ W.reshape(n_exp * k, d).T).reshape(len(B), n_exp, k).transpose(1, 0, 2)
        out[s:s + 4096] = np.linalg.norm(B[None] - np.matmul(S, B * 0 + W[:, :, :] @ np.eye(d)
                                                             if False else np.matmul(S, W)),
                                         axis=2).argmin(0) if False else \
            np.linalg.norm(B[None] - np.matmul(S, W), axis=2).argmin(0)
    return out


def fly(policy, seed, n=N_EVAL):
    rng = np.random.default_rng(seed)
    hits, steps = 0, []
    for _ in range(n):
        s, tgt = any_init(rng)
        for t in range(MAX_STEPS):
            a = policy(sense(s, tgt))
            s = physics(s, (int(np.argmin(np.abs(LEVELS - a[0]))),
                            int(np.argmin(np.abs(LEVELS - a[1])))))
            if at_goal(s, tgt):
                hits += 1; break
            if dead(s):
                break
        steps.append(t + 1)
    return hits / n, float(np.mean(steps))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    Sit, Act = demos(rng)
    Q = unit(pc.encode(Sit))
    tsucc, tsteps = fly(lambda x: None, 0, 1) if False else (0, 0)
    tsucc, tsteps = fly(lambda sit: [LEVELS[i] for i in teacher(
        np.array([sit[0], sit[1], sit[2], sit[3], sit[4], sit[5]]), np.zeros(2))], 7)
    print(f"{len(Sit)} demonstration steps from {N_DEMO} teacher episodes")
    print(f"teacher itself: success {tsucc:.3f}, mean steps {tsteps:.1f}\n", flush=True)

    # ---- motor track: a vocabulary learned from what the teacher does ----
    pcm = PlaceCode(2, nb=NB, lo=0.0, hi=TMAX, halfw=1.5)
    Qm = unit(pcm.encode(Act))
    Wm = tile(Qm, M_MOT, np.random.default_rng(2))
    midx = assign(Wm, Qm)
    acts = np.stack([Act[midx == m].mean(0) if (midx == m).any() else
                     np.full(2, HOVER) for m in range(M_MOT)])
    print(f"motor track: {len(np.unique(midx))}/{M_MOT} categories used, "
          f"thrust {acts.min():.2f}-{acts.max():.2f} (hover {HOVER:.2f})\n", flush=True)

    sub = np.random.default_rng(3).choice(len(Q), min(SUB, len(Q)), replace=False)
    res = {}
    for n_exp in NS:
        W = tile(Q[sub], n_exp, np.random.default_rng(1))
        idx = assign(W, Q)
        used = len(np.unique(idx))
        # constant per cell
        const = np.zeros((n_exp, 2)); const[:] = HOVER
        # TALLY: how often each motor category was used in each sensory cell
        tally = np.zeros((n_exp, M_MOT))
        np.add.at(tally, (idx, midx), 1.0)
        # local linear per cell
        lin = [None] * n_exp
        for h in range(n_exp):
            m = idx == h
            if m.sum() >= 1:
                const[h] = Act[m].mean(0)
            if m.sum() >= 30:
                Xg = np.hstack([Sit[m], np.ones((m.sum(), 1))])
                lin[h] = np.linalg.lstsq(Xg, Act[m], rcond=None)[0]

        def make(kind):
            def pol(sit):
                q = unit(pc.encode(sit[None]))[0]
                S = (q @ W.reshape(n_exp * K, -1).T).reshape(n_exp, K)
                h = int(np.linalg.norm(q[None] - np.einsum('hk,hkd->hd', S, W),
                                       axis=1).argmin())
                if kind == "linear" and lin[h] is not None:
                    return np.clip(np.append(sit, 1.0) @ lin[h], 0, TMAX)
                if kind == "tally":
                    return np.clip(acts[int(tally[h].argmax())], 0, TMAX)
                return np.clip(const[h], 0, TMAX)
            return pol

        row = {}
        for kind in ("tally", "constant", "linear"):
            sc, st = fly(make(kind), 7)
            row[kind] = {"success": sc, "steps": st}
        row["cells_used"] = used
        row["cells_with_linear"] = int(sum(l is not None for l in lin))
        res[n_exp] = row
        print(f"  N={n_exp:>4}  cells used {used:>4}   "
              f"TALLY {row['tally']['success']:.3f}   "
              f"constant {row['constant']['success']:.3f}   "
              f"linear {row['linear']['success']:.3f}   "
              f"(steps {row['tally']['steps']:.0f}/{row['constant']['steps']:.0f}/"
              f"{row['linear']['steps']:.0f})   [{time.time()-t0:.0f}s]", flush=True)
        np.savez_compressed(OUT / f"imitate_{n_exp}.npz", W=W.astype(np.float32),
                            const=const,
                            lin=np.array([l if l is not None else np.zeros((7, 2))
                                          for l in lin]),
                            has_lin=np.array([l is not None for l in lin]),
                            tally=tally, acts=acts)
        (OUT / f"imitate_{n_exp}.json").write_text(json.dumps(
            {"kind": "module", "module": "imitate",
             "dir": "experiments/2026_08_31/drone_imitate",
             "n_exp": n_exp, "policy": "tally"}))

    (OUT / "metrics.json").write_text(json.dumps(
        {"teacher": {"success": tsucc, "steps": tsteps}, "by_n": res,
         "config": {"K": K, "nb": NB, "demo_episodes": N_DEMO,
                    "demo_steps": len(Sit)},
         "seconds": round(time.time() - t0, 1)}, indent=2))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(NS, [res[n]["tally"]["success"] for n in NS], "^-",
            color="#7a4fa3", lw=2, label="TALLY over motor categories")
    ax.plot(NS, [res[n]["constant"]["success"] for n in NS], "o-",
            color="#c1462d", label="constant action per cell")
    ax.plot(NS, [res[n]["linear"]["success"] for n in NS], "s-",
            color="#1b6ca8", label="local linear per cell")
    ax.axhline(tsucc, ls="--", color="#2f8f4e", label="the teacher itself")
    ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_xlabel("sensory experts"); ax.set_ylabel("success")
    ax.set_ylim(-.02, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("imitating the PID teacher", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "01_imitation.png", dpi=130); plt.close(fig)
    print(f"\ndone in {time.time()-t0:.0f}s")


def load_policy(npz, cfg):
    W, const, lin, has = (npz["W"].astype(np.float64), npz["const"],
                          npz["lin"], npz["has_lin"])
    n_exp = len(W)

    def act(s, tgt, lv_prev=None):
        sit = sense(s, tgt)
        q = unit(pc.encode(sit[None]))[0]
        S = (q @ W.reshape(n_exp * K, -1).T).reshape(n_exp, K)
        h = int(np.linalg.norm(q[None] - np.einsum('hk,hkd->hd', S, W), axis=1).argmin())
        pol = cfg.get("policy", "tally")
        if pol == "tally":
            a = npz["acts"][int(npz["tally"][h].argmax())]
        elif pol == "linear" and has[h]:
            a = np.append(sit, 1.0) @ lin[h]
        else:
            a = const[h]
        a = np.clip(a, 0, TMAX)
        return (int(np.argmin(np.abs(LEVELS - a[0]))), int(np.argmin(np.abs(LEVELS - a[1]))))
    return act


if __name__ == "__main__":
    main()
