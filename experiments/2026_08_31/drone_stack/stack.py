"""Two sparse tracks and an L2 that learns their pairing. stack3, for control.

    sensory track   N hypercolumns over the 6 sensors  -> sparse code (top-k,
                    valued by confidence; no dense subspace values leave it)
    motor track     M hypercolumns over the 2 thrusts  -> sparse code, same
    L2              templates over [sensory code ; motor code], winner-take-all,
                    trained on stored PID flight data

Flying is the partial-cue read: write the sensory block, leave the motor block
blank, find the L2 template that best matches what IS written, and read its
motor block back out.

The codes are top-k GRADED, not one-hot. A hard winner would leave only
N x M distinguishable inputs and L2 would degenerate into the tally that
already scored 0.000. The graded code carries where BETWEEN hypercolumns the
state sits, which is the sub-cell resolution a hard winner throws away.
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
N_SENS, K_SENS, M_MOT, K_MOT, NB = 256, 6, 64, 3, 12
KS, KM, K3, RHO, ETA3 = 4, 3, 4096, 1.0, 0.25
READ_M = 8
N_DEMO, MAX_STEPS, N_EVAL, SEED = 1200, 300, 250, 0
SIT_LO = np.array([-8., -8., -5., -5., -np.pi, -8.])
SIT_HI = np.array([8., 8., 5., 5., np.pi, 8.])
pcs = PlaceCode(6, nb=NB, lo=SIT_LO, hi=SIT_HI, halfw=1.5)
pcm = PlaceCode(2, nb=NB, lo=0.0, hi=TMAX, halfw=1.5)


def sense(s, tgt):
    return np.array([s[0] - tgt[0], s[1] - tgt[1], s[2], s[3], wrap(s[4]), s[5]])


def unit(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, EPS)


def dead(s):
    return abs(s[0]) > 12 or abs(s[1]) > 12 or abs(wrap(s[4])) > 1.4


def demos(rng, n_ep=N_DEMO):
    S, A = [], []
    for _ in range(n_ep):
        s, tgt = any_init(rng)
        for _ in range(MAX_STEPS):
            lv = teacher(s, tgt)
            S.append(sense(s, tgt)); A.append([LEVELS[lv[0]], LEVELS[lv[1]]])
            s = physics(s, lv)
            if at_goal(s, tgt) or dead(s):
                break
    return np.array(S), np.array(A)


def tile(X, n_exp, k, rng, epochs=4, batch=2048, eta=0.4):
    d = X.shape[1]
    W = rng.standard_normal((n_exp, k, d))
    W /= np.linalg.norm(W, axis=2, keepdims=True)
    f = np.full(n_exp, 1.0 / n_exp)
    for _ in range(epochs):
        o = rng.permutation(len(X))
        for s in range(0, len(o), batch):
            B = X[o[s:s + batch]]
            if len(B) < 16:
                continue
            S = (B @ W.reshape(n_exp * k, d).T).reshape(len(B), n_exp, k).transpose(1, 0, 2)
            e = np.linalg.norm(B[None] - np.matmul(S, W), axis=2) \
                + 0.5 * (f - 1.0 / n_exp)[:, None]
            win = e.argmin(0)
            c = np.bincount(win, minlength=n_exp)
            f = 0.98 * f + 0.02 * (c / max(c.sum(), 1))
            for h in np.nonzero(c >= 3)[0]:
                W[h] = geo_step(W[h], B[win == h], eta)
    return W


def sparse_code(W, Q, topk, chunk=8192):
    """(n, n_exp) -- top-k hypercolumns, valued by how well each explains."""
    n_exp, k, d = W.shape
    out = np.zeros((len(Q), n_exp), dtype=np.float32)
    for s in range(0, len(Q), chunk):
        B = Q[s:s + chunk]
        S = (B @ W.reshape(n_exp * k, d).T).reshape(len(B), n_exp, k).transpose(1, 0, 2)
        e = np.linalg.norm(B[None] - np.matmul(S, W), axis=2).T          # (n, n_exp)
        idx = np.argpartition(e, topk, axis=1)[:, :topk]
        v = np.take_along_axis(e, idx, 1)
        w = np.maximum(v[:, -1:] - v, 0) + 1e-3                          # graded weights
        np.put_along_axis(out[s:s + chunk], idx, (w / w.sum(1, keepdims=True)).astype(np.float32), 1)
    return out


def join(Cs, Cm=None):
    V = np.zeros((len(Cs), N_SENS + M_MOT))
    V[:, :N_SENS] = Cs
    if Cm is not None:
        g = RHO * np.linalg.norm(Cs, axis=1) / np.maximum(np.linalg.norm(Cm, axis=1), EPS)
        V[:, N_SENS:] = Cm * g[:, None]
    return unit(V)


def train_l2(X, rng, epochs=3, batch=512):
    W = np.zeros((K3, X.shape[1]))
    W[:K3] = X[rng.permutation(len(X))[:K3]] + 0.01 * rng.standard_normal((K3, X.shape[1]))
    W = unit(W)
    for _ in range(epochs):
        o = rng.permutation(len(X))
        for s in range(0, len(o), batch):
            B = X[o[s:s + batch]]
            win = (B @ W.T).argmax(1)
            c = np.bincount(win, minlength=K3)
            for h in np.nonzero(c >= 2)[0]:
                W[h] = geo_step(W[h:h + 1], B[win == h], ETA3)[0]
    return W


def make_policy(Ws, Wm, W3, acts):
    sub = W3[:, :N_SENS]
    nrm = np.linalg.norm(sub, axis=1)
    floor = 0.25 * nrm.max()

    def pol(sit):
        q = unit(pcs.encode(sit[None]))
        cs = sparse_code(Ws, q, KS)
        qq = unit(np.hstack([cs, np.zeros((1, M_MOT))]))[0]
        sc = (sub @ qq[:N_SENS]) / np.maximum(nrm, floor)
        top = np.argpartition(-sc, READ_M)[:READ_M]
        g = np.maximum(sc[top], 0) + 1e-9
        mot = (g / g.sum()) @ W3[top][:, N_SENS:]                # blended motor code
        mot = np.maximum(mot, 0)
        if mot.sum() < EPS:
            return np.full(2, HOVER)
        return np.clip((mot / mot.sum()) @ acts, 0, TMAX)
    return pol


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
    Qs, Qm = unit(pcs.encode(Sit)), unit(pcm.encode(Act))
    print(f"{len(Sit)} demo steps", flush=True)

    Ws = tile(Qs, N_SENS, K_SENS, np.random.default_rng(1))
    Wm = tile(Qm, M_MOT, K_MOT, np.random.default_rng(2))
    Cs, Cm = sparse_code(Ws, Qs, KS), sparse_code(Wm, Qm, KM)
    hard = sparse_code(Wm, Qm, 1).argmax(1)
    acts = np.stack([Act[hard == m].mean(0) if (hard == m).any() else np.full(2, HOVER)
                     for m in range(M_MOT)])
    print(f"tracks: sensory {len(np.unique(sparse_code(Ws,Qs,1).argmax(1)))}/{N_SENS} used, "
          f"motor {len(np.unique(hard))}/{M_MOT} used   [{time.time()-t0:.0f}s]", flush=True)

    X = join(Cs, Cm)
    W3 = train_l2(X, np.random.default_rng(3))
    print(f"L2: {K3} templates trained   [{time.time()-t0:.0f}s]", flush=True)

    sc, st = fly(make_policy(Ws, Wm, W3, acts), 7)
    tsc, tst = fly(lambda sit: [LEVELS[i] for i in teacher(
        np.array(list(sit)), np.zeros(2))], 7)
    print(f"\n  two tracks + L2 partial-cue read : success {sc:.3f}  steps {st:.1f}")
    print(f"  teacher                          : success {tsc:.3f}  steps {tst:.1f}")
    print(f"  (one global linear map, no arch. : success 0.875)")

    np.savez_compressed(OUT / "stack.npz", Ws=Ws.astype(np.float32),
                        Wm=Wm.astype(np.float32), W3=W3.astype(np.float32), acts=acts)
    (OUT / "stack.json").write_text(json.dumps(
        {"kind": "module", "module": "stack",
         "dir": "experiments/2026_08_31/drone_stack"}))
    (OUT / "metrics.json").write_text(json.dumps(
        {"config": {"n_sens": N_SENS, "m_mot": M_MOT, "k3": K3, "ks": KS,
                    "km": KM, "read_m": READ_M, "rho": RHO},
         "success": sc, "steps": st, "teacher": tsc,
         "global_linear_baseline": 0.875,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"done in {time.time()-t0:.0f}s")


def load_policy(npz, cfg):
    Ws, Wm = npz["Ws"].astype(np.float64), npz["Wm"].astype(np.float64)
    W3, acts = npz["W3"].astype(np.float64), npz["acts"]
    pol = make_policy(Ws, Wm, W3, acts)

    def act(s, tgt, lv_prev=None):
        a = pol(sense(s, tgt))
        return (int(np.argmin(np.abs(LEVELS - a[0]))), int(np.argmin(np.abs(LEVELS - a[1]))))
    return act


if __name__ == "__main__":
    main()
