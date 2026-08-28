"""Exp 18 — planar drone: three-track goal-conditioned control (user spec).

A 2D quadrotor (body in the vertical plane, two thrusters). Underactuated:
sideways motion only by tilting first. State 6-D, action 2-D coupled —
the first multi-actuator problem; the arm experiments' law (flat memory
pays exponentially in state dims; escape = factor the policy) is on
trial.

Tracks (all terminate at the top, per the user's architecture):
  proprio  foveated tilt ring x lagged slices [now,-4,-16] + efference
           echoes per thruster
  target   offset-to-goal (dx, dy) as foveated bumps, [now, -8] — the
           LABEL pattern promoted to control; goal-conditioned and
           translation-invariant by construction
  motor    one bipolar thermometer per thruster; arms:
           joint = one motor bank over the thrust PAIR (units name
                   combinations)
           fact  = one bank per thruster (units name marginals)

Top: K=4096 rows, sensory-only metric (action beside — tally), trained
by imitation of an LQR hover/waypoint teacher (discrete Riccati vs this
exact simulator) + DAgger.

Run:  .venv/bin/python experiments/2026_08_28/drone/run_drone.py
Env:  DR_DEMOS (400), DR_ROUNDS (3), DR_EVAL (100), DR_ARMS, DR_TAG
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
import run_recon3 as R                    # noqa: E402

OUTPUT_DIR = HERE / "results" / f"unified{os.environ.get('DR_TAG', '')}"
N_DEMOS = int(os.environ.get("DR_DEMOS", "400"))
ROUNDS = int(os.environ.get("DR_ROUNDS", "3"))
EVAL_N = int(os.environ.get("DR_EVAL", "100"))
ARMS = os.environ.get("DR_ARMS", "joint,fact").split(",")
ROLLOUTS = 50
EPOCHS = 2
EP_CAP = 800

# ---- physics -------------------------------------------------------------
M, GRAV, ARM_L, INERT, DT = 1.0, 9.8, 0.2, 0.02, 0.02
TMAX = 8.0
HOVER = GRAV * M / 2.0
# FOVEATED THRUST LEVELS: hover needs 4.9 per motor, which sat exactly
# between the old uniform levels (4.0 / 5.0) — every near-target moment
# was a decision boundary by construction (permanent bang-bang; tally
# mush as the default state). Now the middle level IS hover, with fine
# steps beside it and coarse ones at the extremes — the fovea principle
# on the action axis.
NLEV = 13
_lu = np.linspace(-1.0, 1.0, NLEV)
LEVELS = np.where(_lu < 0,
                  HOVER - np.abs(_lu) ** 1.7 * HOVER,
                  HOVER + np.abs(_lu) ** 1.7 * (TMAX - HOVER))
NB_F = NLEV - 1
THERMO = -np.ones((NLEV, NB_F), dtype=np.float32)
for k in range(NLEV):
    THERMO[k, :k] = 1.0
THERMO_N = THERMO / (np.linalg.norm(THERMO, axis=1, keepdims=True) + 1e-9)


def physics(s, lv):
    x, y, xd, yd, ph, phd = s
    t1, t2 = LEVELS[lv[0]], LEVELS[lv[1]]
    tt = t1 + t2
    xa = -tt * np.sin(ph) / M
    ya = tt * np.cos(ph) / M - GRAV
    pa = ARM_L * (t2 - t1) / INERT
    return np.array([x + DT * xd, y + DT * yd, xd + DT * xa,
                     yd + DT * ya, ph + DT * phd, phd + DT * pa])


def wrap(a):
    return ((a + np.pi) % (2 * np.pi)) - np.pi


# ---- LQR teacher ---------------------------------------------------------

def _lqr_gain():
    from scipy.linalg import solve_discrete_are
    # error state [ex, exd, ey, eyd, ph, phd], inputs [dT1, dT2]
    A = np.zeros((6, 6))
    A[0, 1] = 1.0
    A[1, 4] = -GRAV
    A[2, 3] = 1.0
    A[4, 5] = 1.0
    B = np.zeros((6, 2))
    B[3, 0] = B[3, 1] = 1.0 / M
    B[5, 0] = -ARM_L / INERT
    B[5, 1] = ARM_L / INERT
    Ad = np.eye(6) + DT * A
    Bd = DT * B
    Q = np.diag([1.5, 1.0, 1.5, 1.0, 4.0, 0.8])
    Rm = np.diag([0.1, 0.1])
    P = solve_discrete_are(Ad, Bd, Q, Rm)
    K = np.linalg.solve(Rm + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K


KLQR = _lqr_gain()


def teacher(s, tgt):
    e = np.array([s[0] - tgt[0], s[2], s[1] - tgt[1], s[3],
                  wrap(s[4]), s[5]])
    du = -KLQR @ e
    return tuple(int(np.argmin(np.abs(LEVELS - np.clip(HOVER + du[i],
                                                       0, TMAX))))
                 for i in range(2))


def any_init(rng):
    return (np.array([rng.uniform(-4, 4), rng.uniform(-4, 4),
                      rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5),
                      rng.uniform(-1.0, 1.0), rng.uniform(-1.5, 1.5)]),
            np.array([rng.uniform(-3, 3), rng.uniform(-3, 3)]))


def at_goal(s, tgt):
    return (np.hypot(s[0] - tgt[0], s[1] - tgt[1]) < 0.25
            and np.hypot(s[2], s[3]) < 0.3 and abs(wrap(s[4])) < 0.17)


def run_episode(policy, s0, tgt, cap=EP_CAP):
    s = s0.copy()
    hold = 0
    trace = []
    for t in range(cap):
        lv = policy(s, tgt, t)
        trace.append((s.copy(), lv))
        s = physics(s, lv)
        if abs(s[0]) > 30 or abs(s[1]) > 30:
            return False, trace
        if at_goal(s, tgt):
            hold += 1
            if hold >= 10:
                return True, trace
        else:
            hold = 0
    return False, trace


# ---- encodings -----------------------------------------------------------
NB_T, NB_O = 32, 24
LAGS_T, LAG_O = [4, 16], 8
MAXLAG = 16
_u = (np.arange(NB_T) + 0.5) / NB_T * 2 - 1
TILT_C = np.sign(_u) * np.pi * np.abs(_u) ** 1.7
_sp = np.abs(np.diff(np.concatenate([TILT_C, [TILT_C[0] + 2 * np.pi]])))
TILT_S = 1.2 * _sp
_v = (np.arange(NB_O) + 0.5) / NB_O * 2 - 1
OFF_C = np.sign(_v) * 6.0 * np.abs(_v) ** 1.7
_so = np.abs(np.gradient(OFF_C))
OFF_S = 1.3 * _so
GE = 0.5
ALPHA_EFF = 0.4
PDIM = 3 * NB_T + 2 * NB_F                 # tilt slices + two eff echoes
TDIM = 2 * 2 * NB_O                        # (dx, dy) x [now, -8]


def cn(v):
    v = np.asarray(v, dtype=np.float64).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def tilt_code(ph):
    d = wrap(ph - TILT_C)
    v = np.exp(-0.5 * (d / TILT_S) ** 2)
    return v / (np.linalg.norm(v) + 1e-9)


def off_code(o):
    d = np.clip(o, -6, 6) - OFF_C
    v = np.exp(-0.5 * (d / OFF_S) ** 2)
    return v / (np.linalg.norm(v) + 1e-9)


class Hist:
    def __init__(self):
        self.tb = None
        self.ob = None
        self.eff = np.zeros(2 * NB_F)

    def see(self, s, tgt):
        ph = wrap(s[4])
        dx, dy = s[0] - tgt[0], s[1] - tgt[1]
        if self.tb is None:
            self.tb = [ph] * (MAXLAG + 1)
            self.ob = [(dx, dy)] * (MAXLAG + 1)
        self.tb = (self.tb + [ph])[-(MAXLAG + 1):]
        self.ob = (self.ob + [(dx, dy)])[-(MAXLAG + 1):]

    def did(self, lv):
        f = np.concatenate([THERMO_N[lv[0]], THERMO_N[lv[1]]])
        self.eff = ALPHA_EFF * f + (1 - ALPHA_EFF) * self.eff

    def proprio(self):
        parts = [tilt_code(self.tb[-1])]
        parts += [tilt_code(self.tb[-1 - L]) for L in LAGS_T]
        parts.append(GE * self.eff)
        return cn(np.concatenate(parts))

    def target(self):
        dx, dy = self.ob[-1]
        px, py = self.ob[-1 - LAG_O]
        return cn(np.concatenate([off_code(dx), off_code(dy),
                                  off_code(px), off_code(py)]))


def motor_in(lv):
    return cn(np.concatenate([THERMO_N[lv[0]], THERMO_N[lv[1]]]))


# ---- model ---------------------------------------------------------------
K_P, K_T, K_MJ, K_MF, K_TOP = 192, 128, 48, 16, 4096


class Bank:
    def __init__(self, k, dim, seed):
        self.layer = R.Layer(k, dim, 0.05, np.random.default_rng(seed))

    def learn(self, x):
        self.layer.learn(x, self.layer.forward(x))

    def code(self, x):
        return np.maximum(self.layer.forward(x), 0.0)

    @property
    def W(self):
        return self.layer.W


def build_arm(arm, samples, seed):
    rng = np.random.default_rng(seed)
    bp = Bank(K_P, PDIM, seed)
    bt = Bank(K_T, TDIM, seed + 1)
    for _ in range(1):
        for i in rng.permutation(len(samples)):
            p, tg, _ = samples[i]
            bp.learn(p)
            bt.learn(tg)
    if arm == "joint":
        bm = [Bank(K_MJ, 2 * NB_F, seed + 2)]
        for l1 in range(NLEV):
            for l2 in range(NLEV):
                for _ in range(10):
                    bm[0].learn(motor_in((l1, l2)))
        mdim = K_MJ
    else:
        bm = [Bank(K_MF, NB_F, seed + 2), Bank(K_MF, NB_F, seed + 3)]
        for b, half in ((bm[0], 0), (bm[1], 1)):
            for lv in list(range(NLEV)) * 30:
                th = THERMO_N[lv]
                b.learn(cn(th))
        mdim = 2 * K_MF

    def skey(p, tg):
        return cn(np.concatenate([cn(bp.code(p)), cn(bt.code(tg))]))

    def mcode(lv):
        if arm == "joint":
            return cn(bm[0].code(motor_in(lv)))
        return np.concatenate([cn(bm[0].code(cn(THERMO_N[lv[0]]))),
                               cn(bm[1].code(cn(THERMO_N[lv[1]])))])

    top = R.Layer(K_TOP, K_P + K_T, 0.05, np.random.default_rng(seed + 4))
    for _ in range(EPOCHS):
        for i in rng.permutation(len(samples)):
            p, tg, lv = samples[i]
            top.learn(skey(p, tg), top.forward(skey(p, tg)))
    tally = np.zeros((K_TOP, mdim))
    cnt = np.zeros(K_TOP)
    for p, tg, lv in samples:
        w = int(np.argmax(top.forward(skey(p, tg))))
        tally[w] += mcode(lv)
        cnt[w] += 1
    tally /= np.maximum(cnt, 1)[:, None]
    return {"arm": arm, "bp": bp, "bt": bt, "bm": bm, "top": top,
            "tally": tally, "skey": skey}


def act(model, p, tg):
    mh = model["tally"][int(np.argmax(
        model["top"].forward(model["skey"](p, tg))))]
    bm = model["bm"]
    if model["arm"] == "joint":
        u = int(np.argmax(mh))
        tpl = bm[0].W[u]
        l1 = int(np.argmax(THERMO_N @ cn(tpl[:NB_F])))
        l2 = int(np.argmax(THERMO_N @ cn(tpl[NB_F:])))
    else:
        u1 = int(np.argmax(mh[:K_MF]))
        u2 = int(np.argmax(mh[K_MF:]))
        l1 = int(np.argmax(THERMO_N @ cn(bm[0].W[u1])))
        l2 = int(np.argmax(THERMO_N @ cn(bm[1].W[u2])))
    return (l1, l2)


def tf_agreement(model, data, rng, n=2500):
    """Per-thruster and exact-pair agreement with the teacher."""
    idx = rng.choice(len(data), size=min(n, len(data)), replace=False)
    per = pair = 0.0
    for i in idx:
        p, tg, lv = data[i]
        a = act(model, p, tg)
        per += (a[0] == lv[0]) + (a[1] == lv[1])
        pair += (a == tuple(lv))
    return round(per / (2 * len(idx)), 3), round(pair / len(idx), 3)


def save_model(m, tag, extra):
    np.savez(OUTPUT_DIR / f"{tag}.npz",
             Wtop=m["top"].W, Wp=m["bp"].W, Wt=m["bt"].W,
             Wm=np.concatenate([b.W.ravel() for b in m["bm"]]),
             tally=m["tally"])
    with open(OUTPUT_DIR / f"{tag}.json", "w") as f:
        json.dump({"arm": m["arm"], **extra}, f)


def make_policy(model):
    hist = Hist()

    def policy(s, tgt, t):
        if t == 0:
            hist.__init__()
        hist.see(s, tgt)
        lv = act(model, hist.proprio(), hist.target())
        hist.did(lv)
        return lv
    return policy


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    trng = np.random.default_rng(60)
    t_rate = np.mean([run_episode(lambda s, g, t: teacher(s, g),
                                  *any_init(trng))[0] for _ in range(60)])
    print(f"LQR teacher rate: {t_rate:.2f}", flush=True)
    with open(OUTPUT_DIR / "expert.json", "w") as f:
        json.dump({"K": KLQR.tolist(), "rate": float(t_rate)}, f)

    samples = []
    n = 0
    while n < N_DEMOS:
        s0, tgt = any_init(rng)
        ok, trace = run_episode(lambda s, g, t: teacher(s, g), s0, tgt)
        if not ok:
            continue
        n += 1
        hist = Hist()
        for s, lv in trace:
            hist.see(s, tgt)
            samples.append((hist.proprio(), hist.target(), lv))
            hist.did(lv)
        # hover tail with gusts near the goal
        s_c = physics(trace[-1][0], trace[-1][1])
        for _ in range(80):
            if rng.random() < 0.06:
                s_c[2] += rng.uniform(-1.2, 1.2)
                s_c[3] += rng.uniform(-1.2, 1.2)
            hist.see(s_c, tgt)
            lv = teacher(s_c, tgt)
            samples.append((hist.proprio(), hist.target(), lv))
            hist.did(lv)
            s_c = physics(s_c, lv)
    print(f"demos {N_DEMOS}, samples {len(samples)}", flush=True)

    res = {"n_demos": N_DEMOS, "teacher_rate": float(t_rate),
           "nlev": NLEV, "k": [K_P, K_T, K_TOP]}

    # ---- controls: is the encoding sufficient? (the pole playbook) -----
    from sklearn.neural_network import MLPClassifier
    crng = np.random.default_rng(3)
    sub = crng.choice(len(samples), size=min(40000, len(samples)),
                      replace=False)
    Xc = np.array([np.concatenate([samples[i][0], samples[i][1]])
                   for i in sub], np.float32)
    cut = int(0.8 * len(Xc))
    for th in (0, 1):
        yc = np.array([samples[i][2][th] for i in sub])
        mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200,
                            random_state=0)
        mlp.fit(Xc[:cut], yc[:cut])
        acc = float(mlp.score(Xc[cut:], yc[cut:]))
        hits = 0
        nn_tr = Xc[:4000]
        for i in range(cut, min(cut + 1500, len(Xc))):
            j = int(np.argmax(nn_tr @ Xc[i]))
            hits += int(yc[j] == yc[i])
        res[f"mlp_acc_t{th}"] = round(acc, 3)
        res[f"nn1_acc_t{th}"] = round(hits / 1500, 3)
        print(f"controls thruster {th}: MLP {acc:.3f}  "
              f"1-NN {hits / 1500:.3f}", flush=True)
    best = {"score": -1}
    for arm in ARMS:
        data = list(samples)
        for rnd in range(ROUNDS):
            model = build_arm(arm, data, seed=5 + rnd)
            save_model(model, "checkpoint",
                       {"round": rnd, "score": "evaluating"})
            per, pair = tf_agreement(model, data,
                                     np.random.default_rng(rnd))
            vrng = np.random.default_rng(100 + rnd)
            oks = sum(int(run_episode(make_policy(model),
                                      *any_init(vrng))[0])
                      for _ in range(EVAL_N))
            res[f"{arm}_r{rnd}"] = oks
            res[f"{arm}_r{rnd}_tf"] = [per, pair]
            save_model(model, "checkpoint",
                       {"round": rnd, "score": oks, "tf": per})
            print(f"[{arm}] round {rnd}: {oks}/{EVAL_N} tf {per}/{pair} "
                  f"(data {len(data)})", flush=True)
            with open(OUTPUT_DIR / "metrics.json", "w") as f:
                json.dump(res, f, indent=2)
            if oks > best["score"]:
                best = {"score": oks, "arm": arm, "round": rnd,
                        "model": model}
            if rnd == ROUNDS - 1:
                break
            crng = np.random.default_rng(200 + rnd)
            for _ in range(ROLLOUTS):
                s, tgt = any_init(crng)
                hist = Hist()
                for t in range(EP_CAP):
                    hist.see(s, tgt)
                    data.append((hist.proprio(), hist.target(),
                                 teacher(s, tgt)))
                    lv = act(model, hist.proprio(), hist.target())
                    hist.did(lv)
                    s = physics(s, lv)
                    if abs(s[0]) > 30 or abs(s[1]) > 30 or at_goal(s, tgt):
                        break

    m = best["model"]
    save_model(m, "weights_best",
               {"round": best["round"], "score": best["score"]})
    (OUTPUT_DIR / "config_best.json").write_text(
        (OUTPUT_DIR / "weights_best.json").read_text())
    res["best"] = {k: v for k, v in best.items() if k != "model"}
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
