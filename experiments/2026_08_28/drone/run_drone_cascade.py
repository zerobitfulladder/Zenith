"""Exp 19 — cascaded drone: inner attitude loop + outer position loop.

The curse-of-dimensionality escape, serial form: two small memories in
a command chain instead of one 6-D memory.

  OUTER (position): sees offset-to-target [now, -8] only.
         Emits an INSTRUCTION: desired lean (9 foveated levels, +-0.6
         rad) + collective thrust (the 13 foveated hover levels).
  INNER (attitude): sees tilt [now,-4,-16] + the commanded lean (a bump
         on the same foveated ring — the goal channel, label-style) +
         differential-efference echo. Emits differential thrust
         (9 foveated levels, +-2.5, middle = exactly zero).

Thrusters: t = collective -/+ differential, quantized to motor levels.
Teacher: a CASCADED PD autopilot (position->lean, attitude->diff,
height->collective), gain-auditioned, quantized through the same
ladders the pupil uses. Both loops trained beside-the-metric with
mode-read command banks; DAgger relabels both loops; TF + MLP/1-NN
controls per loop; viewer-loadable checkpoint each round.

Run:  .venv/bin/python experiments/2026_08_28/drone/run_drone_cascade.py
Env:  DC_DEMOS (400), DC_ROUNDS (3), DC_EVAL (100), DC_TAG
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
import run_drone as D                     # noqa: E402

OUTPUT_DIR = HERE / "results" / f"cascade{os.environ.get('DC_TAG', '')}"
N_DEMOS = int(os.environ.get("DC_DEMOS", "400"))
ROUNDS = int(os.environ.get("DC_ROUNDS", "3"))
EVAL_N = int(os.environ.get("DC_EVAL", "100"))
ROLLOUTS = 50
EP_CAP = 800
# Association form: "tally" (rows sensory-only, action tallied beside)
# or "bound" (user's vision: rows = [raw sensor channels ; GM_B * motor
# code], top-1 learning, DENSE projection of all rows' motor halves at
# read time, the motor track's own argmax commits). No sensory banks in
# bound mode — bump populations go to the top directly.
ASSOC = os.environ.get("DC_ASSOC", "tally")
GM_B = 0.4

# ---- command ladders (fovea on every action axis) ------------------------
PHIMAX, DMAX = 0.6, 2.5
N_P = N_D = 9


def _fov(n, span):
    u = np.linspace(-1.0, 1.0, n)
    return np.sign(u) * np.abs(u) ** 1.7 * span


PLEV = _fov(N_P, PHIMAX)                   # desired-lean levels, 0 centered
DLEV = _fov(N_D, DMAX)                     # differential levels, 0 centered
CLEV = D.LEVELS                            # collective (hover centered)


def _thermo(n):
    t = -np.ones((n, n - 1), dtype=np.float32)
    for k in range(n):
        t[k, :k] = 1.0
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)


TH_P, TH_D, TH_C = _thermo(N_P), _thermo(N_D), _thermo(D.NLEV)


def q(levels, v):
    return int(np.argmin(np.abs(levels - v)))


# ---- cascaded teacher ----------------------------------------------------

def make_cascade_teacher(kpx, kdx, ka, kda, ky=1.2, kvy=1.8):
    def teacher(s, tgt):
        ex, ey = s[0] - tgt[0], s[1] - tgt[1]
        ph = D.wrap(s[4])
        lp = q(PLEV, np.clip(kpx * ex + kdx * s[2], -PHIMAX, PHIMAX))
        lc = q(CLEV, np.clip(D.HOVER - ky * ey - kvy * s[3], 0.3, D.TMAX))
        ld = q(DLEV, np.clip(ka * (PLEV[lp] - ph) - kda * s[5],
                             -DMAX, DMAX))
        return lp, lc, ld
    return teacher


def levels_from(lp, lc, ld):
    t1 = np.clip(CLEV[lc] - DLEV[ld], 0, D.TMAX)
    t2 = np.clip(CLEV[lc] + DLEV[ld], 0, D.TMAX)
    return (q(CLEV, t1), q(CLEV, t2))


def teacher_policy(tch):
    def pol(s, tgt, t):
        return levels_from(*tch(s, tgt))
    return pol


# ---- encodings -----------------------------------------------------------
EFF_B = N_D - 1
IDIM = 4 * D.NB_T + EFF_B                  # tilt x3 + goal bump + eff echo
# Outer sensory: foveated offset bumps + EXPLICIT velocity populations.
# The lagged-foveated-position difference was measured unreadable off
# the fovea (far-field bumps ~1 m wide vs a 0.24 m velocity baseline).
# Same information (finite difference of the offset history), re-encoded
# in uniform bumps: speed equally readable everywhere.
NB_V = 16
VEL_C = np.linspace(-3.0, 3.0, NB_V)
VEL_S = 1.3 * (VEL_C[1] - VEL_C[0])
ODIM = 2 * D.NB_O + 2 * NB_V


def vel_code(v):
    d = np.clip(v, -3.0, 3.0) - VEL_C
    x = np.exp(-0.5 * (d / VEL_S) ** 2)
    return x / (np.linalg.norm(x) + 1e-9)


class CHist:
    def __init__(self):
        self.tb = None
        self.ob = None
        self.eff = np.zeros(EFF_B)

    def see(self, s, tgt):
        ph = D.wrap(s[4])
        dx, dy = s[0] - tgt[0], s[1] - tgt[1]
        if self.tb is None:
            self.tb = [ph] * (D.MAXLAG + 1)
            self.ob = [(dx, dy)] * (D.MAXLAG + 1)
        self.tb = (self.tb + [ph])[-(D.MAXLAG + 1):]
        self.ob = (self.ob + [(dx, dy)])[-(D.MAXLAG + 1):]

    def did(self, ld):
        self.eff = 0.4 * TH_D[ld] + 0.6 * self.eff

    def outer_code(self):
        dx, dy = self.ob[-1]
        px, py = self.ob[-1 - D.LAG_O]
        dt = D.LAG_O * D.DT
        return D.cn(np.concatenate([D.off_code(dx), D.off_code(dy),
                                    vel_code((dx - px) / dt),
                                    vel_code((dy - py) / dt)]))

    def inner_code(self, phi_des):
        parts = [D.tilt_code(self.tb[-1])]
        parts += [D.tilt_code(self.tb[-1 - L]) for L in D.LAGS_T]
        parts.append(D.tilt_code(phi_des))
        parts.append(0.5 * self.eff)
        return D.cn(np.concatenate(parts))


# ---- model ---------------------------------------------------------------
K_BO, K_BI, K_CMD = 128, 192, 16
K_OUT, K_IN = 1024, 1024
# Sliced-read channel groups over the raw outer code
# [dx 0:24 | dy 24:48 | vx 48:64 | vy 64:80]: the drone's physics
# splits laterally/vertically, so the lateral slice reads the lean
# piece and the vertical slice reads the collective piece.
LAT_IDX = np.concatenate([np.arange(0, 24), np.arange(48, 64)])
VERT_IDX = np.concatenate([np.arange(24, 48), np.arange(64, 80)])


def build(samples_o, samples_i, seed):
    # Memory grows with experience: hold moments-per-row roughly
    # constant instead of diluting a fixed bank as DAgger adds data
    # (measured: fixed K -> tf 0.75 -> 0.62 across rounds).
    global K_OUT, K_IN
    K_OUT = int(min(6144, max(1024, len(samples_o) // 60)))
    K_IN = int(min(6144, max(1024, len(samples_i) // 60)))
    if ASSOC != "tally":
        return build_bound(samples_o, samples_i, seed)
    rng = np.random.default_rng(seed)
    bo = D.Bank(K_BO, ODIM, seed)
    bi = D.Bank(K_BI, IDIM, seed + 1)
    for i in rng.permutation(len(samples_o)):
        bo.learn(samples_o[i][0])
    for i in rng.permutation(len(samples_i)):
        bi.learn(samples_i[i][0])
    cb = {"p": D.Bank(K_CMD, N_P - 1, seed + 2),
          "c": D.Bank(K_CMD, D.NLEV - 1, seed + 3),
          "d": D.Bank(K_CMD, N_D - 1, seed + 4)}
    for _ in range(30):
        for l in range(N_P):
            cb["p"].learn(D.cn(TH_P[l]))
        for l in range(D.NLEV):
            cb["c"].learn(D.cn(TH_C[l]))
        for l in range(N_D):
            cb["d"].learn(D.cn(TH_D[l]))

    def okey(x):
        return D.cn(bo.code(x))

    def ikey(x):
        return D.cn(bi.code(x))

    t_out = R.Layer(K_OUT, K_BO, 0.05, np.random.default_rng(seed + 5))
    t_in = R.Layer(K_IN, K_BI, 0.05, np.random.default_rng(seed + 6))
    for _ in range(2):
        for i in rng.permutation(len(samples_o)):
            k = okey(samples_o[i][0])
            t_out.learn(k, t_out.forward(k))
        for i in rng.permutation(len(samples_i)):
            k = ikey(samples_i[i][0])
            t_in.learn(k, t_in.forward(k))
    ty_out = np.zeros((K_OUT, 2 * K_CMD))
    co = np.zeros(K_OUT)
    for x, lp, lc in samples_o:
        w = int(np.argmax(t_out.forward(okey(x))))
        ty_out[w] += np.concatenate([D.cn(cb["p"].code(D.cn(TH_P[lp]))),
                                     D.cn(cb["c"].code(D.cn(TH_C[lc])))])
        co[w] += 1
    ty_out /= np.maximum(co, 1)[:, None]
    ty_in = np.zeros((K_IN, K_CMD))
    ci = np.zeros(K_IN)
    for x, ld in samples_i:
        w = int(np.argmax(t_in.forward(ikey(x))))
        ty_in[w] += D.cn(cb["d"].code(D.cn(TH_D[ld])))
        ci[w] += 1
    ty_in /= np.maximum(ci, 1)[:, None]
    return {"assoc": "tally", "bo": bo, "bi": bi, "cb": cb,
            "t_out": t_out, "t_in": t_in,
            "ty_out": ty_out, "ty_in": ty_in, "okey": okey, "ikey": ikey}


def build_bound(samples_o, samples_i, seed):
    """User's arm: bound rows [sensor key ; motor code]; top-1 learning.
    ASSOC=bound:    raw channels, DENSE projection read (ran: 0/100).
    ASSOC=bound_t1: sensory banks RESTORED (de-confounded) + TOP-1
                    read — the winner row's own stored motor half
                    projects to the motor track (user's fix)."""
    rng = np.random.default_rng(seed)
    banked = ASSOC == "bound_t1"          # sliced runs on RAW channels
                                          # (bank units mix channels —
                                          # slicing needs channel identity)
    bo = bi = None
    if banked:
        bo = D.Bank(K_BO, ODIM, seed)
        bi = D.Bank(K_BI, IDIM, seed + 1)
        for i in rng.permutation(len(samples_o)):
            bo.learn(samples_o[i][0])
        for i in rng.permutation(len(samples_i)):
            bi.learn(samples_i[i][0])
    cb = {"p": D.Bank(K_CMD, N_P - 1, seed + 2),
          "c": D.Bank(K_CMD, D.NLEV - 1, seed + 3),
          "d": D.Bank(K_CMD, N_D - 1, seed + 4)}
    for _ in range(30):
        for l in range(N_P):
            cb["p"].learn(D.cn(TH_P[l]))
        for l in range(D.NLEV):
            cb["c"].learn(D.cn(TH_C[l]))
        for l in range(N_D):
            cb["d"].learn(D.cn(TH_D[l]))

    def mo_code(lp, lc):
        return np.concatenate([D.cn(cb["p"].code(D.cn(TH_P[lp]))),
                               D.cn(cb["c"].code(D.cn(TH_C[lc])))])

    def mi_code(ld):
        return D.cn(cb["d"].code(D.cn(TH_D[ld])))

    okey = (lambda x: D.cn(bo.code(x))) if banked else (lambda x: x)
    ikey = (lambda x: D.cn(bi.code(x))) if banked else (lambda x: x)
    kod = K_BO if banked else ODIM
    kid = K_BI if banked else IDIM
    t_out = R.Layer(K_OUT, kod + 2 * K_CMD, 0.05,
                    np.random.default_rng(seed + 5))
    t_in = R.Layer(K_IN, kid + K_CMD, 0.05,
                   np.random.default_rng(seed + 6))
    for _ in range(2):
        for i in rng.permutation(len(samples_o)):
            x, lp, lc = samples_o[i]
            z = D.cn(np.concatenate([okey(x), GM_B * mo_code(lp, lc)]))
            t_out.learn(z, t_out.forward(z))
        for i in rng.permutation(len(samples_i)):
            x, ld = samples_i[i]
            z = D.cn(np.concatenate([ikey(x), GM_B * mi_code(ld)]))
            t_in.learn(z, t_in.forward(z))
    return {"assoc": ASSOC, "bo": bo, "bi": bi, "cb": cb,
            "t_out": t_out, "t_in": t_in, "ty_out": None, "ty_in": None,
            "okey": okey, "ikey": ikey, "kod": kod, "kid": kid}


def _mode(bank, code, thermo):
    u = int(np.argmax(code))
    return int(np.argmax(thermo @ (bank.W[u] / (np.linalg.norm(bank.W[u])
                                                + 1e-9))))


def inner_ld(m, x):
    """Differential level from an inner-loop sensory code."""
    if m["assoc"] != "tally":
        q = D.cn(np.concatenate([m["ikey"](x), np.zeros(K_CMD)]))
        if m["assoc"] in ("bound_t1", "sliced"):
            row = m["t_in"].W[int(np.argmax(m["t_in"].forward(q)))]
            mi = row[m["kid"]:]
        else:
            s = np.maximum(m["t_in"].forward(q), 0.0)
            mi = s @ m["t_in"].W[:, m["kid"]:]
        return _mode(m["cb"]["d"], mi, TH_D)
    ti = m["ty_in"][int(np.argmax(m["t_in"].forward(m["ikey"](x))))]
    return _mode(m["cb"]["d"], ti, TH_D)


def cascade_act(m, hist):
    if m["assoc"] == "sliced":
        # Two competitions, one per physics slice; each winner
        # contributes only the command piece it was consulted about.
        x = hist.outer_code()
        W = m["t_out"].W
        kod = m["kod"]
        lp = lc = None
        for idx, piece in ((LAT_IDX, "p"), (VERT_IDX, "c")):
            qs = D.cn(x[idx])
            Ws = W[:, idx]
            sims = (Ws @ qs) / (np.linalg.norm(Ws, axis=1) + 1e-9)
            wrow = W[int(np.argmax(sims))]
            if piece == "p":
                lp = _mode(m["cb"]["p"], wrow[kod:kod + K_CMD], TH_P)
            else:
                lc = _mode(m["cb"]["c"], wrow[kod + K_CMD:], TH_C)
        ld = inner_ld(m, hist.inner_code(PLEV[lp]))
        return lp, lc, ld
    if m["assoc"].startswith("bound"):
        q = D.cn(np.concatenate([m["okey"](hist.outer_code()),
                                 np.zeros(2 * K_CMD)]))
        if m["assoc"] == "bound_t1":
            row = m["t_out"].W[int(np.argmax(m["t_out"].forward(q)))]
            mo = row[m["kod"]:]
        else:
            s = np.maximum(m["t_out"].forward(q), 0.0)
            mo = s @ m["t_out"].W[:, m["kod"]:]
        lp = _mode(m["cb"]["p"], mo[:K_CMD], TH_P)
        lc = _mode(m["cb"]["c"], mo[K_CMD:], TH_C)
    else:
        to = m["ty_out"][int(np.argmax(m["t_out"].forward(
            m["okey"](hist.outer_code()))))]
        lp = _mode(m["cb"]["p"], to[:K_CMD], TH_P)
        lc = _mode(m["cb"]["c"], to[K_CMD:], TH_C)
    ld = inner_ld(m, hist.inner_code(PLEV[lp]))
    return lp, lc, ld


def make_policy(m):
    hist = CHist()

    def pol(s, tgt, t):
        if t == 0:
            hist.__init__()
        hist.see(s, tgt)
        lp, lc, ld = cascade_act(m, hist)
        hist.did(ld)
        return levels_from(lp, lc, ld)
    return pol


def save_model(m, tag, extra):
    z1 = np.zeros((1, 1))
    np.savez(OUTPUT_DIR / f"{tag}.npz",
             Wbo=(m["bo"].W if m["bo"] is not None else z1),
             Wbi=(m["bi"].W if m["bi"] is not None else z1),
             Wcp=m["cb"]["p"].W, Wcc=m["cb"]["c"].W, Wcd=m["cb"]["d"].W,
             Wto=m["t_out"].W, Wti=m["t_in"].W,
             tyo=(m["ty_out"] if m["ty_out"] is not None else z1),
             tyi=(m["ty_in"] if m["ty_in"] is not None else z1))
    with open(OUTPUT_DIR / f"{tag}.json", "w") as f:
        json.dump({"cascade": True, "assoc": m["assoc"], **extra}, f)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    best_t, best_ok = None, -1
    for kpx in (0.15, 0.3):
        for kdx in (0.3, 0.5):
            for ka in (2.0, 4.0):
                for kda in (0.6, 1.2):
                    tch = make_cascade_teacher(kpx, kdx, ka, kda)
                    ok = sum(int(D.run_episode(teacher_policy(tch),
                                               *D.any_init(
                        np.random.default_rng(50)))[0]) for _ in range(12))
                    if ok > best_ok:
                        best_t, best_ok = (kpx, kdx, ka, kda), ok
    tch = make_cascade_teacher(*best_t)
    erng = np.random.default_rng(60)
    t_rate = np.mean([D.run_episode(teacher_policy(tch),
                                    *D.any_init(erng))[0]
                      for _ in range(50)])
    print(f"cascade teacher {best_t} audition {best_ok}/12 rate {t_rate:.2f}",
          flush=True)
    with open(OUTPUT_DIR / "expert.json", "w") as f:
        json.dump({"cascade": True, "gains": best_t,
                   "rate": float(t_rate)}, f)

    def collect(s, tgt, hist, so, si):
        hist.see(s, tgt)
        lp, lc, ld = tch(s, tgt)
        so.append((hist.outer_code(), lp, lc))
        si.append((hist.inner_code(PLEV[lp]), ld))
        hist.did(ld)
        return levels_from(lp, lc, ld)

    so, si = [], []
    n = 0
    while n < N_DEMOS:
        s0, tgt = D.any_init(rng)
        ok, trace = D.run_episode(teacher_policy(tch), s0, tgt)
        if not ok:
            continue
        n += 1
        hist = CHist()
        s = s0.copy()
        for st, _ in trace:
            lv = collect(st, tgt, hist, so, si)
        s_c = D.physics(trace[-1][0], trace[-1][1])
        for _ in range(80):
            if rng.random() < 0.06:
                s_c[2] += rng.uniform(-1.2, 1.2)
                s_c[3] += rng.uniform(-1.2, 1.2)
            lv = collect(s_c, tgt, hist, so, si)
            s_c = D.physics(s_c, lv)
    print(f"demos {N_DEMOS}: outer {len(so)}, inner {len(si)}", flush=True)

    res = {"teacher_rate": float(t_rate), "gains": best_t}

    # controls per loop (low-D vindication check)
    from sklearn.neural_network import MLPClassifier
    crng = np.random.default_rng(3)
    for name, data, ycol in (("outer_phi", so, 1), ("inner_diff", si, 1)):
        sub = crng.choice(len(data), size=min(30000, len(data)),
                          replace=False)
        X = np.array([data[i][0] for i in sub], np.float32)
        y = np.array([data[i][ycol] for i in sub])
        cut = int(0.8 * len(X))
        mlp = MLPClassifier((64,), max_iter=200, random_state=0)
        mlp.fit(X[:cut], y[:cut])
        acc = float(mlp.score(X[cut:], y[cut:]))
        hits = 0
        for i in range(cut, min(cut + 1500, len(X))):
            j = int(np.argmax(X[:4000] @ X[i]))
            hits += int(y[j] == y[i])
        res[f"mlp_{name}"] = round(acc, 3)
        res[f"nn1_{name}"] = round(hits / 1500, 3)
        print(f"controls {name}: MLP {acc:.3f} 1-NN {hits / 1500:.3f}",
              flush=True)

    best = {"score": -1}
    d_o, d_i = list(so), list(si)
    for rnd in range(ROUNDS):
        model = build(d_o, d_i, seed=5 + rnd)
        save_model(model, "checkpoint", {"round": rnd,
                                         "score": "evaluating"})
        trng2 = np.random.default_rng(rnd)
        idx = trng2.choice(len(d_i), size=2000, replace=False)
        tfi = np.mean([inner_ld(model, d_i[i][0]) == d_i[i][1]
                       for i in idx])
        vrng = np.random.default_rng(100 + rnd)
        oks = sum(int(D.run_episode(make_policy(model),
                                    *D.any_init(vrng))[0])
                  for _ in range(EVAL_N))
        res[f"r{rnd}"] = oks
        res[f"r{rnd}_tf_inner"] = round(float(tfi), 3)
        save_model(model, "checkpoint", {"round": rnd, "score": oks})
        print(f"round {rnd}: {oks}/{EVAL_N} tf_inner {tfi:.3f} "
              f"(outer {len(d_o)})", flush=True)
        with open(OUTPUT_DIR / "metrics.json", "w") as f:
            json.dump(res, f, indent=2)
        if oks > best["score"]:
            best = {"score": oks, "round": rnd, "model": model}
        if rnd == ROUNDS - 1:
            break
        crng2 = np.random.default_rng(200 + rnd)
        for _ in range(ROLLOUTS):
            s, tgt = D.any_init(crng2)
            hist = CHist()
            pol_hist = CHist()
            for t in range(EP_CAP):
                hist.see(s, tgt)
                lp, lc, ld = tch(s, tgt)
                d_o.append((hist.outer_code(), lp, lc))
                d_i.append((hist.inner_code(PLEV[lp]), ld))
                pol_hist.see(s, tgt)
                alp, alc, ald = cascade_act(model, pol_hist)
                pol_hist.did(ald)
                hist.did(ald)
                s = D.physics(s, levels_from(alp, alc, ald))
                if abs(s[0]) > 30 or abs(s[1]) > 30 or D.at_goal(s, tgt):
                    break

    save_model(best["model"], "weights_best",
               {"round": best["round"], "score": best["score"]})
    res["best"] = {"score": best["score"], "round": best["round"]}
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
