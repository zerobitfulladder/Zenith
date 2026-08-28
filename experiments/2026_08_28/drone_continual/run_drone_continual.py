"""Exp 20 — the continual drone: one life, SLICED read, nothing rebuilt.

The user's full specification, no compromises:
  - Templates store bound wholes: [sensory channels ; command codes].
    NO tally anywhere.
  - The read is SLICED: the outer hypercolumn holds two competitions,
    one per physics slice — lateral channels (dx, vx) crown a minicolumn
    whose stored LEAN piece is used; vertical channels (dy, vy) crown a
    possibly different minicolumn whose COLLECTIVE piece is used. The
    inner hypercolumn reads whole (its physics does not factorise).
  - Hypercolumns are FIXED SIZE from tick zero: adopt-until-full over
    the early stream, then pure competition forever. No growth.
  - CONTINUAL: no rounds, no rebuilds, weights never wiped. The pupil
    flies; at every tick the oracle is asked what it would have done
    and every hypercolumn learns from that answer immediately.
    Episodes reset (random state, random target) on success, escape or
    timeout — the weights simply live on.

Report every DN_REPORT ticks: episodes, rolling success over the last
100 episodes, per-tick agreement with the oracle, and the best-match
similarity per hypercolumn (the starvation diagnostic: if it sags as
the pupil's own states drift away from the warmup distribution, the
fixed size was too small; if it holds, fixed size is vindicated).
Viewer-loadable checkpoint at every report.

Run:  .venv/bin/python experiments/2026_08_28/drone_continual/run_drone_continual.py
Env:  DN_TICKS (600000), DN_WARM (40000), DN_REPORT (20000), DN_K, DN_TAG
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recon_ladder"))
sys.path.insert(0, str(HERE.parent / "drone"))
import run_recon3 as R                    # noqa: E402
import run_drone as D                     # noqa: E402
import run_drone_cascade as C             # noqa: E402

OUTPUT_DIR = HERE / "results" / os.environ.get('DN_TAG', '').lstrip('_')
TICKS = int(os.environ.get("DN_TICKS", "600000"))
WARM = int(os.environ.get("DN_WARM", "40000"))
REPORT = int(os.environ.get("DN_REPORT", "20000"))
K = int(os.environ.get("DN_K", "4096"))
EP_CAP = 800


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exp_p = HERE.parent / "drone" / "results" / "cascade" / "expert.json"
    gains = json.load(open(exp_p))["gains"] if exp_p.exists() \
        else [0.15, 0.3, 2.0, 0.6]
    tch = C.make_cascade_teacher(*gains)

    rng = np.random.default_rng(0)
    # command codebooks (small hypercolumns over the thermometer levels)
    cb = {"p": D.Bank(C.K_CMD, C.N_P - 1, 3),
          "c": D.Bank(C.K_CMD, D.NLEV - 1, 4),
          "d": D.Bank(C.K_CMD, C.N_D - 1, 5)}
    for _ in range(30):
        for l in range(C.N_P):
            cb["p"].learn(D.cn(C.TH_P[l]))
        for l in range(D.NLEV):
            cb["c"].learn(D.cn(C.TH_C[l]))
        for l in range(C.N_D):
            cb["d"].learn(D.cn(C.TH_D[l]))

    t_out = R.Layer(K, C.ODIM + 2 * C.K_CMD, 0.05, np.random.default_rng(6))
    t_in = R.Layer(K, C.IDIM + C.K_CMD, 0.05, np.random.default_rng(7))

    model = {"assoc": "sliced", "bo": None, "bi": None, "cb": cb,
             "t_out": t_out, "t_in": t_in, "ty_out": None, "ty_in": None,
             "okey": lambda x: x, "ikey": lambda x: x,
             "kod": C.ODIM, "kid": C.IDIM}

    def mo_code(lp, lc):
        return np.concatenate([D.cn(cb["p"].code(D.cn(C.TH_P[lp]))),
                               D.cn(cb["c"].code(D.cn(C.TH_C[lc])))])

    def mi_code(ld):
        return D.cn(cb["d"].code(D.cn(C.TH_D[ld])))

    def save_ckpt(tag, extra):
        z1 = np.zeros((1, 1))
        np.savez(OUTPUT_DIR / f"{tag}.npz",
                 Wbo=z1, Wbi=z1, Wcp=cb["p"].W, Wcc=cb["c"].W,
                 Wcd=cb["d"].W, Wto=t_out.W, Wti=t_in.W, tyo=z1, tyi=z1)
        with open(OUTPUT_DIR / f"{tag}.json", "w") as f:
            json.dump({"cascade": True, "assoc": "sliced", **extra}, f)

    s, tgt = D.any_init(rng)
    hist = C.CHist()
    ep_tick = hold = episodes = 0
    recent, agree_o, agree_i, sim_o, sim_i = [], [], [], [], []
    stats = []
    best = {"score": -1.0, "tick": 0}      # peak, kept separately

    for t in range(TICKS):
        hist.see(s, tgt)
        x_o = hist.outer_code()
        lp_t, lc_t, ld_t = tch(s, tgt)          # the oracle's answer, now

        # ---- outer hypercolumn learns the bound whole ----------------
        z_o = D.cn(np.concatenate([x_o, C.GM_B * mo_code(lp_t, lc_t)]))
        t_out.learn(z_o, t_out.forward(z_o))

        # ---- outer read (sliced), or oracle during warmup ------------
        if t < WARM:
            lp_a, lc_a = lp_t, lc_t
        else:
            lp_a, lc_a = None, None
            for idx, piece in ((C.LAT_IDX, "p"), (C.VERT_IDX, "c")):
                qs = D.cn(x_o[idx])
                Ws = t_out.W[:, idx]
                sims = (Ws @ qs) / (np.linalg.norm(Ws, axis=1) + 1e-9)
                w = int(np.argmax(sims))
                row = t_out.W[w]
                if piece == "p":
                    sim_o.append(float(sims[w]))
                    lp_a = C._mode(cb["p"], row[C.ODIM:C.ODIM + C.K_CMD],
                                   C.TH_P)
                else:
                    lc_a = C._mode(cb["c"], row[C.ODIM + C.K_CMD:], C.TH_C)
        agree_o.append(int(lp_a == lp_t))

        # ---- inner hypercolumn: goal channel carries the ACTED lean --
        x_i = hist.inner_code(C.PLEV[lp_a])
        ld_lbl = C.q(C.DLEV, np.clip(
            gains[2] * (C.PLEV[lp_a] - D.wrap(s[4])) - gains[3] * s[5],
            -C.DMAX, C.DMAX))
        z_i = D.cn(np.concatenate([x_i, C.GM_B * mi_code(ld_lbl)]))
        t_in.learn(z_i, t_in.forward(z_i))

        if t < WARM:
            ld_a = ld_lbl
        else:
            q_i = D.cn(np.concatenate([x_i, np.zeros(C.K_CMD)]))
            f = t_in.forward(q_i)
            w = int(np.argmax(f))
            sim_i.append(float(f[w]))
            ld_a = C._mode(cb["d"], t_in.W[w][C.IDIM:], C.TH_D)
        agree_i.append(int(ld_a == ld_lbl))

        hist.did(ld_a)
        s = D.physics(s, C.levels_from(lp_a, lc_a, ld_a))
        ep_tick += 1

        done = False
        if D.at_goal(s, tgt):
            hold += 1
            if hold >= 10:
                recent.append(1)
                done = True
        else:
            hold = 0
        if abs(s[0]) > 30 or abs(s[1]) > 30 or ep_tick >= EP_CAP:
            recent.append(0)
            done = True
        if done:
            episodes += 1
            s, tgt = D.any_init(rng)
            hist = C.CHist()
            ep_tick = hold = 0
            recent = recent[-100:]

        if (t + 1) % REPORT == 0:
            row = {"tick": t + 1,
                   "phase": "WARM(oracle flies)" if t < WARM else "pupil",
                   "episodes": episodes,
                   "success_last100": round(float(np.mean(recent))
                                            if recent else 0.0, 3),
                   "agree_outer": round(float(np.mean(agree_o)), 3),
                   "agree_inner": round(float(np.mean(agree_i)), 3),
                   "bestmatch_outer": round(float(np.mean(sim_o)), 3)
                   if sim_o else None,
                   "bestmatch_inner": round(float(np.mean(sim_i)), 3)
                   if sim_i else None}
            stats.append(row)
            print(json.dumps(row), flush=True)
            agree_o, agree_i, sim_o, sim_i = [], [], [], []
            # "checkpoint" = the living creature right now (overwritten);
            # "weights_best" = the PEAK, kept whenever it improves;
            # "weights_final" = whatever it ends as.
            save_ckpt("checkpoint", {"tick": t + 1,
                                     "score": row["success_last100"]})
            if t >= WARM and row["success_last100"] > best["score"]:
                best = {"score": row["success_last100"], "tick": t + 1}
                save_ckpt("weights_best", best)
                print(f"  new best {best['score']} @ tick {best['tick']}",
                      flush=True)
            with open(OUTPUT_DIR / "metrics.json", "w") as f:
                json.dump({"best": best, "reports": stats}, f, indent=2)

    save_ckpt("weights_final", {"tick": TICKS,
                                "score": stats[-1]["success_last100"]})
    print(f"continual life complete — best {best['score']} "
          f"@ tick {best['tick']}", flush=True)


if __name__ == "__main__":
    main()
