"""Single-layer sparse-OR drone — the continual run.

One hypercolumn, one giant OR'ed sparse vector per moment holding both
the sensed state and the oracle's thrust commands. The pupil flies; at
every tick the oracle is asked what it would have done and the
hypercolumn learns that moment immediately. No rounds, no rebuilds, no
tally, no second layer. Episodes reset; the weights never do.

Reports every SL_REPORT ticks and writes three checkpoints:
  checkpoint    - the creature right now (overwritten)
  weights_best  - the peak rolling success, never overwritten by worse
  weights_final - whatever it ends as

Run:  .venv/bin/python experiments/2026_08_28/single_layer_drone/run_sl_drone.py
Env:  SL_TICKS SL_WARM SL_REPORT SL_K SL_SIZE SL_ETA SL_TAG
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sl_drone as W                       # noqa: E402
import sl_viz as V                         # noqa: E402

OUTPUT_DIR = HERE / "results" / os.environ.get('SL_TAG', '').lstrip('_')
TICKS = int(os.environ.get("SL_TICKS", "400000"))
WARM = int(os.environ.get("SL_WARM", "40000"))
REPORT = int(os.environ.get("SL_REPORT", "10000"))
K = int(os.environ.get("SL_K", "4096"))
SIZE = int(os.environ.get("SL_SIZE", "8192"))
ETA = float(os.environ.get("SL_ETA", "0.05"))
CONSOL = float(os.environ.get("SL_CONSOL", "0"))
GPU = os.environ.get("SL_GPU", "0") == "1"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enc = W.Encoder(SIZE, seed=0)
    hc = W.Hypercolumn(K, SIZE, eta=ETA, seed=1, consol=CONSOL, gpu=GPU)
    rng = np.random.default_rng(0)

    def save(tag, extra):
        np.savez(OUTPUT_DIR / f"{tag}.npz", Wt=hc.weights_numpy(),
                 n_boot=hc.n_boot, wins=hc.wins)
        with open(OUTPUT_DIR / f"{tag}.json", "w") as f:
            json.dump({"size": SIZE, "k": K, "nb": W.NB,
                       "enc_seed": 0, **extra}, f)

    s, tgt = W.any_init(rng)
    lv_prev = (W.NLEV // 2, W.NLEV // 2)
    ep_tick = hold = episodes = 0
    recent, agree, sim, occ = [], [], [], []
    stats, best = [], {"score": -1.0, "tick": 0}

    for t in range(TICKS):
        sens = W.sense(s, tgt, lv_prev)
        lv_t = W.teacher(s, tgt)                    # the oracle, every tick

        # learn the bound moment: state OR'ed with the oracle's commands
        hc.learn(*enc.encode_sparse(sens, lv_t))

        # act: sensory bits only, motor bits left empty
        if t < WARM:
            lv_a = lv_t
        else:
            qi, qv = enc.encode_sparse(sens, None)
            w, c = hc.compete(qi, qv)
            if w < 0:
                lv_a = (W.NLEV // 2, W.NLEV // 2)
            else:
                lv_a = enc.read_motors(hc.row(w))
                sim.append(c)
            occ.append(len(qi) / SIZE)
        agree.append(0.5 * ((lv_a[0] == lv_t[0]) + (lv_a[1] == lv_t[1])))

        s = W.physics(s, lv_a)
        lv_prev = lv_a
        ep_tick += 1

        done = False
        if W.at_goal(s, tgt):
            hold += 1
            if hold >= 10:
                recent.append(1)
                done = True
        else:
            hold = 0
        if abs(s[0]) > 30 or abs(s[1]) > 30 or ep_tick >= W.EP_CAP:
            recent.append(0)
            done = True
        if done:
            episodes += 1
            s, tgt = W.any_init(rng)
            lv_prev = (W.NLEV // 2, W.NLEV // 2)
            ep_tick = hold = 0
            recent = recent[-100:]

        if (t + 1) % REPORT == 0:
            row = {"tick": t + 1,
                   "phase": "WARM(oracle flies)" if t < WARM else "pupil",
                   "episodes": episodes,
                   "success_last100": round(float(np.mean(recent))
                                            if recent else 0.0, 3),
                   "agree": round(float(np.mean(agree)), 3),
                   "bestmatch": round(float(np.mean(sim)), 3) if sim else None,
                   "minicolumns_used": int(hc.n_boot),
                   "code_occupancy": round(float(np.mean(occ)), 4)
                   if occ else None}
            stats.append(row)
            print(json.dumps(row), flush=True)
            agree, sim, occ = [], [], []
            # pictures: what the minicolumns have become, and one moment
            V.save_templates(hc, enc,
                             OUTPUT_DIR / f"templates_{t + 1:07d}.png",
                             t + 1)
            wi, _ = hc.compete(*enc.encode_sparse(sens, None))
            V.save_sample(enc, sens, lv_t,
                          OUTPUT_DIR / f"sample_{t + 1:07d}.png", t + 1,
                          winner=hc.row(wi) if wi >= 0 else None)
            save("checkpoint", {"tick": t + 1,
                                "score": row["success_last100"]})
            if t >= WARM and row["success_last100"] > best["score"]:
                best = {"score": row["success_last100"], "tick": t + 1}
                save("weights_best", best)
                print(f"  new best {best['score']} @ {best['tick']}",
                      flush=True)
            with open(OUTPUT_DIR / "metrics.json", "w") as f:
                json.dump({"best": best, "config": {
                    "k": K, "size": SIZE, "nb": W.NB, "eta": ETA,
                    "warm": WARM}, "reports": stats}, f, indent=2)

    save("weights_final", {"tick": TICKS,
                           "score": stats[-1]["success_last100"]})
    print(f"done — best {best['score']} @ tick {best['tick']}", flush=True)


if __name__ == "__main__":
    main()
