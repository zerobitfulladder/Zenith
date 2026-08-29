"""Pictures for the balance rig. Writes PNGs beside the checkpoint.

  training_curves.png  per-report read metrics and EVAL over the run
  eval_traces.png      12 frozen episodes: |tilt| and speed vs time
  thrust_traces.png    LEFT and RIGHT thrust in newtons, pupil vs the
                       teacher from the same start
  read_scatter.png     frozen pupil vs teacher command, on-path

  .venv/bin/python experiments/2026_08_29/balance/balance_viz.py [ckpt.npz]
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "temporal_drone"))      # shared rig modules
sys.path.insert(0, str(HERE.parents[1] / "2026_08_28" / "single_layer_drone"))

import sl_drone as W                       # noqa: E402
import td_tracks as B                      # noqa: E402
import td3_stack as M                      # noqa: E402
import td_bound as T                       # noqa: E402
import td_balance as L                     # noqa: E402

HOVER_LV = (W.NLEV // 2, W.NLEV // 2)
SENSE = None                    # set in main() from the checkpoint cfg


def fly(st, s0, driver):
    """One episode from s0; driver(st, s, t) -> lv. Returns traces."""
    s = s0.copy()
    st.reset()
    out = {"tilt": [], "speed": [], "l": [], "r": []}
    for t in range(L.EP_LEN):
        lv = driver(st, s, t)
        out["l"].append(float(W.LEVELS[lv[0]]))
        out["r"].append(float(W.LEVELS[lv[1]]))
        s = W.physics(s, lv)
        out["tilt"].append(abs(float(W.wrap(s[4]))))
        out["speed"].append(float(np.hypot(s[2], s[3])))
    return {k: np.array(v) for k, v in out.items()}


def pupil_driver(st, s, t):
    st.push(SENSE(s))
    lv, _, _ = st.act()
    return lv


def teacher_driver(st, s, t):
    return L.teacher(s)


def main():
    ck = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE / "results" / "weights_best.npz"
    cfg = json.load(open(ck.with_suffix(".json")))
    st = T.load_stack(np.load(ck), cfg)
    global SENSE
    SENSE = L.make_sense(ns=int(cfg.get("ns", 3)),
                         vwarp=bool(cfg.get("vwarp", False)),
                         fovea=bool(cfg.get("fovea", False)))
    out = ck.parent
    tsec = np.arange(L.EP_LEN) * W.DT

    # ---- training curves ------------------------------------------------
    mets = json.load(open(out / "metrics.json"))["reports"]
    ticks = [r["tick"] for r in mets]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, lab in (("coll_corr", "collective corr"),
                   ("diff_corr", "differential corr"),
                   ("agree", "exact-level agreement")):
        pts = [(r["tick"], r[k]) for r in mets if r.get(k) is not None]
        ax[0].plot(*zip(*pts), marker="o", ms=3, label=lab)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("tick")
    ax[0].legend(fontsize=8)
    ax[0].set_title("on-path read vs the teacher")
    ax[1].plot(ticks, [r["EVAL"] for r in mets], marker="o", ms=3)
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_xlabel("tick")
    ax[1].set_title("EVAL: balanced 10 ticks, frozen")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130)
    plt.close(fig)

    # ---- frozen eval traces --------------------------------------------
    rng = np.random.default_rng(99)
    fig, ax = plt.subplots(3, 4, figsize=(13, 7), sharex=True)
    for a in ax.ravel():
        tr = fly(st, L.air_init(rng), pupil_driver)
        a.plot(tsec, tr["tilt"], lw=0.9, label="|tilt| (rad)")
        a.plot(tsec, tr["speed"], lw=0.9, label="speed (m/s)")
        a.axhline(0.17, color="tab:blue", ls=":", lw=0.7)
        a.axhline(0.3, color="tab:orange", ls=":", lw=0.7)
        a.axvspan(0, B.HIST * W.DT, color="gray", alpha=0.15)
        a.set_ylim(0, 3.2)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("the pupil flying frozen — grey = blind warm-up, "
                 "dotted = the success thresholds")
    fig.tight_layout()
    fig.savefig(out / "eval_traces.png", dpi=130)
    plt.close(fig)

    # ---- left/right thrust, pupil vs teacher, same start ---------------
    s0 = L.air_init(np.random.default_rng(7))
    tr_p = fly(st, s0, pupil_driver)
    tr_t = fly(st, s0, teacher_driver)
    fig, ax = plt.subplots(3, 2, figsize=(12, 7), sharex=True,
                           sharey="row")
    for col, (tr, nm) in enumerate(((tr_t, "teacher"), (tr_p, "pupil"))):
        ax[0, col].plot(tsec, tr["l"], lw=0.9, label="left thrust (N)")
        ax[0, col].plot(tsec, tr["r"], lw=0.9, label="right thrust (N)")
        ax[0, col].axhline(W.HOVER, color="k", ls=":", lw=0.7)
        ax[0, col].set_title(f"{nm} — same start, tilt "
                             f"{s0[4]:+.2f} rad, "
                             f"v ({s0[2]:+.1f}, {s0[3]:+.1f}) m/s")
        ax[1, col].plot(tsec, tr["r"] - tr["l"], lw=0.9, color="tab:red")
        ax[1, col].axhline(0, color="k", ls=":", lw=0.7)
        ax[2, col].plot(tsec, tr["tilt"], lw=0.9, label="|tilt| (rad)")
        ax[2, col].plot(tsec, tr["speed"], lw=0.9, label="speed (m/s)")
        for a in ax[:, col]:
            a.axvspan(0, B.HIST * W.DT, color="gray", alpha=0.15)
    ax[0, 0].set_ylabel("thrust (N)")
    ax[1, 0].set_ylabel("right − left (N)")
    ax[2, 0].set_ylabel("state")
    ax[0, 0].legend(fontsize=7)
    ax[2, 0].legend(fontsize=7)
    ax[2, 0].set_xlabel("seconds")
    ax[2, 1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out / "thrust_traces.png", dpi=130)
    plt.close(fig)

    # ---- on-path read scatter ------------------------------------------
    rng = np.random.default_rng(11)
    gc, gd, oc, od = [], [], [], []
    for _ in range(30):
        s = L.air_init(rng)
        st.reset()
        for t in range(L.EP_LEN):
            st.push(SENSE(s))
            _, cd, win = st.act()
            lv = HOVER_LV if t < B.HIST else L.teacher(s)
            if win >= 0 and t >= B.HIST:
                tc, td = M.levels_to_cd(lv)
                gc.append(cd[0]); gd.append(cd[1])
                oc.append(tc); od.append(td)
            s = W.physics(s, lv)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, o, g, nm in ((ax[0], np.array(oc), np.array(gc), "collective"),
                        (ax[1], np.array(od), np.array(gd),
                         "differential")):
        a.plot(o, g, ".", ms=2, alpha=0.25)
        lo, hi = min(o.min(), g.min()), max(o.max(), g.max())
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        a.set_title(f"{nm} (N)   corr {np.corrcoef(g, o)[0, 1]:.3f}")
        a.set_xlabel("teacher")
        a.set_ylabel("pupil (frozen read)")
    fig.tight_layout()
    fig.savefig(out / "read_scatter.png", dpi=130)
    plt.close(fig)
    print(f"wrote 4 figures to {out}/")


if __name__ == "__main__":
    main()
