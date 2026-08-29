"""Pictures for the bound rig.

Reads a checkpoint (+ metrics.json beside it), writes PNGs beside it:

  training_curves.png  the per-report read metrics over the run
  read_scatter.png     frozen pupil vs oracle, per channel, on-path
  read_traces.png      one episode as time series — retrieval error is
                       piecewise-constant, this is where you see it
  stored_commands.png  every L3 template's command cells decoded — what
                       the memory can say at all, vs what the oracle says
  track_vocab.png      each track's L1 templates as 5-tap curves
  episodes.png         frozen episodes flown by the pupil, 2D paths

  .venv/bin/python experiments/2026_08_29/bound/bound_viz.py [ckpt.npz]
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
import fast_oracle as F                    # noqa: E402
import td_tracks as B                      # noqa: E402
import td3_stack as M                      # noqa: E402
import td_bound as T                       # noqa: E402


def probe(st, n=12000, seed=11):
    """Oracle flies; record (episode, oracle cd, pupil cd) per read tick."""
    rng = np.random.default_rng(seed)
    s, tgt = W.any_init(rng)
    st.reset()
    rows, ep = [], 0
    for _ in range(n):
        st.push(B.sense(s, tgt))
        _, cd, win = st.act()
        lv = F.teacher(s, tgt)
        tc, td = M.levels_to_cd(lv)
        if win >= 0:
            rows.append((ep, tc, td, cd[0], cd[1]))
        s = W.physics(s, lv)
        if W.at_goal(s, tgt) or abs(s[0]) > 30 or abs(s[1]) > 30:
            ep += 1
            s, tgt = W.any_init(rng)
            st.reset()
    return np.array(rows)


def fly(st, n_ep=8, seed=99):
    """The pupil flies frozen; same seed as the run's EVAL."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_ep):
        s, tgt = W.any_init(rng)
        st.reset()
        xs = [(s[0], s[1])]
        for _ in range(W.EP_CAP):
            st.push(B.sense(s, tgt))
            lv, _, _ = st.act()
            s = W.physics(s, lv)
            xs.append((s[0], s[1]))
            if abs(s[0]) > 30 or abs(s[1]) > 30:
                break
        out.append((np.array(xs), tgt))
    return out


def decode_block(prof, centers):
    p = np.clip(prof, 0, None)
    return B.sharpen(p, centers) if p.max() > 0 else np.nan


def main():
    ck = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        HERE / "results" / "weights_final.npz"
    if not ck.exists():
        ck = ck.parent / "checkpoint.npz"
    cfg = json.load(open(ck.with_suffix(".json")))
    st = T.load_stack(np.load(ck), cfg)
    out = ck.parent
    ds = st.dim_sensor
    print(f"{ck.name}: l3={st.l3.n} templates")

    # ---- training curves ------------------------------------------------
    mets = json.load(open(out / "metrics.json"))["reports"]
    ticks = [r["tick"] for r in mets if r.get("coll_corr") is not None]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, lab in (("coll_corr", "collective corr"),
                   ("diff_corr", "differential corr"),
                   ("agree", "exact-level agreement")):
        ax[0].plot(ticks, [r[k] for r in mets if r.get(k) is not None],
                   marker="o", ms=3, label=lab)
    ax[0].set_xlabel("tick")
    ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=8)
    ax[0].set_title("on-path read vs the oracle")
    ax[1].plot(ticks, [r["winners"] for r in mets
                       if r.get("winners") is not None], marker="o", ms=3)
    ax[1].set_xlabel("tick")
    ax[1].set_title("distinct winners per report window")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130)
    plt.close(fig)

    # ---- frozen probe ---------------------------------------------------
    rows = probe(st)
    ep, oc, od, gc, gd = rows.T
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, o, g, nm, cen in ((ax[0], oc, gc, "collective (N)", W.HOVER),
                             (ax[1], od, gd, "differential (N)", 0.0)):
        a.plot(o, g, ".", ms=2, alpha=0.25)
        lo, hi = min(o.min(), g.min()), max(o.max(), g.max())
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        r = np.corrcoef(g, o)[0, 1]
        a.set_title(f"{nm}   corr {r:.3f}")
        a.set_xlabel("oracle")
        a.set_ylabel("pupil (frozen read)")
    fig.tight_layout()
    fig.savefig(out / "read_scatter.png", dpi=130)
    plt.close(fig)

    # ---- one episode as traces -----------------------------------------
    eps, counts = np.unique(ep, return_counts=True)
    pick = eps[np.argmax(counts)]
    m = ep == pick
    t = np.arange(m.sum()) * W.DT if hasattr(W, "DT") else np.arange(m.sum())
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for a, o, g, nm in ((ax[0], oc[m], gc[m], "collective (N)"),
                        (ax[1], od[m], gd[m], "differential (N)")):
        a.plot(t, o, lw=1.4, label="oracle")
        a.plot(t, g, lw=1.0, label="pupil, frozen read")
        a.set_ylabel(nm)
        a.legend(fontsize=8)
    ax[1].set_xlabel("time in episode")
    ax[0].set_title("the read along one oracle episode")
    fig.tight_layout()
    fig.savefig(out / "read_traces.png", dpi=130)
    plt.close(fig)

    # ---- what the memory can say ---------------------------------------
    cc = np.array([decode_block(st.l3.W[j, ds:ds + T.NCMD], M.COLL_C)
                   for j in range(st.l3.n)])
    dd = np.array([decode_block(st.l3.W[j, ds + T.NCMD:], M.DIFF_C)
                   for j in range(st.l3.n)])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    ax[0].plot(cc, dd, ".", ms=2, alpha=0.2)
    ax[0].set_xlabel("stored collective (N)")
    ax[0].set_ylabel("stored differential (N)")
    ax[0].set_title(f"all {st.l3.n} templates' command cells")
    for a, sv, ov, nm in ((ax[1], cc, oc, "collective"),
                          (ax[2], dd, od, "differential")):
        a.hist(ov, bins=60, density=True, alpha=0.6, label="oracle commands")
        a.hist(sv[~np.isnan(sv)], bins=60, density=True, alpha=0.6,
               label="stored in templates")
        a.set_title(nm)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "stored_commands.png", dpi=130)
    plt.close(fig)

    # ---- track vocabularies --------------------------------------------
    fig, ax = plt.subplots(1, T.NS, figsize=(12, 3.6), sharey=True)
    for c in range(T.NS):
        l1 = st.tr[c].l1
        for i in range(l1.n):
            taps = [decode_block(l1.W[i, k * B.NB1:(k + 1) * B.NB1], B.CEN1)
                    for k in range(B.T1)]
            taps = np.array(taps)
            if np.isnan(taps).any():
                continue
            slope = taps[-1] - taps[0]
            ax[c].plot(taps, lw=0.8, alpha=0.6,
                       color=plt.cm.coolwarm(0.5 + np.clip(slope, -1, 1) / 2))
        ax[c].set_title(f"{T.SENSORS[c]} — {l1.n} L1 templates")
        ax[c].set_xlabel("tap (240 ms window)")
    ax[0].set_ylabel("decoded value (normalised)")
    fig.suptitle("each track's vocabulary of motion shapes "
                 "(colour = slope)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "track_vocab.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- frozen episodes ------------------------------------------------
    paths = fly(st)
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for a, (xs, tgt) in zip(ax.ravel(), paths):
        a.plot(xs[:, 0], xs[:, 1], lw=0.9)
        a.plot(*xs[0], "go", ms=5)
        a.plot(*tgt, "r*", ms=10)
        near = float(np.hypot(xs[:, 0] - tgt[0], xs[:, 1] - tgt[1]).min())
        a.set_title(f"closest {near:.2f} m", fontsize=9)
        a.set_aspect("equal")
        a.set_xlim(-12, 12)
        a.set_ylim(-12, 12)
    fig.suptitle("the pupil flying frozen (green start, red star target)")
    fig.tight_layout()
    fig.savefig(out / "episodes.png", dpi=130)
    plt.close(fig)
    print(f"wrote 6 figures to {out}/")


if __name__ == "__main__":
    main()
