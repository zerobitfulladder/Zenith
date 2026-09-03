"""Label protects, belief recruits: sum the two biases.

`purity_split.py` found that label routing keeps every intruder off the
committed templates (drift 0.000, online table within 0.4 pts of a recount) but
recruits nobody -- ~254 templates stay dead, as under pure geometry -- while
belief routing wakes all 400 but lets intruders through (drift 0.168 with the
purity step). The two biases are the same table entry weighted differently, so:

    win = argmin( err - beta_L * T[t,c,y] - beta_B * q . T[t,c,:] )

Purity step throughout; the champion (cnt/n + belief 1.5) as reference.
`conv` = share of the templates committed after 0-4 whose row now names a
different class (converted to 5-9); `drift kept` = drift of the ones that were
NOT converted. Under belief routing nearly every template is committed after
0-4, so 5-9 must convert some; the question is how many, and whether the rest
stood still.

Usage:  uv run python combined.py [--smoke]
"""

import json, sys, time
from pathlib import Path
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import purity_split as P
import rf_sweep as R

OUT = P.OUT
#        tag                 rule      route     beta_L  beta_B
ARMS = [("champion",        "cntn",   "belief",  0.0,   1.5),
        ("belief1.5",       "purity", "belief",  0.0,   1.5),
        ("label0.25",       "purity", "label",   0.25,  0.0),
        ("L0.25+B0.5",      "purity", "both",    0.25,  0.5),
        ("L0.25+B1.0",      "purity", "both",    0.25,  1.0),
        ("L0.25+B1.5",      "purity", "both",    0.25,  1.5),
        ("L0.1+B1.5",       "purity", "both",    0.1,   1.5),
        ("L0.5+B1.5",       "purity", "both",    0.5,   1.5)]
if P.SMOKE:
    ARMS = ARMS[:1] + ARMS[5:6]


def main():
    OUT.mkdir(exist_ok=True)
    rig = R.Rig(P.PS)
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    if P.SMOKE:
        Xtr, ytr, Xte, yte = Xtr[:3000], ytr[:3000], Xte[:600], yte[:600]
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    phases = [(Xtr[A], ytr_g[A], "0-4"), (Xtr[~A], ytr_g[~A], "5-9"), (Xtr, ytr_g, "0-9")]
    print(f"summed biases, seeds {P.SEEDS}, {P.EPOCHS} epochs/phase\n")
    print(f"{'':<14}{'after 0-4':>10}{'after 5-9: old/new':>20}{'final old/new/all':>22}"
          f"{'online':>8}{'  #com':>6}{'pur B':>7}{'drift':>7}{'conv':>6}{'d-kept':>7}{'dead':>6}")
    res = {}
    for tag, rule, route, bl, bb in ARMS:
        P.BETA_L, P.BETA_B = bl, bb
        t0 = time.time()
        runs = []
        for seed in P.SEEDS:
            c, b, sn = P.run(rig, rule, route, phases, Xtr, ytr_g, Xte, yte_g, seed)
            runs.append({"seed": seed, "curve": c, "bounds": b})
        res[tag] = {"rule": rule, "route": route, "beta_L": bl, "beta_B": bb, "runs": runs}
        e = [P.end_points(r["curve"]) for r in runs]
        mean = lambda key: np.mean([x[key] for x in e])
        endB = [[p for p in r["curve"] if p["phase"] == 1][-1] for r in runs]
        conv = np.mean([p.get("converted", np.nan) for p in endB])
        dkept = np.mean([p.get("drift_kept", np.nan) for p in endB])
        dead = np.mean([r["curve"][-1]["n_dead"] for r in runs])
        print(f"  {tag:<12}{mean('A_old'):>10.4f}"
              f"{mean('B_old'):>10.4f}/{mean('B_new'):.4f}"
              f"{mean('C_old'):>8.4f}/{mean('C_new'):.4f}/{mean('C_all'):.4f}"
              f"{mean('C_online'):>8.4f}{mean('n_com'):>6.0f}{mean('pur_B'):>7.3f}"
              f"{mean('drift_B'):>7.3f}{conv:>6.2f}{dkept:>7.3f}{dead:>6.0f}   ({time.time() - t0:.0f}s)",
              flush=True)
    (OUT / "combined.json").write_text(json.dumps(res, indent=1))
    draw(res)


def draw(res):
    cmap = plt.get_cmap("viridis")
    both = [t for t in res if res[t]["route"] == "both"]
    style = {}
    for t in res:
        r = res[t]
        if r["route"] == "both":
            style[t] = dict(color=cmap(both.index(t) / max(1, len(both) - 1)), ls="-", lw=1.8)
        elif t == "champion":
            style[t] = dict(color="k", ls=":", lw=2.2)
        elif r["route"] == "belief":
            style[t] = dict(color="#1b6ca8", ls=":", lw=1.8)
        else:
            style[t] = dict(color="#c1121f", ls=":", lw=1.8)
    panels = [("old classes 0-4, recount", lambda p: p["recount"][0]),
              ("new classes 5-9, recount", lambda p: p["recount"][1]),
              ("all ten, online table", lambda p: p["online"][2]),
              ("purity of templates committed after 0-4", lambda p: p.get("purity_committed", np.nan)),
              ("drift of committed templates (1 - |cos|)", lambda p: p.get("drift_committed", np.nan)),
              ("share of committed templates converted to a new class",
               lambda p: p.get("converted", np.nan))]
    bounds = res[next(iter(res))]["runs"][0]["bounds"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (ttl, get) in zip(axes.flat, panels):
        for t, r in res.items():
            x, yv = P.mean_curve(r["runs"], get)
            ax.plot(x, yv, style[t]["ls"], color=style[t]["color"], lw=style[t]["lw"], label=t)
        for bb, nm in zip(bounds, ("0-4", "5-9", "0-9")):
            ax.axvline(bb, color="k", lw=0.8, ls=":")
            ax.text(bb + 3, ax.get_ylim()[0], nm, fontsize=8, va="bottom")
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("training batches")
    axes[0, 0].legend(fontsize=7, loc="lower left", ncol=2)
    fig.suptitle(f"28x28 split, {len(P.SEEDS)} seed(s), purity step.  Dotted = single bias, "
                 f"solid = label + belief summed", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "combined.png", dpi=140); plt.close(fig)
    print(f"\nwrote {OUT / 'combined.png'}")


if __name__ == "__main__":
    main()
