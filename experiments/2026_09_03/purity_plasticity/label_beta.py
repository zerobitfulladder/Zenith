"""Label routing at beta 1.5 protected perfectly and collapsed capacity: 7
templates in use, ~250 dead. The label bonus (a log-ratio up to ~2, times 1.5)
dwarfs any geometric difference (err gaps ~0.1-0.3), so the first template to
commit to a class takes the whole class. The belief route gets away with 1.5
because its belief is soft (top class ~0.5 of the mass), which halves the
effective bonus. So: the same two step rules with label routing at smaller beta.

Usage:  uv run python label_beta.py
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

BETAS = [0.1, 0.25, 0.5, 1.0]
OUT = P.OUT


def main():
    OUT.mkdir(exist_ok=True)
    rig = R.Rig(P.PS)
    Xtr, ytr, Xte, yte = R.E.load("mnist")
    Xtr = cp.asarray(Xtr.reshape(len(Xtr), -1), cp.float32)
    Xte = cp.asarray(Xte.reshape(len(Xte), -1), cp.float32)
    ytr_g, yte_g = cp.asarray(ytr), cp.asarray(yte)
    A = ytr_g < 5
    phases = [(Xtr[A], ytr_g[A], "0-4"), (Xtr[~A], ytr_g[~A], "5-9"), (Xtr, ytr_g, "0-9")]
    print(f"label routing, betas {BETAS}, seeds {P.SEEDS}\n")
    print(f"{'':<18}{'after 0-4':>10}{'after 5-9: old/new':>20}{'final old/new/all':>22}"
          f"{'online':>8}{'  #com':>6}{'pur B':>7}{'drift':>7}{'wsB':>6}{'dead':>6}")
    res = {}
    for beta in BETAS:
        P.BETA = P.BETA_L = beta
        for rule in P.RULES:
            tag = f"{rule}_label_b{beta:g}"
            t0 = time.time()
            runs = []
            for seed in P.SEEDS:
                c, b, _ = P.run(rig, rule, "label", phases, Xtr, ytr_g, Xte, yte_g, seed)
                runs.append({"seed": seed, "curve": c, "bounds": b})
            res[tag] = {"rule": rule, "route": "label", "beta": beta, "runs": runs}
            e = [P.end_points(r["curve"]) for r in runs]
            dead = np.mean([r["curve"][-1]["n_dead"] for r in runs])
            mean = lambda key: np.mean([x[key] for x in e])
            print(f"  {tag:<16}{mean('A_old'):>10.4f}"
                  f"{mean('B_old'):>10.4f}/{mean('B_new'):.4f}"
                  f"{mean('C_old'):>8.4f}/{mean('C_new'):.4f}/{mean('C_all'):.4f}"
                  f"{mean('C_online'):>8.4f}{mean('n_com'):>6.0f}{mean('pur_B'):>7.3f}"
                  f"{mean('drift_B'):>7.3f}{mean('ws_B'):>6.2f}{dead:>6.0f}"
                  f"   ({time.time() - t0:.0f}s)", flush=True)
    (OUT / "label_beta.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for rule, ls, mk in (("cntn", ":", "s"), ("purity", "-", "o")):
        pts = [(res[f"{rule}_label_b{b:g}"], b) for b in BETAS]
        E = [[P.end_points(r["curve"]) for r in rr["runs"]] for rr, _ in pts]
        m = lambda key: [np.mean([x[key] for x in e]) for e in E]
        dead = [np.mean([r["curve"][-1]["n_dead"] for r in rr["runs"]]) for rr, _ in pts]
        axes[0].plot(BETAS, m("C_all"), ls, marker=mk, color="#c1121f", label=f"{rule}, all ten")
        axes[0].plot(BETAS, m("C_old"), ls, marker=mk, color="#333333", label=f"{rule}, old 0-4")
        axes[1].plot(BETAS, m("B_old"), ls, marker=mk, color="#333333", label=f"{rule}, old after 5-9")
        axes[1].plot(BETAS, m("B_new"), ls, marker=mk, color="#c1121f", label=f"{rule}, new after 5-9")
        axes[2].plot(BETAS, m("drift_B"), ls, marker=mk, color="#1b6ca8", label=rule)
        axes[3].plot(BETAS, dead, ls, marker=mk, color="#1b6ca8", label=rule)
    for ax, t in zip(axes, ("final recount", "after the 5-9 phase",
                            "drift of committed templates over 5-9", "dead templates at the end")):
        ax.set_title(t, fontsize=10); ax.set_xlabel("beta (label routing)"); ax.set_xscale("log")
        ax.set_xticks(BETAS); ax.set_xticklabels([str(b) for b in BETAS])
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7)
    fig.suptitle("Label-routed competition: how strong before it collapses capacity?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "label_beta.png", dpi=140); plt.close(fig)
    print(f"\nwrote {OUT / 'label_beta.png'}")


if __name__ == "__main__":
    main()
