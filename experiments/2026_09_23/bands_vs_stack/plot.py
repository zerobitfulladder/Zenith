"""Summarise results/runs/*.json into results/summary.md and three figures."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = HERE / "results"
runs = {}
for f in sorted(glob.glob(str(R / "runs/*.json"))):
    d = json.loads(Path(f).read_text()); runs[d["tag"]] = d
names = next(iter(runs.values()))["names"]
short = [n.replace("_", " ") for n in names]

def get(arm, cond, seed=0):
    return runs.get(f"{arm}_{cond}_s{seed}")

ARMS = [("pixels", "pixels, linear"), ("bands1", "bands, 1 conv each"),
        ("bands3", "bands, 3 convs each"), ("stack", "stack (CNN, avg pool)"), ("stack_max", "stack (CNN, max pool)")]
lines = ["# Results\n", "Balanced accuracy on the 5,000 held-out images, mean over the 40 attributes, at the epoch",
         "with the best validation score. `acc` is plain accuracy.\n",
         "| arm | params | aligned | acc | jitter | acc |", "|---|---|---|---|---|---|"]
for arm, label in ARMS:
    a, j = get(arm, "aligned"), get(arm, "jitter")
    if not a and not j: continue
    f = lambda d, k: (f"{np.mean(d['best'][k]):.4f}" if d else "—")
    params = (a or j)["params"]
    lines.append(f"| {label} | {params:,} | {f(a,'test')} | {f(a,'test_acc')} | {f(j,'test')} | {f(j,'test_acc')} |")

# ---- figure 1: the headline bars
fig, ax = plt.subplots(figsize=(7, 3.6))
arms = [(a, l) for a, l in ARMS if get(a, "aligned") or get(a, "jitter")]
x = np.arange(len(arms)); w = 0.38
for k, (cond, col) in enumerate([("aligned", "#3b6ea5"), ("jitter", "#d0803a")]):
    vals = [get(a, cond)["best"]["test"] if get(a, cond) else np.nan for a, _ in arms]
    ax.bar(x + (k - 0.5) * w, vals, w, label=cond, color=col)
    for xi, v in zip(x, vals):
        if not np.isnan(v): ax.text(xi + (k - 0.5) * w, v + 0.003, f"{v:.3f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in arms], fontsize=8, rotation=10)
ax.set_ylabel("balanced accuracy, mean of 40"); ax.set_ylim(0.5, 0.9); ax.legend(frameon=False)
ax.set_title("Bands read the image at five scales; the stack reads the layer below", fontsize=10)
plt.tight_layout(); plt.savefig(R / "headline.png", dpi=140); plt.close()

# ---- figure 2: per-attribute gap, stack minus bands, whatever conditions exist
series = []
for cond, other, col, label in [("aligned", "bands1", "#3b6ea5", "aligned: stack - bands (1 conv)"),
                                ("aligned", "bands3", "#6fa3d9", "aligned: stack - bands (3 convs)"),
                                ("jitter", "bands1", "#d0803a", "jitter: stack - bands (1 conv)")]:
    s_, b_ = get("stack", cond), get(other, cond)
    if s_ and b_:
        series.append((np.array(s_["best"]["test_bal"]) - np.array(b_["best"]["test_bal"]), col, label))
if series:
    order = np.argsort(series[0][0])
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(40); w = 0.8 / len(series)
    for k, (gk, col, label) in enumerate(series):
        ax.barh(y + (k - (len(series) - 1) / 2) * w, gk[order], w, label=label, color=col)
    ax.set_yticks(y); ax.set_yticklabels([short[i] for i in order], fontsize=7)
    ax.axvline(0, color="k", lw=0.6); ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_xlabel("difference in balanced accuracy")
    ax.set_title("Where the stack wins, per attribute", fontsize=10)
    plt.tight_layout(); plt.savefig(R / "gap_per_attribute.png", dpi=140); plt.close()
    sa, ba = get("stack", "aligned"), get("bands1", "aligned")
    if sa and ba:
        b3 = get("bands3", "aligned")
        lines += ["", "## Per attribute, aligned", "",
                  "| attribute | pixels | bands 1 conv | bands 3 convs | stack | stack - bands1 |", "|---|---|---|---|---|---|"]
        px = get("pixels", "aligned")
        for i in order[::-1]:
            f3 = f"{b3['best']['test_bal'][i]:.3f}" if b3 else "—"
            fp = f"{px['best']['test_bal'][i]:.3f}" if px else "—"
            lines.append(f"| {short[i]} | {fp} | {ba['best']['test_bal'][i]:.3f} | {f3} | {sa['best']['test_bal'][i]:.3f} | "
                         f"{sa['best']['test_bal'][i]-ba['best']['test_bal'][i]:+.3f} |")

# ---- figure 3: each band alone, per attribute
solo = [runs.get(f"bands1_L{L}_aligned_s0") for L in range(1, 6)]
if all(solo):
    M = np.array([d["best"]["test_bal"] for d in solo])          # 5 x 40
    allb = get("bands1", "aligned")
    order = np.argsort(M.argmax(0) + 0.001 * M.max(0))
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(M[:, order], aspect="auto", cmap="viridis", vmin=0.5, vmax=0.95)
    ax.set_yticks(range(5)); ax.set_yticklabels(["full (3 px)", "1/2 (6 px)", "1/4 (12 px)", "1/8 (24 px)", "1/16 (48 px)"], fontsize=8)
    ax.set_xticks(range(40)); ax.set_xticklabels([short[i] for i in order], rotation=75, fontsize=6.5, ha="right")
    ax.set_ylabel("band alone (kernel span at full res)", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="balanced acc.")
    ax.set_title("Which band knows which attribute (each band alone, aligned)", fontsize=10)
    plt.tight_layout(); plt.savefig(R / "band_alone.png", dpi=140); plt.close()
    lines += ["", "## Each band alone (aligned), mean over attributes", "", "| band | kernel span at full res | mean balanced acc. |", "|---|---|---|"]
    for L, d, span in zip(range(1, 6), solo, ["3 px", "6 px", "12 px", "24 px", "48 px"]):
        lines.append(f"| 1/{2**(L-1)} | {span} | {d['best']['test']:.4f} |")
    if allb: lines.append(f"| all five | | {allb['best']['test']:.4f} |")
    lines += ["", "Best single band per attribute:", ""]
    for L in range(5):
        won = [short[i] for i in range(40) if M[:, i].argmax() == L]
        lines.append(f"- band 1/{2**L}: " + (", ".join(won) if won else "none"))


# ---- figure 4: test score per epoch, aligned
fig, ax = plt.subplots(figsize=(6.5, 3.6))
cols = dict(pixels="#888888", bands1="#3b6ea5", bands3="#6fa3d9", stack="#d0803a", stack_max="#e8b07a")
for arm, label in ARMS:
    d = get(arm, "aligned")
    if not d: continue
    ax.plot([r["epoch"] for r in d["log"]], [r["test"] for r in d["log"]], "o-", ms=3, label=label, color=cols[arm])
ax.set_xlabel("epoch"); ax.set_ylabel("test balanced accuracy"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Aligned faces, score per epoch", fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(R / "curves.png", dpi=140); plt.close()

(R / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:12]))
print("-> results/summary.md, headline.png, curves.png, gap_per_attribute.png, band_alone.png")
