"""Wide-3 showcase: generation variety + per-attribute classification report.

Figures (wide3/results/):
  portraits_gated.png  - attribute-conditioned retrieval (contrast+IDF+gate)
  archive_lowwin.png   - mid- and low-rehearsal memories (diversity check)
  sampled_variety.png  - sample-at-top variants of retrieved bases
  morphs.png           - code-space interpolation between memory pairs
Report: classification_report.md - 40 attributes, probe on L3 code.

Run:  .venv/bin/python experiments/2026_08_23/wide3/run_wide3_showcase.py
"""

import json
import os
from pathlib import Path

os.environ["GF_LEVELS"] = "[[4,1,64],[5,2,256],[5,1,256]]"
os.environ["GF_KTOP2"] = "1000"
os.environ["GF_OUT"] = "wide3/results"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyramid"))  # noqa: E402

from run_pyramid import (  # noqa: E402
    CH, CODE_DIM, GRIDS, LEVELS, Dict, ETAS, encode_stack,
    harden_np, render_down, xp, XP_NAME,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "2026_08_23" / "wide3" / "results"
ATTR_NAMES = json.loads((ROOT / "data/celeba/attr_names.json").read_text())
A_IDX = {n: i for i, n in enumerate(ATTR_NAMES)}
LAM = 0.5
G = GRIDS[-1]
KT = LEVELS[-1][2]

PORTRAITS = [
    ["Male", "No_Beard"], ["Male", "Mustache"], ["Male", "Eyeglasses"],
    ["Male", "Wearing_Hat"], ["Wearing_Lipstick", "Smiling"],
    ["Heavy_Makeup", "Blond_Hair"], ["Wearing_Lipstick", "Wavy_Hair"],
    ["Bald", "Male"], ["Male", "Goatee"], ["Heavy_Makeup", "Young"],
]


def main():
    w = np.load(OUT / "weights.npz")

    class Bank:
        pass

    dics_np = []
    for i in range(len(LEVELS)):
        b = Bank()
        b.W = w[f"W{i+1}"]
        b.k = LEVELS[i][2]
        dics_np.append(b)
    Wtn, wins = w["Wtop"], w["top_wins"]
    L = Wtn[:, CODE_DIM:]
    Lc = L - L.mean(axis=0, keepdims=True)
    A = np.load(ROOT / "data/celeba/attrs_48.npy").astype(np.float64)
    idf = 1.0 / np.sqrt(A.mean(axis=0) + 0.01)
    gate = wins >= np.median(wins)

    def retrieve(names):
        q = np.zeros(len(ATTR_NAMES))
        for n in names:
            q[A_IDX[n]] = idf[A_IDX[n]]
        s = Lc @ q
        s[~gate] = -1e9
        return int(np.argmax(s))

    def mem_code(m):
        return np.maximum(Wtn[m, :CODE_DIM], 0.0).reshape(G, G, KT)

    # ---- 1. Gated portraits ----------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4))
    for ax, names in zip(axes.flat, PORTRAITS):
        m = retrieve(names)
        ax.imshow(render_down(mem_code(m), dics_np), cmap="gray")
        ax.set_title("+".join(n.replace("Wearing_", "") for n in names) + f" m{m}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Wide-3 portraits (gated retrieval, luminance)")
    fig.tight_layout()
    fig.savefig(OUT / "portraits_gated.png", dpi=110)
    plt.close(fig)
    print("portraits done", flush=True)

    # ---- 2. Mid- and low-rehearsal memories (diversity check) -------------
    order = np.argsort(-wins)
    med = order[len(order) // 2 - 6:len(order) // 2 + 6]
    low = order[-40:-28]
    fig, axes = plt.subplots(4, 6, figsize=(13, 9))
    for ax, m in zip(axes.flat, np.concatenate([med, low])):
        ax.imshow(render_down(mem_code(int(m)), dics_np), cmap="gray")
        ax.set_title(f"m{int(m)} ({int(wins[int(m)])})", fontsize=7)
        ax.axis("off")
    fig.suptitle("Median-rehearsal (top 2 rows) and low-rehearsal (bottom 2) memories")
    fig.tight_layout()
    fig.savefig(OUT / "archive_lowwin.png", dpi=110)
    plt.close(fig)
    print("lowwin done", flush=True)

    # ---- 3. Sampled variety ----------------------------------------------
    rng = np.random.default_rng(7)

    def sample_top(code3, temp=0.5):
        o = np.zeros_like(code3)
        for a in range(G):
            for b_ in range(G):
                seg = np.maximum(code3[a, b_], 0.0)
                mx = seg.max()
                if mx <= 0:
                    continue
                p = (seg / mx) ** (1.0 / temp)
                p /= p.sum()
                j = int(rng.choice(KT, p=p))
                o[a, b_, j] = seg[j] if seg[j] > 0 else mx
        return o

    bases = [["Wearing_Lipstick", "Smiling"], ["Male", "No_Beard"],
             ["Heavy_Makeup", "Blond_Hair"], ["Male", "Eyeglasses"]]
    fig, axes = plt.subplots(4, 5, figsize=(11, 9))
    for row, names in enumerate(bases):
        c3 = mem_code(retrieve(names))
        axes[row, 0].imshow(render_down(c3, dics_np), cmap="gray")
        axes[row, 0].set_ylabel("+".join(n[:6] for n in names), fontsize=7)
        for col in range(1, 5):
            axes[row, col].imshow(render_down(sample_top(c3), dics_np), cmap="gray")
        for col in range(5):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("Sample-at-top variety (col 0 = argmax base, T=0.5 samples)")
    fig.tight_layout()
    fig.savefig(OUT / "sampled_variety.png", dpi=110)
    plt.close(fig)
    print("variety done", flush=True)

    # ---- 4. Memory morphs -------------------------------------------------
    pairs = [(retrieve(["Wearing_Lipstick", "Smiling"]), retrieve(["Male", "No_Beard"])),
             (retrieve(["Heavy_Makeup", "Blond_Hair"]), retrieve(["Bald", "Male"]))]
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.6))
    for row, (ma, mb) in enumerate(pairs):
        ca, cb = mem_code(ma), mem_code(mb)
        for col, t in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
            axes[row, col].imshow(render_down((1 - t) * ca + t * cb, dics_np), cmap="gray")
            if row == 0:
                axes[row, col].set_title(f"t={t}", fontsize=8)
            axes[row, col].axis("off")
    fig.suptitle("Code-space morphs between memories")
    fig.tight_layout()
    fig.savefig(OUT / "morphs.png", dpi=110)
    plt.close(fig)
    print("morphs done", flush=True)

    # ---- 5. Classification report (probe on L3 code) ----------------------
    X = np.load(ROOT / "data/celeba/images_48.npy")
    dims = [16] + [LEVELS[li][0] ** 2 * CH[li - 1] for li in range(1, len(LEVELS))]
    dics = [Dict(LEVELS[i][2], dims[i], ETAS[i]) for i in range(len(LEVELS))]
    for i, d in enumerate(dics):
        d.W = xp.asarray(w[f"W{i+1}"], dtype=xp.float32)
        d.n_boot = d.k
    N = 6000
    codes = []
    for s in range(0, N, 64):
        maps = encode_stack(xp.asarray(X[s:min(s + 64, N)].astype(np.float32)), dics, False)
        c = maps[-1]
        codes.append((c.get() if XP_NAME == "cupy" else c).reshape(len(c), -1).astype(np.float32))
        if XP_NAME == "cupy":
            xp.get_default_memory_pool().free_all_blocks()
    C = np.concatenate(codes)
    print("codes encoded", flush=True)

    rows = []
    for ai, name in enumerate(ATTR_NAMES):
        y = A[:N, ai]
        base = max(y.mean(), 1 - y.mean())
        if y[:4000].std() < 1e-6:
            rows.append((name, y.mean(), base, float("nan"), float("nan")))
            continue
        clf = LogisticRegression(max_iter=300)
        clf.fit(C[:4000], y[:4000])
        pred = clf.predict(C[4000:])
        acc = float((pred == y[4000:]).mean())
        f1 = float(f1_score(y[4000:], pred, zero_division=0))
        rows.append((name, y.mean(), base, acc, f1))
        print(f"  {name}: acc={acc:.3f} f1={f1:.3f}", flush=True)

    rows.sort(key=lambda r: (r[3] - r[2]) if r[3] == r[3] else -1, reverse=True)
    lines = ["# Wide-3 attribute classification report (probe on L3 code)",
             "", f"n={N} (4000 train / 2000 test), 40 binary probes.",
             "", "| attribute | pos rate | majority baseline | probe acc | F1(pos) | gain |",
             "|---|---|---|---|---|---|"]
    for name, rate, base, acc, f1 in rows:
        g = acc - base if acc == acc else float("nan")
        lines.append(f"| {name} | {rate:.2f} | {base:.3f} | {acc:.3f} | {f1:.3f} | {g:+.3f} |")
    (OUT / "classification_report.md").write_text("\n".join(lines) + "\n")
    mean_gain = np.nanmean([r[3] - r[2] for r in rows if r[3] == r[3]])
    print(f"RESULT report done, mean gain over majority baseline: {mean_gain:+.3f}", flush=True)


if __name__ == "__main__":
    main()
