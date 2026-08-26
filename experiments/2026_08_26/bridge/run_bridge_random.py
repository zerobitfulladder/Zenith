"""Bridge generalization: train on RANDOM pairs, test on digits.

User's test: train the whole encoder/unification/decoder system on
random images (never a digit), then present unseen MNIST digits to the
encoder and see what the decoder draws. Probes whether the bridge is a
lookup (top-1 recites the nearest random memory) or, read gradedly, an
interpolator over the span of its experience (population blend =
kernel-style function approximation).

Two training distributions x two read modes:
  scribbles — 2-5 random anti-aliased line segments (content-random,
              stroke-like: span plausibly covers digits)
  noise     — uniform pixel noise (no shared structure at all)
  reads: top-1 winner vs graded population (relu(match)^4 weights)

Predictions (before running):
1. Noise-trained fails in every mode — its L1 vocabulary never learned
   strokes; digits are outside both span and vocabulary.
2. Scribble-trained + top-1: draws a scribble, not the digit
   (recitation). Scribble-trained + graded: partial digit
   reconstruction — generalization = span + smooth read.

Run:  .venv/bin/python experiments/2026_08_26/bridge/run_bridge_random.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "bridge"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_4layer_topk import load_data  # noqa: E402
from run_temporal_digits import SeqDict, cn  # noqa: E402
from run_enc_dec import (  # noqa: E402
    CODE,
    KU,
    ETA,
    EPOCHS_U,
    l1_map,
    l2_code,
    render_decoder,
    train_stack,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "bridge" / "results" / "random"
H = 28
UNI_N = 15000
STACK_N = 1500


def scribble(rng):
    img = np.zeros((H, H), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:H].astype(float)
    for _ in range(rng.integers(2, 6)):
        x0, y0 = rng.uniform(4, 24, 2)
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(6, 16)
        dx, dy = np.cos(ang), np.sin(ang)
        px, py = xx - x0, yy - y0
        t = np.clip(px * dx + py * dy, 0, L)
        d = np.sqrt((px - t * dx) ** 2 + (py - t * dy) ** 2)
        img = np.maximum(img, np.clip(1 - d / 1.2, 0, 1).astype(np.float32))
    return img


def noise(rng):
    return rng.random((H, H)).astype(np.float32)


def build(gen, seed):
    rng = np.random.default_rng(seed)
    Xr = np.stack([gen(rng) for _ in range(max(STACK_N, 2000))])
    E1, E2 = train_stack(Xr, seed=seed + 10)
    D1, D2 = train_stack(Xr, seed=seed + 20)
    uni = SeqDict(KU, 2 * CODE, ETA, seed=seed + 30)
    for _ in range(EPOCHS_U):
        rng2 = np.random.default_rng(seed + 40)
        for _ in range(UNI_N):
            img = gen(rng2)
            enc = l2_code(l1_map(img, E1), E2)
            dec = l2_code(l1_map(img, D1), D2)
            uni.step(cn(np.concatenate([enc, dec])).astype(np.float32))
    return E1, E2, D1, D2, uni


def reconstruct(img, E1, E2, D1, D2, uni, mode):
    enc = l2_code(l1_map(img, E1), E2)
    q = cn(np.concatenate([enc, np.zeros(CODE, np.float32)])).astype(np.float32)
    s = uni.W @ q
    if mode == "top1":
        dec_code = np.maximum(uni.W[int(np.argmax(s)), CODE:], 0.0)
    else:
        w = np.maximum(s, 0.0) ** 4
        w = w / (w.sum() + 1e-9)
        dec_code = np.maximum((w[:, None] * uni.W[:, CODE:]).sum(axis=0), 0.0)
    return render_decoder(dec_code, D1, D2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, _, Xte, yte = load_data()
    picks = [int(np.where(yte == c)[0][0]) for c in range(10)]

    conditions = []
    for dist_name, gen, seed in [("scribble", scribble, 500),
                                 ("noise", noise, 600)]:
        E1, E2, D1, D2, uni = build(gen, seed)
        print(f"{dist_name}: trained", flush=True)
        for mode in ["top1", "graded"]:
            recs, corrs = [], []
            for c, idx in enumerate(picks):
                img = Xte[idx]
                rec = reconstruct(img, E1, E2, D1, D2, uni, mode)
                recs.append(rec)
                corrs.append(float(cn(rec.ravel()) @ cn(img.ravel())))
            conditions.append((f"{dist_name} {mode}", recs, corrs))
            print(f"{dist_name} {mode}: mean corr(input)="
                  f"{np.mean(corrs):.3f}", flush=True)

    fig, axes = plt.subplots(len(conditions) + 1, 10,
                             figsize=(10.5, 1.15 * (len(conditions) + 1) + 0.6))
    for c, idx in enumerate(picks):
        axes[0, c].imshow(Xte[idx], cmap="gray")
        axes[0, c].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=7, rotation=0, ha="right",
                          va="center")
    for ri, (name, recs, corrs) in enumerate(conditions):
        for c in range(10):
            ax = axes[ri + 1, c]
            ax.imshow(recs[c], cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[ri + 1, 0].set_ylabel(name.replace(" ", "\n"), fontsize=7,
                                   rotation=0, ha="right", va="center")
    fig.suptitle("Digits through a bridge trained ONLY on random pairs")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "generalization.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Bridge generalization: random training pairs, digit tests",
         "",
         "| condition | mean corr to input |",
         "|---|---|"]
        + [f"| {name} | {np.mean(corrs):.3f} |"
           for name, _, corrs in conditions]
        + ["", "Figure: generalization.png"]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
