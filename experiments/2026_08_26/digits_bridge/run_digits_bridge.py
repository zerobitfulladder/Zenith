"""Digit carousel with SPLIT vocabularies: see with the eye, draw with
the hand, and re-see your own drawing (the honest loop).

Encoder E1 and decoder D1: same geometry, different seeds — disjoint
template vocabularies. Temporal bank (identical machinery to the
validated carousel): [enc trails ; label ; NEXT frame in DECODER
vocabulary ; next label]. Free-run: emit hand-code -> RENDER pixels
via D1 -> RE-ENCODE via E1 -> trails update from the re-perception.
No code bypasses the pixel world. The self-recognition gap is now a
live circuit element.

Run:  .venv/bin/python experiments/2026_08_26/digits_bridge/run_digits_bridge.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_4layer_topk import load_data  # noqa: E402
from run_temporal_digits import (  # noqa: E402
    ALPHA_F,
    ALPHA_S,
    CODE_DIM,
    GG,
    HOLD,
    JDIM,
    K2,
    POS,
    SeqDict,
    WIN,
    cn,
    cn_rows,
    encode,
    render,
)

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "digits_bridge" / "results"
K1 = 64
L1_DIGITS = 1000
N_CYCLES = 1200
FREE_RUN = 5 * 10 * HOLD


def train_vocab(X, order, seed):
    bank = SeqDict(K1, WIN * WIN, 0.05, seed=seed)
    for i in order:
        V = np.stack([X[i][r:r + WIN, c:c + WIN].ravel() for r, c in POS])
        Vh, ok = cn_rows(V)
        for v in Vh[ok]:
            bank.step(v.astype(np.float32))
    return bank.W


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, _, _ = load_data()
    pools = [np.where(ytr == c)[0] for c in range(10)]

    def exemplar(cyc, c):
        return Xtr[pools[c][cyc % len(pools[c])]]

    rngE = np.random.default_rng(10)
    rngD = np.random.default_rng(20)
    E1 = train_vocab(Xtr, rngE.permutation(L1_DIGITS), seed=11)
    D1 = train_vocab(Xtr, rngD.permutation(L1_DIGITS), seed=22)
    print("vocabularies trained (eye and hand, disjoint seeds)", flush=True)

    d2 = SeqDict(K2, JDIM, 0.05, seed=1)
    eye = np.eye(10, dtype=np.float32)
    F = np.zeros(CODE_DIM, dtype=np.float32)
    S = np.zeros(CODE_DIM, dtype=np.float32)
    cache = {}

    def codes_of(cyc, c):
        key = (cyc, c)
        if key not in cache:
            if len(cache) > 40:
                cache.clear()
            img = exemplar(cyc, c)
            cache[key] = (encode(img, E1), encode(img, D1))
        return cache[key]

    for cyc in range(N_CYCLES):
        for c in range(10):
            enc_now, _ = codes_of(cyc, c)
            for h in range(HOLD):
                F = ALPHA_F * enc_now + (1 - ALPHA_F) * F
                S = ALPHA_S * enc_now + (1 - ALPHA_S) * S
                last = h == HOLD - 1
                nc_class = (c + 1) % 10 if last else c
                nc_cyc = cyc + 1 if (last and c == 9) else cyc
                _, dec_next = codes_of(nc_cyc, nc_class)
                d2.step(cn(np.concatenate(
                    [F, GG * S, GG * eye[c], GG * dec_next,
                     GG * eye[nc_class]])).astype(np.float32))
    print("temporal trained", flush=True)

    # ---- Cold-start free-run through the PIXEL loop ----------------------
    F = np.zeros(CODE_DIM, dtype=np.float32)
    S = np.zeros(CODE_DIM, dtype=np.float32)
    lab = eye[0].copy()
    gen, gen_lab = [], []
    for _ in range(FREE_RUN):
        q = cn(np.concatenate(
            [F, GG * S, GG * lab, np.zeros(CODE_DIM, np.float32),
             np.zeros(10, np.float32)])).astype(np.float32)
        u = int(np.argmax(d2.W @ q))
        row = d2.W[u]
        dec_code = np.maximum(row[2 * CODE_DIM + 10:3 * CODE_DIM + 10], 0.0)
        dec_code = (dec_code / (np.linalg.norm(dec_code) + 1e-9)
                    ).astype(np.float32)
        nl = int(np.argmax(row[3 * CODE_DIM + 10:]))
        img = render(dec_code, D1)          # the hand draws
        seen = encode(img, E1)              # the eye re-sees the drawing
        gen.append(img)
        gen_lab.append(nl)
        F = ALPHA_F * seen + (1 - ALPHA_F) * F
        S = ALPHA_S * seen + (1 - ALPHA_S) * S
        lab = eye[nl].copy()
    print("emitted labels (first 80):",
          "".join(str(l) for l in gen_lab[:80]), flush=True)

    class_means = [Xtr[pools[c][:N_CYCLES]].mean(axis=0) for c in range(10)]
    picks = {}
    for i, l in enumerate(gen_lab):
        if i % HOLD == HOLD // 2:
            picks.setdefault(l, []).append(gen[i])
    n_show = min(5, max((len(v) for v in picks.values()), default=0))
    fig, axes = plt.subplots(n_show + 1, 10,
                             figsize=(10.5, 1.1 * (n_show + 1) + 0.6))
    for c in range(10):
        axes[0, c].imshow(class_means[c], cmap="gray")
        axes[0, c].axis("off")
        for r in range(n_show):
            ax = axes[r + 1, c]
            if c in picks and r < len(picks[c]):
                ax.imshow(picks[c][r], cmap="gray")
            ax.axis("off")
    fig.suptitle("Split-vocabulary carousel — row 0: class means; rows 1+: "
                 "the hand's digits per generated cycle (seen by the eye)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cycles.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Split-vocabulary digit carousel (see -> draw -> re-see)",
         "",
         f"Labels (first 80): {''.join(str(l) for l in gen_lab[:80])}",
         "",
         "Reference (shared-vocabulary carousel): loops all 10 digits,",
         "holds 1-6 frames, canonical drawings.",
         "Figure: cycles.png"]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
