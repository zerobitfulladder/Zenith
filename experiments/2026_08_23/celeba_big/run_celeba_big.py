"""Big faces run: 5x archive, deeper rehearsal, and zero-pair composition.

Dictionaries 48/96/128, top archive 1000 memories (~51M params), 6 epochs,
clean births. Generation: (a) existing-label portraits via rehearsal-gated
contrast retrieval; (b) attribute pairs with ZERO occurrences in the data,
composed as base-memory + purified attribute delta (delta computed within
its natural gender to remove the Male confound), hardened and rendered.

Run:  .venv/bin/python experiments/2026_08_23/celeba_big/run_celeba_big.py
"""

import json
import os
import time
from pathlib import Path

os.environ["GF_REPORT"] = "sparselearn"
os.environ["GF_SIDE"] = "48"
os.environ["GF_W1_STR"] = "1"
os.environ["GF_K1"] = "48"
os.environ["GF_K2L"] = "96"
os.environ["GF_K3"] = "128"
os.environ["GF_KTOP"] = "1000"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rig"))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "celeba_faces"))  # noqa: E402

from run_gpu_minibatch import B, Dict, XP_NAME, _center_norm_rows, level_pass, xp  # noqa: E402
from run_4layer_topk import (  # noqa: E402
    CODE3_DIM, ETA1, ETA2, ETA3, ETATOP, G1, G2, G3, K1, K2, K3, KTOP,
    K_OUT1, K_OUT2, K_OUT3, LAM, W2_STR, W2_WIN, W3_STR, W3_WIN,
    encode_over_batch, encode_pixels_batch, expand, render_pixels,
)
from gain_feedback import center_norm  # noqa: E402
from run_celeba_faces import harden_map, ATTR_NAMES, A_IDX, attr_query  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "2026_08_23" / "celeba_big" / "results"
EPOCHS = 6
DELTA_N = 2000
LABEL_DIM = len(ATTR_NAMES)
TOP_DIM = CODE3_DIM + LABEL_DIM

PORTRAITS = [
    ["Male", "No_Beard"], ["Male", "Mustache"], ["Male", "Eyeglasses"],
    ["Male", "Wearing_Hat"], ["Wearing_Lipstick", "Smiling"],
    ["Heavy_Makeup", "Blond_Hair"], ["Lipstick_placeholder"],  # replaced below
]
PORTRAITS = [
    ["Male", "No_Beard"], ["Male", "Mustache"], ["Male", "Eyeglasses"],
    ["Male", "Wearing_Hat"], ["Wearing_Lipstick", "Smiling"],
    ["Heavy_Makeup", "Blond_Hair"], ["Wearing_Lipstick", "Wavy_Hair"],
    ["Bald", "Male"], ["Male", "Goatee"], ["Heavy_Makeup", "Young"],
]

# (base attrs for retrieval, delta attr, gender to purify delta within)
ZERO_PAIRS = [
    (["Wearing_Lipstick", "Smiling"], "Mustache", "male"),
    (["Heavy_Makeup", "Young"], "Bald", "male"),
    (["Wearing_Lipstick", "Wavy_Hair"], "Goatee", "male"),
    (["Male", "Mustache"], "Heavy_Makeup", "female"),
]


def render_code3(code3, b1, b2, b3):
    m2 = harden_map(expand(harden_map(code3), b3, W3_WIN, W3_STR, (G2, G2, K2)))
    m1 = harden_map(expand(m2, b2, W2_WIN, W2_STR, (G1, G1, K1)))
    return render_pixels(m1, b1)


def POS(g_in, win, stride):
    return [(r, c) for r in range(0, g_in - win + 1, stride)
            for c in range(0, g_in - win + 1, stride)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(ROOT / "data/celeba/images_48.npy")
    A = np.load(ROOT / "data/celeba/attrs_48.npy").astype(np.float64)
    N = len(X)

    for base, dattr, _ in ZERO_PAIRS:
        m = np.ones(N, bool)
        for n in base:
            m &= A[:, A_IDX[n]] > 0.5
        m &= A[:, A_IDX[dattr]] > 0.5
        print(f"census {'+'.join(base)}+{dattr}: {int(m.sum())} faces", flush=True)

    d1 = Dict(K1, 16, ETA1)
    d2 = Dict(K2, W2_WIN * W2_WIN * K1, ETA2)
    d3 = Dict(K3, W3_WIN * W3_WIN * K2, ETA3)
    top = Dict(KTOP, TOP_DIM, ETATOP)
    p1, p2, p3 = POS(48, 4, 1), POS(G1, W2_WIN, W2_STR), POS(G2, W3_WIN, W3_STR)

    Xx = xp.asarray(X.astype(np.float64))
    Lx = xp.asarray(A * LAM)

    t0 = time.time()
    for ep in range(EPOCHS):
        for s in range(0, N, B):
            xb = Xx[s:s + B]
            lb = Lx[s:s + B]
            m1 = level_pass(xb[..., None], d1, p1, 4, 1, G1, False, True)
            m2 = level_pass(m1, d2, p2, W2_WIN, K1, G2, True, True)
            m3 = level_pass(m2, d3, p3, W3_WIN, K2, G3, True, True)
            H = m3.reshape(len(xb), -1)
            H, okh = _center_norm_rows(H)
            Z = xp.concatenate([H, lb], axis=1)
            Z, okz = _center_norm_rows(Z)
            okz = okz & okh
            if top.n_boot < top.k:
                top.bootstrap(Z[okz])
            if top.n_boot >= top.k:
                Ct = Z @ top.W.T
                winners = xp.argmax(Ct, axis=1)
                cvals = xp.max(Ct, axis=1) * okz
                keep = cvals > 0
                top.update(Z[keep], winners[keep], cvals[keep])
        print(f"epoch {ep + 1}/{EPOCHS} {time.time() - t0:.0f}s", flush=True)
    if XP_NAME == "cupy":
        xp.cuda.Stream.null.synchronize()
    train_s = time.time() - t0

    def to_np(a):
        return a.get() if XP_NAME == "cupy" else a

    W1n, W2n, W3n, Wtn = map(to_np, (d1.W, d2.W, d3.W, top.W))
    wins = to_np(top.win_counts)
    np.savez(OUT / "weights.npz", W1=W1n, W2=W2n, W3=W3n, Wtop=Wtn, top_wins=wins)

    class Bank:
        def __init__(self, W):
            self.W = W
            self.k = W.shape[0]

    b1, b2, b3 = Bank(W1n), Bank(W2n), Bank(W3n)
    L = Wtn[:, CODE3_DIM:]
    Lc = L - L.mean(axis=0, keepdims=True)
    idf = 1.0 / np.sqrt(A.mean(axis=0) + 0.01)
    rehearsed = wins >= np.median(wins)

    def retrieve(names):
        q = np.zeros(LABEL_DIM)
        for n in names:
            q[A_IDX[n]] = idf[A_IDX[n]]
        score = Lc @ q
        score[~rehearsed] = -1e9
        return int(np.argmax(score))

    fig, axes = plt.subplots(2, 5, figsize=(12, 5.4))
    dwin = []
    for ax, names in zip(axes.flat, PORTRAITS):
        win = retrieve(names)
        dwin.append(win)
        code3 = np.maximum(Wtn[win, :CODE3_DIM], 0.0).reshape(G3, G3, K3)
        ax.imshow(render_code3(code3, b1, b2, b3), cmap="gray")
        ax.set_title("+".join(n.replace("Wearing_", "") for n in names) + f" m{win}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Portraits — big archive (1000 memories)")
    fig.tight_layout()
    fig.savefig(OUT / "portraits.png", dpi=110)
    plt.close(fig)
    print(f"portrait winners distinct: {len(set(dwin))}/10", flush=True)

    # ---- Purified deltas from an encoded subset ---------------------------
    sub = X[:DELTA_N].astype(np.float64)
    M1 = encode_pixels_batch(sub, b1, K_OUT1)
    M2 = encode_over_batch(M1, b2, W2_WIN, W2_STR, G2, K_OUT2)
    M3 = encode_over_batch(M2, b3, W3_WIN, W3_STR, G3, K_OUT3)
    C3 = M3.reshape(DELTA_N, -1).astype(np.float64)
    Asub = A[:DELTA_N]
    male = Asub[:, A_IDX["Male"]] > 0.5

    def purified_delta(attr, gender):
        gmask = male if gender == "male" else ~male
        i = A_IDX[attr]
        pos = C3[gmask & (Asub[:, i] > 0.5)]
        neg = C3[gmask & (Asub[:, i] < 0.5)]
        print(f"delta {attr} ({gender}): n+={len(pos)} n-={len(neg)}", flush=True)
        return pos.mean(axis=0) - neg.mean(axis=0)

    fig, axes = plt.subplots(len(ZERO_PAIRS), 4, figsize=(11, 2.8 * len(ZERO_PAIRS)))
    for row, (base_names, dattr, gender) in enumerate(ZERO_PAIRS):
        win = retrieve(base_names)
        base = np.maximum(Wtn[win, :CODE3_DIM], 0.0)
        dv = purified_delta(dattr, gender)
        dv = dv / (np.linalg.norm(dv) + 1e-9)
        bnorm = np.linalg.norm(base)
        for col, beta in enumerate([0.0, 0.3, 0.5, 0.8]):
            comp = np.maximum(base + beta * bnorm * dv, 0.0).reshape(G3, G3, K3)
            ax = axes[row, col]
            ax.imshow(render_code3(comp, b1, b2, b3), cmap="gray")
            if row == 0:
                ax.set_title(f"+{beta}", fontsize=9)
            if col == 0:
                ax.set_ylabel("+".join(n.replace("Wearing_", "") for n in base_names)
                              + f"\n+ {dattr}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Zero-count pairs, composed via purified deltas")
    fig.tight_layout()
    fig.savefig(OUT / "zero_pairs.png", dpi=110)
    plt.close(fig)

    print(f"RESULT train_s={train_s:.0f} figures in {OUT}", flush=True)
    (OUT / "report.md").write_text(
        f"# Big faces run\n\nK={K1}/{K2}/{K3}/{KTOP}, epochs {EPOCHS}, "
        f"train {train_s:.0f}s, code3 {CODE3_DIM}, top params {KTOP * TOP_DIM:,}.\n"
        "Figures: portraits.png, zero_pairs.png\n")


if __name__ == "__main__":
    main()
