"""Conv L1 as a CODEC: sweep the row balance (current window vs trail).

Diagnosis this tests (probes in the 08-27 log): the conv temporal unit's
row is 74% lagged trail / 26% arriving window, so on the carousel — where
the local history at a position is a blur of ten different digits — units
get indexed by an uninformative history and each one's cargo becomes an
average over unrelated windows. Measured: per-window cargo fidelity 0.232,
whole-frame reconstruction 0.516, decoder ceiling 0.26 (judge on real
frames 0.87). Everything downstream is scored through that ceiling.

Arms (one axis: the trail's share of the L1 row, plus one capacity control)
  base     8x8 K=64  GC=0.5   trail 74%   <- today's rig, reference
  gc2      8x8 K=64  GC=2.0   trail 33%   <- cargo gets more of the row
  gc4      8x8 K=64  GC=4.0   trail 20%   <- window dominates, trail = context
  notrail  8x8 K=64  no trail             <- L1 = pure spatial dictionary
  w4k36    4x4 K=36  GC=2.0   trail 33%   <- validated dictionary geometry
  w4notrail 4x4 K=36 no trail             <- best of both, added after the sweep

L1 is never queried with an empty half in this rig (it only encodes and
renders; the cargo-empty successor read happens at L2), so the "key must
dominate the row" law does not bind here — that split was inherited from
the uniform-unit template, not required by the read.

Gate FIRST (codec fidelity, no memory involved), then the L2 successor
read + bounce generation on whatever survives. Every arm writes templates,
a reconstruction panel/gif, and generation gifs.

Run:  CODEC_ARM=gc2 .venv/bin/python experiments/2026_08_27/conv_codec/run_conv_codec.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_27" / "completion"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_square_completion import cn1            # noqa: E402  (sets sys.path)
from run_temporal_square import save_gif         # noqa: E402
from run_gpu_minibatch import Dict               # noqa: E402

ARMS = {
    "base":    dict(win=8, k1=64, gc=0.5, trail=True),
    "gc2":     dict(win=8, k1=64, gc=2.0, trail=True),
    "gc4":     dict(win=8, k1=64, gc=4.0, trail=True),
    "notrail": dict(win=8, k1=64, gc=None, trail=False),
    "w4k36":   dict(win=4, k1=36, gc=2.0, trail=True),
    "w4notrail": dict(win=4, k1=36, gc=None, trail=False),
}
ARM = os.environ.get("CODEC_ARM", "base")
LOAD = os.environ.get("CODEC_LOAD") == "1"   # re-run eval on saved weights
CFG = ARMS[ARM]
WIN, K1, GC, TRAIL = CFG["win"], CFG["k1"], CFG["gc"], CFG["trail"]

SIDE = 28
NP1 = SIDE - WIN + 1
NPOS, WD = NP1 * NP1, WIN * WIN
CODE_D = NPOS * K1
ROW_D = 2 * WD if TRAIL else WD
K2 = 32
G1, G2, GC2 = 0.5, 0.5, 0.5
EPS, SQ = 1e-6, 0.10
EPOCHS = int(os.environ.get("CODEC_EPOCHS", "2"))
TRAIN_TICKS = int(os.environ.get("CODEC_TICKS", "2000"))
PRIME, FREE = 50, 100

OUTPUT_DIR = (ROOT / "experiments" / "2026_08_27"
              / "conv_codec" / "results" / ARM)
_t = np.ones(WIN)
_t[0] = _t[-1] = 0.5
FEATHER = np.outer(_t, _t)

X = np.load(ROOT / "data/mnist/digits/train_images.npy")
y = np.load(ROOT / "data/mnist/digits/train_labels.npy")
BYCLASS = [X[y == c] for c in range(10)]
CM_N = np.stack([cn1(b.mean(axis=0).ravel()) for b in BYCLASS])


def frame_for(t):
    c, lap = t % 10, t // 10
    return BYCLASS[c][lap % len(BYCLASS[c])].astype(np.float32)


def classify(f):
    return int(np.argmax(CM_N @ cn1(np.asarray(f).ravel())))


def corr(a, b):
    return float(cn1(np.asarray(a).ravel()) @ cn1(np.asarray(b).ravel()))


def cn_rows(M):
    M = M - M.mean(axis=1, keepdims=True)
    return (M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True),
                           1e-9)).astype(np.float32)


def win_view(frame):
    s = np.lib.stride_tricks.sliding_window_view(frame, (WIN, WIN))
    return s.reshape(NPOS, WD)


class L1:
    """The conv unit as a codec: choose a template per position, render back."""

    def __init__(self):
        self.bank = Dict(K1, ROW_D, 0.05)
        self.reset()

    def reset(self):
        self.T = np.zeros((NPOS, WD), np.float32)

    def row(self, wins, live):
        if TRAIL:
            return cn_rows(np.concatenate(
                [GC * cn_rows(wins[live]), cn_rows(self.T[live])], axis=1))
        return cn_rows(wins[live])

    def encode(self, frame, learning):
        """-> graded code, skeleton code, (live, winners) for fidelity."""
        wins = win_view(frame)
        alive = wins.max(axis=1) > EPS
        if TRAIL:
            alive |= self.T.max(axis=1) > EPS
        live = np.where(alive)[0]
        cg = np.zeros((NPOS, K1), np.float32)
        cs = np.zeros((NPOS, K1), np.float32)
        ws = np.zeros(0, np.int64)
        if len(live):
            Z = self.row(wins, live)
            if learning and self.bank.n_boot < K1:
                self.bank.bootstrap(Z[: K1 - self.bank.n_boot])
            if self.bank.n_boot >= K1:
                C = Z @ self.bank.W.T
                ws = np.argmax(C, axis=1)
                cv = np.maximum(C[np.arange(len(live)), ws], 0.0)
                if learning:
                    self.bank.update(Z, ws.astype(np.int64),
                                     cv.astype(np.float32))
                cg[live] = np.maximum(C, 0.0)
                cs[live, ws] = cv
        if TRAIL:
            self.T *= G1
            self.T += wins
        return cg, cs, live, ws, wins

    @property
    def cargo(self):
        return np.maximum(self.bank.W[:, :WD] if TRAIL else self.bank.W, 0.0)


def render(prof, basis, mode="graded"):
    """Validated recipe: per-position squelch, peak gating, confidence-
    weighted feathered overlap-add."""
    num = np.zeros((SIDE, SIDE))
    den = np.zeros((SIDE, SIDE))
    peak = prof.max()
    if peak <= 0:
        return num
    for p in range(NPOS):
        seg = np.maximum(prof[p], 0.0)
        m = seg.max()
        if m <= 0 or m < SQ * peak:
            continue
        if mode == "committed":
            s = np.zeros_like(seg)
            s[int(np.argmax(seg))] = m
            seg = s
        else:
            seg = np.where(seg < SQ * m, 0.0, seg)
        patch = (seg @ basis).reshape(WIN, WIN)
        r, c = divmod(p, NP1)
        num[r:r + WIN, c:c + WIN] += patch * FEATHER * m
        den[r:r + WIN, c:c + WIN] += FEATHER * m
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
    return out / out.max() if out.max() > 0 else out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{ARM}] win={WIN} K1={K1} GC={GC} trail={TRAIL} "
          f"NPOS={NPOS} CODE_D={CODE_D}", flush=True)

    # ---- Phase A: L1 alone -------------------------------------------------
    l1 = L1()
    saved = np.load(OUTPUT_DIR / "weights.npz") if LOAD else None
    if LOAD:
        l1.bank.W = saved["W1"]
        l1.bank.n_boot = K1
        print(f"[{ARM}] loaded L1 from weights.npz", flush=True)
    else:
        for ep in range(EPOCHS):
            l1.reset()
            for t in range(TRAIN_TICKS):
                l1.encode(frame_for(t), True)
            print(f"[{ARM}] L1 epoch {ep + 1}/{EPOCHS} done", flush=True)

    basis = l1.cargo
    if TRAIL:
        km = np.linalg.norm(l1.bank.W[:, WD:], axis=1)
        cm = np.linalg.norm(l1.bank.W[:, :WD], axis=1)
        share = float(np.mean(cm / (km + cm)))
    else:
        share = 1.0

    # ---- Gate: is L1 a codec? ---------------------------------------------
    l1.reset()
    fid, rc_c, rc_g, ceil_c, ceil_g, use = [], [], [], [], [], np.zeros(K1)
    panel, gif = {}, []
    for t in range(150):
        frame = frame_for(t)
        cg, cs, live, ws, wins = l1.encode(frame, False)
        if t < 50 or not len(live):
            continue
        for u in ws:
            use[u] += 1
        A = cn_rows(wins[live])
        B = cn_rows(basis[ws])
        fid.extend((A * B).sum(axis=1).tolist())
        rec_c = render(cs, basis, "committed")
        rec_g = render(cg, basis, "graded")
        rc_c.append(corr(rec_c, frame))
        rc_g.append(corr(rec_g, frame))
        ceil_c.append(classify(rec_c) == t % 10)
        ceil_g.append(classify(rec_g) == t % 10)
        if t % 10 not in panel:
            panel[t % 10] = (frame, rec_c, rec_g)
        if len(gif) < 40:
            gif.append(np.concatenate([frame / max(frame.max(), 1e-9), rec_c],
                                      axis=1))
    judge = float(np.mean([classify(frame_for(t)) == t % 10
                           for t in range(50, 150)]))
    m = dict(arm=ARM, win=WIN, k1=K1, gc=GC, trail=TRAIL,
             cargo_share_of_row=round(share, 3),
             window_fidelity=round(float(np.mean(fid)), 3),
             recon_corr_committed=round(float(np.mean(rc_c)), 3),
             recon_corr_graded=round(float(np.mean(rc_g)), 3),
             decoder_ceiling_committed=round(float(np.mean(ceil_c)), 3),
             decoder_ceiling_graded=round(float(np.mean(ceil_g)), 3),
             judge_on_real_frames=round(judge, 3),
             l1_units_used=int((use > 0).sum()),
             l1_top_unit_share=round(float(use.max() / max(use.sum(), 1)), 3))
    print(f"[{ARM}] GATE {json.dumps(m)}", flush=True)

    save_gif(gif, OUTPUT_DIR / "recon.gif", scale=6, ms=400)
    fig, axes = plt.subplots(3, 10, figsize=(12, 4))
    for c in range(10):
        for r, im in enumerate(panel[c]):
            axes[r, c].imshow(im, cmap="gray")
            axes[r, c].axis("off")
        axes[0, c].set_title(str(c), fontsize=9)
    for r, lab in enumerate(["input", "committed", "graded"]):
        axes[r, 0].set_ylabel(lab)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
    fig.suptitle(f"{ARM}: reconstruction through the L1 codec "
                 f"(window fidelity {m['window_fidelity']}, "
                 f"frame corr {m['recon_corr_committed']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "recon_panel.png", dpi=140)
    plt.close(fig)

    ncol = 16 if K1 >= 32 else 12
    nrow = int(np.ceil(K1 / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.1, nrow * 1.5))
    axes = np.atleast_2d(axes)
    for i in range(nrow * ncol):
        ax = axes[i // ncol, i % ncol]
        ax.axis("off")
        if i >= K1:
            continue
        cargo = basis[i].reshape(WIN, WIN)
        cargo = cargo / max(cargo.max(), 1e-9)
        if TRAIL:
            keys = np.maximum(l1.bank.W[i, WD:], 0.0).reshape(WIN, WIN)
            stack = np.vstack([keys / max(keys.max(), 1e-9),
                               np.full((1, WIN), 0.5), cargo])
        else:
            stack = cargo
        ax.imshow(stack, cmap="inferno", vmin=0, vmax=1)
    fig.suptitle(f"{ARM}: L1 units — "
                 + ("KEYS (trail, top) / CARGO (window, bottom)"
                    if TRAIL else "spatial templates"), fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "templates_L1.png", dpi=140)
    plt.close(fig)

    # ---- L2 over the code (staged: fresh bank on frozen L1) ---------------
    b2 = Dict(K2, 2 * CODE_D, 0.05)

    def l2_row(code, T2):
        return cn1(np.concatenate([GC2 * cn1(code), cn1(T2)]))

    wins_hist = np.zeros(K2)
    if LOAD:
        b2.W = saved["W2"]
        b2.n_boot = K2
        old = json.loads((OUTPUT_DIR / "metrics.json").read_text())
        wins_hist[0] = 1.0   # placeholder; real values carried over below
    for _ in range(0 if LOAD else 2):
        l1.reset()
        T2 = np.zeros(CODE_D, np.float32)
        for t in range(TRAIN_TICKS):
            _, cs, _, _, _ = l1.encode(frame_for(t), False)
            code = cs.ravel()
            z = l2_row(code, T2)[None, :]
            if b2.n_boot < K2:
                b2.bootstrap(z)
            else:
                C = (z @ b2.W.T)[0]
                w = int(np.argmax(C))
                b2.update(z, np.array([w]),
                          np.array([max(C[w], 0.0)], np.float32))
                wins_hist[w] += 1
            T2 = code + G2 * T2
    if LOAD:
        m["l2_rows_ever_win"] = old["l2_rows_ever_win"]
        m["l2_top_row_share"] = old["l2_top_row_share"]
    else:
        m["l2_rows_ever_win"] = int((wins_hist > 0).sum())
        m["l2_top_row_share"] = round(float(wins_hist.max()
                                            / max(wins_hist.sum(), 1)), 3)
        np.savez(OUTPUT_DIR / "weights.npz", W1=l1.bank.W, W2=b2.W)

    def successor(T2):
        q = cn1(np.concatenate([np.zeros(CODE_D, np.float32), cn1(T2)]))
        w2 = int(np.argmax(b2.W @ q))
        return w2, np.maximum(b2.W[w2, :CODE_D], 0.0).reshape(NPOS, K1)

    # teacher-forced: judged in CODE space (no decoder in the loop) and in
    # pixels (the decoder-limited number the old report used)
    l1.reset()
    T2 = np.zeros(CODE_D, np.float32)
    acc = {c: [] for c in range(10)}
    for t in range(400):
        _, cs, _, _, _ = l1.encode(frame_for(t), False)
        acc[t % 10].append(cs.ravel())
    CODE_MEAN_N = np.stack([cn1(np.mean(v, axis=0)) for _, v in sorted(acc.items())])

    l1.reset()
    T2 = np.zeros(CODE_D, np.float32)
    tf_code, tf_pix, ws2, trails, labs = [], [], set(), [], []
    for t in range(150):
        if t >= PRIME:
            w2, prof = successor(T2)
            ws2.add(w2)
            tf_code.append(int(np.argmax(CODE_MEAN_N @ cn1(prof.ravel())))
                           == t % 10)
            tf_pix.append(classify(render(prof, basis, "graded")) == t % 10)
            trails.append(cn1(T2))
            labs.append(t % 10)
        _, cs, _, _, _ = l1.encode(frame_for(t), False)
        T2 = cs.ravel() + G2 * T2
    Tr, labs = np.stack(trails), np.array(labs)
    means = np.stack([cn1(Tr[labs == c].mean(axis=0)) for c in range(10)])
    m["phase_from_code_trail"] = round(
        float((np.argmax(Tr @ means.T, axis=1) == labs).mean()), 3)
    m["tf_read_in_code_space"] = round(float(np.mean(tf_code)), 3)
    m["tf_through_decoder"] = round(float(np.mean(tf_pix)), 3)
    m["tf_distinct_winners"] = len(ws2)

    # ---- Bounce generation -------------------------------------------------
    for mode in ("graded", "committed"):
        l1.reset()
        T2 = np.zeros(CODE_D, np.float32)
        for t in range(PRIME):
            _, cs, _, _, _ = l1.encode(frame_for(t), False)
            T2 = cs.ravel() + G2 * T2
        gen, cls, bw = [], [], []
        for _ in range(FREE):
            w2, prof = successor(T2)
            bw.append(w2)
            em = render(prof, basis, mode)
            gen.append(em)
            cls.append(classify(em))
            _, cs, _, _, _ = l1.encode(em.astype(np.float32), False)
            T2 = cs.ravel() + G2 * T2
        cls = np.array(cls)
        m[f"bounce_{mode}_distinct_winners"] = len(set(bw))
        m[f"bounce_{mode}_top_winner_share"] = round(
            float(max(bw.count(w) for w in set(bw)) / len(bw)), 3)
        m[f"bounce_{mode}_advance"] = round(
            float(np.mean((np.diff(cls) % 10) == 1)), 3)
        m[f"bounce_{mode}_timeline"] = round(
            float(np.mean(cls == (np.arange(PRIME, PRIME + FREE) % 10))), 3)
        m[f"bounce_{mode}_first30"] = "".join(map(str, cls[:30]))
        save_gif(gen, OUTPUT_DIR / f"generated_{mode}.gif", scale=8, ms=350)
        fig, axes = plt.subplots(2, 15, figsize=(15, 2.6))
        for i in range(30):
            ax = axes[i // 15, i % 15]
            ax.imshow(gen[i], cmap="gray")
            ax.set_title(f"{cls[i]}", fontsize=8, color="black")
            ax.axis("off")
        fig.suptitle(f"{ARM} / {mode}: free-run emissions 1-30 "
                     f"(judged class above; advance "
                     f"{m[f'bounce_{mode}_advance']})", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"generated_panel_{mode}.png", dpi=140)
        plt.close(fig)

    print(f"[{ARM}] FULL {json.dumps(m)}", flush=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(m, indent=2))
    (OUTPUT_DIR / "report.md").write_text(
        f"# Conv codec arm `{ARM}`\n\n"
        f"win {WIN}x{WIN}, K1={K1}, GC={GC}, trail={TRAIL}\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in m.items()) + "\n")


if __name__ == "__main__":
    main()
