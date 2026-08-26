"""Temporal v5: compositional generalization — 4 lines, 4 bits, held-out combos.

Stimulus: 16x16 canvas, four rotating lines (one per 8x8 quadrant),
10 deg/frame each; a 4-bit label gives each line's direction (1=CW).
Train on ALL combos with <= 2 bits set (11 of 16); NEVER show 1110 or
1111; then cold-start generate them from the label alone.

Arms:
  G — the current architecture: one GLOBAL temporal bank over the
      whole-frame code. User's prediction: cannot compose (a top-1
      winner can only emit a stored whole-frame; unseen conjunctions
      have none). Expected failure signature: locks onto the
      hamming-nearest TRAINED combo, <= 3/4 directions right.
  L — LOCAL temporal banks, one per quadrant (own code slice, own
      label bit, own trails, own emission; frame = assembly).
      Prediction: 4/4 on both held-out combos — the unseen global
      conjunction is locally familiar everywhere. Top-1 is not the
      blocker; global scope is. (Locality is hard-wired here, as it
      is in the spatial rigs' windows; discovering it is future work.)

Run:  .venv/bin/python experiments/2026_08_26/temporal_compose/run_temporal_compose.py
"""

import os
import sys
from itertools import product
from pathlib import Path

os.environ.setdefault("GF_W1_STR", "1")
os.environ["GF_XP"] = "numpy"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_23" / "rig"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_digits"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "temporal_square"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_temporal_digits import SeqDict, cn  # noqa: E402
from run_temporal_square import save_gif  # noqa: E402

OUTPUT_DIR = ROOT / "experiments" / "2026_08_26" / "temporal_compose" / "results"
H, Q = 16, 8
QUADS = [(0, 0), (0, 8), (8, 0), (8, 8)]
DEG = 10
PERIOD = 180 // DEG                 # 18 frames
WIDTH, RADIUS = 1.0, 3.0
K1 = 64
KG, KL = 256, 48
ETA = 0.05
ALPHA_F, ALPHA_S = 0.35, 0.06
GG = 0.5
TICKS = 540                         # 30 cycles per combo, interleaved
FREE_RUN = 72
TRAIN_COMBOS = [c for c in product((0, 1), repeat=4) if sum(c) <= 2]
TEST_COMBOS = [(1, 1, 1, 0), (1, 1, 1, 1)]


def quad_line(t, cw):
    a = np.deg2rad(((1 if cw else -1) * DEG * t) % 180)
    d = np.array([np.cos(a), np.sin(a)])
    yy, xx = np.mgrid[0:Q, 0:Q]
    p = np.stack([xx - (Q - 1) / 2, yy - (Q - 1) / 2], axis=-1)
    along = p @ d
    perp = np.abs(p[..., 0] * d[1] - p[..., 1] * d[0])
    return (np.clip(1.0 - perp / WIDTH, 0, 1)
            * (np.abs(along) <= RADIUS)).astype(np.float32)


def frame(combo, t):
    f = np.zeros((H, H), dtype=np.float32)
    for (r, c), cw in zip(QUADS, combo):
        f[r:r + Q, c:c + Q] = quad_line(t, cw)
    return f


def encode_q(img8, W1):
    v = cn(img8.ravel().astype(np.float32))
    code = np.maximum(W1 @ v, 0.0)
    n = np.linalg.norm(code)
    return (code / n if n > 1e-9 else code).astype(np.float32)


def render_q(code, W1):
    if code.max() <= 0:
        return np.zeros((Q, Q))
    patch = W1[int(np.argmax(code))].reshape(Q, Q)
    v = np.maximum(patch, 0.0)
    return v / (v.max() + 1e-9)


def angle_q(img):
    yy, xx = np.mgrid[0:Q, 0:Q]
    m = img.sum()
    if m <= 0:
        return None
    cx, cy = (img * xx).sum() / m, (img * yy).sum() / m
    mu20 = (img * (xx - cx) ** 2).sum()
    mu02 = (img * (yy - cy) ** 2).sum()
    mu11 = (img * (xx - cx) * (yy - cy)).sum()
    return np.rad2deg(0.5 * np.arctan2(2 * mu11, mu20 - mu02)) % 180


def direction_of(quad_frames):
    angs = [angle_q(f) for f in quad_frames]
    ds = []
    for a, b in zip(angs, angs[1:]):
        if a is None or b is None:
            continue
        ds.append(((b - a + 90) % 180) - 90)
    return int(np.sign(np.median(ds))) if ds else 0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_gif([frame((1, 1, 0, 0), t) for t in range(2 * PERIOD)],
             OUTPUT_DIR / "input_1100.gif", scale=14)

    # ---- L1: shared 8x8 oriented templates -------------------------------
    d1 = SeqDict(K1, Q * Q, ETA)
    for t in range(600):
        for cw in (0, 1):
            v = cn(quad_line(t, cw).ravel().astype(np.float32))
            d1.step(v)
    W1 = d1.W

    # ---- Calibrate direction sign on true movies -------------------------
    sign_cw = direction_of([quad_line(t, 1) for t in range(PERIOD)])

    # ---- Arm G: global bank ----------------------------------------------
    GDIM = 3 * (4 * K1) + 4
    dg = SeqDict(KG, GDIM, ETA, seed=2)
    # Arm L: one bank per quadrant (oracle: sees ONLY its own bit)
    LDIM = 3 * K1 + 2
    dls = [SeqDict(KL, LDIM, ETA, seed=3 + i) for i in range(4)]
    # Arm H (user design v1): hypercolumns — per-quadrant banks that each
    # see the FULL 4-bit label IN the metric; association must discover
    # relevance. (Measured: fails — units fragment by whole-label context.)
    HDIM = 3 * K1 + 4
    dhs = [SeqDict(KL, HDIM, ETA, seed=7 + i) for i in range(4)]
    # Arm A (user design v2): label BESIDE the metric — units compete on
    # trail alone; a Hebbian tally per unit records the average label
    # under which it won; at generation the full label enters as a
    # retrieval bias. Marginal statistics do the relevance discovery.
    ADIM = 3 * K1
    das = [SeqDict(KL, ADIM, ETA, seed=11 + i) for i in range(4)]
    Aassoc = [np.zeros((KL, 4), dtype=np.float32) for _ in range(4)]
    Awins = [np.zeros(KL, dtype=np.float32) for _ in range(4)]
    BETA = 1.0    # 0.3 oscillated: bias must dominate direction choice
                  # (trail still picks the phase within the direction)

    stG = {c: (np.zeros(4 * K1, np.float32), np.zeros(4 * K1, np.float32))
           for c in TRAIN_COMBOS}
    stL = {(c, qi): (np.zeros(K1, np.float32), np.zeros(K1, np.float32))
           for c in TRAIN_COMBOS for qi in range(4)}
    stH = {(c, qi): (np.zeros(K1, np.float32), np.zeros(K1, np.float32))
           for c in TRAIN_COMBOS for qi in range(4)}
    stA = {(c, qi): (np.zeros(K1, np.float32), np.zeros(K1, np.float32))
           for c in TRAIN_COMBOS for qi in range(4)}
    eye2 = np.eye(2, dtype=np.float32)

    for t in range(TICKS):
        for combo in TRAIN_COMBOS:
            codes = [encode_q(quad_line(t, cw), W1) for cw in combo]
            ncodes = [encode_q(quad_line(t + 1, cw), W1) for cw in combo]
            gcode, gnext = np.concatenate(codes), np.concatenate(ncodes)
            F, S = stG[combo]
            F = ALPHA_F * gcode + (1 - ALPHA_F) * F
            S = ALPHA_S * gcode + (1 - ALPHA_S) * S
            stG[combo] = (F, S)
            lab = np.array(combo, dtype=np.float32)
            dg.step(cn(np.concatenate([F, GG * S, GG * lab, GG * gnext])
                       ).astype(np.float32))
            for qi in range(4):
                Fq, Sq = stL[(combo, qi)]
                Fq = ALPHA_F * codes[qi] + (1 - ALPHA_F) * Fq
                Sq = ALPHA_S * codes[qi] + (1 - ALPHA_S) * Sq
                stL[(combo, qi)] = (Fq, Sq)
                dls[qi].step(cn(np.concatenate(
                    [Fq, GG * Sq, GG * eye2[combo[qi]], GG * ncodes[qi]])
                ).astype(np.float32))
                Fh, Sh = stH[(combo, qi)]
                Fh = ALPHA_F * codes[qi] + (1 - ALPHA_F) * Fh
                Sh = ALPHA_S * codes[qi] + (1 - ALPHA_S) * Sh
                stH[(combo, qi)] = (Fh, Sh)
                dhs[qi].step(cn(np.concatenate(
                    [Fh, GG * Sh, GG * lab, GG * ncodes[qi]])
                ).astype(np.float32))
                Fa, Sa = stA[(combo, qi)]
                Fa = ALPHA_F * codes[qi] + (1 - ALPHA_F) * Fa
                Sa = ALPHA_S * codes[qi] + (1 - ALPHA_S) * Sa
                stA[(combo, qi)] = (Fa, Sa)
                ua = das[qi].step(cn(np.concatenate(
                    [Fa, GG * Sa, GG * ncodes[qi]])).astype(np.float32))
                Aassoc[qi][ua] += lab
                Awins[qi][ua] += 1.0

    # ---- Arm J: joint association LAYER above the columns ----------------
    # (user's design: labels + column outputs learned jointly by the
    # standard rule in a real layer; influence flows down as a GRADED
    # reprojection — advice, not content. Second pass, columns frozen.)
    KJ = 64
    JDIM2 = 4 + 4 * KL
    ALPHA_TOP = 0.1     # the top watches the columns through a SLOWER
                        # clock: its activity trace spreads over phases,
                        # so its memories are direction-general (feeding
                        # instantaneous one-hots made its advice
                        # phase-specific — measured failure)
    dj = SeqDict(KJ, JDIM2, ETA, seed=17)
    stJ = {(c, qi): (np.zeros(K1, np.float32), np.zeros(K1, np.float32))
           for c in TRAIN_COMBOS for qi in range(4)}
    actF = {c: np.zeros(4 * KL, dtype=np.float32) for c in TRAIN_COMBOS}
    for t in range(TICKS):
        for combo in TRAIN_COMBOS:
            lab = np.array(combo, dtype=np.float32)
            act = np.zeros(4 * KL, dtype=np.float32)
            for qi in range(4):
                code = encode_q(quad_line(t, combo[qi]), W1)
                ncode = encode_q(quad_line(t + 1, combo[qi]), W1)
                Fq, Sq = stJ[(combo, qi)]
                Fq = ALPHA_F * code + (1 - ALPHA_F) * Fq
                Sq = ALPHA_S * code + (1 - ALPHA_S) * Sq
                stJ[(combo, qi)] = (Fq, Sq)
                zq = cn(np.concatenate([Fq, GG * Sq, GG * ncode])
                        ).astype(np.float32)
                act[qi * KL + int(np.argmax(das[qi].W @ zq))] = 1.0
            actF[combo] = ALPHA_TOP * act + (1 - ALPHA_TOP) * actF[combo]
            if t >= PERIOD:     # let the trace warm before the top learns
                dj.step(cn(np.concatenate([lab, actF[combo]])
                           ).astype(np.float32))

    # ---- Cold-start generation of held-out combos ------------------------
    results = []
    strips = []
    for combo in TEST_COMBOS:
        lab = np.array(combo, dtype=np.float32)

        F = np.zeros(4 * K1, np.float32)
        S = np.zeros(4 * K1, np.float32)
        genG = []
        for _ in range(FREE_RUN):
            q = cn(np.concatenate([F, GG * S, GG * lab,
                                   np.zeros(4 * K1, np.float32)])).astype(np.float32)
            u = int(np.argmax(dg.W @ q))
            nc = np.maximum(dg.W[u, 2 * (4 * K1) + 4:], 0.0)
            nc = (nc / (np.linalg.norm(nc) + 1e-9)).astype(np.float32)
            img = np.zeros((H, H))
            for qi, (r, c) in enumerate(QUADS):
                img[r:r + Q, c:c + Q] = render_q(nc[qi * K1:(qi + 1) * K1], W1)
            genG.append(img)
            F = ALPHA_F * nc + (1 - ALPHA_F) * F
            S = ALPHA_S * nc + (1 - ALPHA_S) * S

        # Arm Gq: SAME global bank, per-quadrant LOCAL READS — each
        # quadrant retrieves from the unit whose corresponding slice
        # (its code trails + its label bit) best matches, and emits
        # only that unit's slice of the next-code.
        NB = 4 * K1
        genGq = []
        Fq4 = [np.zeros(K1, np.float32) for _ in range(4)]
        Sq4 = [np.zeros(K1, np.float32) for _ in range(4)]
        subW = []
        for qi in range(4):
            sl = np.concatenate([
                dg.W[:, qi * K1:(qi + 1) * K1],
                dg.W[:, NB + qi * K1:NB + (qi + 1) * K1],
                dg.W[:, 2 * NB + qi:2 * NB + qi + 1],
            ], axis=1)
            subW.append(sl / (np.linalg.norm(sl, axis=1, keepdims=True) + 1e-9))
        for _ in range(FREE_RUN):
            img = np.zeros((H, H))
            newF, newS = [], []
            for qi, (r, c) in enumerate(QUADS):
                qv = np.concatenate([Fq4[qi], GG * Sq4[qi],
                                     [GG * combo[qi]]]).astype(np.float32)
                qv = qv - qv.mean()
                qv /= (np.linalg.norm(qv) + 1e-9)
                u = int(np.argmax(subW[qi] @ qv))
                nc = np.maximum(
                    dg.W[u, 2 * NB + 4 + qi * K1:2 * NB + 4 + (qi + 1) * K1], 0.0)
                nc = (nc / (np.linalg.norm(nc) + 1e-9)).astype(np.float32)
                img[r:r + Q, c:c + Q] = render_q(nc, W1)
                newF.append(ALPHA_F * nc + (1 - ALPHA_F) * Fq4[qi])
                newS.append(ALPHA_S * nc + (1 - ALPHA_S) * Sq4[qi])
            Fq4, Sq4 = newF, newS
            genGq.append(img)

        # Arm H: hypercolumn free-run (full label at every column)
        genH = []
        FqH = [np.zeros(K1, np.float32) for _ in range(4)]
        SqH = [np.zeros(K1, np.float32) for _ in range(4)]
        for _ in range(FREE_RUN):
            img = np.zeros((H, H))
            for qi, (r, c) in enumerate(QUADS):
                q = cn(np.concatenate([FqH[qi], GG * SqH[qi], GG * lab,
                                       np.zeros(K1, np.float32)])
                       ).astype(np.float32)
                u = int(np.argmax(dhs[qi].W @ q))
                nc = np.maximum(dhs[qi].W[u, 2 * K1 + 4:], 0.0)
                nc = (nc / (np.linalg.norm(nc) + 1e-9)).astype(np.float32)
                img[r:r + Q, c:c + Q] = render_q(nc, W1)
                FqH[qi] = ALPHA_F * nc + (1 - ALPHA_F) * FqH[qi]
                SqH[qi] = ALPHA_S * nc + (1 - ALPHA_S) * SqH[qi]
            genH.append(img)

        # Arm A: trail-only match + Hebbian label bias
        genA = []
        FqA = [np.zeros(K1, np.float32) for _ in range(4)]
        SqA = [np.zeros(K1, np.float32) for _ in range(4)]
        uprev = [-1, -1, -1, -1]
        Anorm = [Aassoc[qi] / np.maximum(Awins[qi], 1.0)[:, None]
                 for qi in range(4)]
        for _ in range(FREE_RUN):
            img = np.zeros((H, H))
            for qi, (r, c) in enumerate(QUADS):
                q = cn(np.concatenate([FqA[qi], GG * SqA[qi],
                                       np.zeros(K1, np.float32)])
                       ).astype(np.float32)
                # Gate-then-match: the label bias selects the candidate
                # pool (what it knows: direction); the trail argmax picks
                # the phase within it (what the bias cannot know).
                b = Anorm[qi] @ (2 * lab - 1)
                sc = das[qi].W @ q
                sc[b < np.median(b)] = -1e9
                if uprev[qi] >= 0:
                    sc[uprev[qi]] = -1e9   # refractory: phase always advances
                u = int(np.argmax(sc))
                uprev[qi] = u
                nc = np.maximum(das[qi].W[u, 2 * K1:], 0.0)
                nc = (nc / (np.linalg.norm(nc) + 1e-9)).astype(np.float32)
                img[r:r + Q, c:c + Q] = render_q(nc, W1)
                FqA[qi] = ALPHA_F * nc + (1 - ALPHA_F) * FqA[qi]
                SqA[qi] = ALPHA_S * nc + (1 - ALPHA_S) * SqA[qi]
            genA.append(img)

        # Arm J free-run: graded top read by label -> per-column bias gate
        genJ = []
        FqJ = [np.zeros(K1, np.float32) for _ in range(4)]
        SqJ = [np.zeros(K1, np.float32) for _ in range(4)]
        uprevJ = [-1, -1, -1, -1]
        qtop = cn(np.concatenate([lab, np.zeros(4 * KL, np.float32)])
                  ).astype(np.float32)
        s_top = dj.W @ qtop
        top5 = np.argsort(s_top)[::-1][:5]     # (debug display only)
        w_all = np.maximum(s_top, 0.0) ** 4    # sharp population read:
        w_all = w_all / (w_all.sum() + 1e-9)   # every memory votes,
                                               # label-similar ones loudest
        print(f"  [J debug {"".join(map(str, combo))}] top5 scores="
              + " ".join(f"{s_top[m]:+.3f}" for m in top5)
              + " | stored label parts: "
              + " ; ".join("[" + " ".join(f"{v:+.2f}" for v in dj.W[m, :4])
                           + "]" for m in top5), flush=True)
        for qi in range(4):
            bias_dbg = (w_all[:, None]
                        * dj.W[:, 4 + qi * KL:4 + (qi + 1) * KL]).sum(axis=0)
            cw_family = Anorm[qi][:, qi] > 0.5
            passed = bias_dbg >= np.median(bias_dbg)
            print(f"  [J debug {"".join(map(str, combo))}] col{qi}: bias range "
                  f"[{bias_dbg.min():+.4f},{bias_dbg.max():+.4f}] "
                  f"gate keeps {int(passed.sum())}/{KL}, "
                  f"of which CW-family {int((passed & cw_family).sum())} "
                  f"(bank CW total {int(cw_family.sum())})", flush=True)
        for _ in range(FREE_RUN):
            img = np.zeros((H, H))
            for qi, (r, c) in enumerate(QUADS):
                bias = (w_all[:, None]
                        * dj.W[:, 4 + qi * KL:4 + (qi + 1) * KL]).sum(axis=0)
                q = cn(np.concatenate([FqJ[qi], GG * SqJ[qi],
                                       np.zeros(K1, np.float32)])
                       ).astype(np.float32)
                sc = das[qi].W @ q
                keep = np.argsort(bias)[::-1][:KL // 3]
                mask = np.ones(KL, dtype=bool)
                mask[keep] = False
                sc[mask] = -1e9
                if uprevJ[qi] >= 0:
                    sc[uprevJ[qi]] = -1e9
                u = int(np.argmax(sc))
                uprevJ[qi] = u
                nc = np.maximum(das[qi].W[u, 2 * K1:], 0.0)
                nc = (nc / (np.linalg.norm(nc) + 1e-9)).astype(np.float32)
                img[r:r + Q, c:c + Q] = render_q(nc, W1)
                FqJ[qi] = ALPHA_F * nc + (1 - ALPHA_F) * FqJ[qi]
                SqJ[qi] = ALPHA_S * nc + (1 - ALPHA_S) * SqJ[qi]
            genJ.append(img)

        genL_q = []
        for qi in range(4):
            Fq = np.zeros(K1, np.float32)
            Sq = np.zeros(K1, np.float32)
            qframes = []
            for _ in range(FREE_RUN):
                q = cn(np.concatenate([Fq, GG * Sq, GG * eye2[combo[qi]],
                                       np.zeros(K1, np.float32)])).astype(np.float32)
                u = int(np.argmax(dls[qi].W @ q))
                nc = np.maximum(dls[qi].W[u, 2 * K1 + 2:], 0.0)
                nc = (nc / (np.linalg.norm(nc) + 1e-9)).astype(np.float32)
                qframes.append(render_q(nc, W1))
                Fq = ALPHA_F * nc + (1 - ALPHA_F) * Fq
                Sq = ALPHA_S * nc + (1 - ALPHA_S) * Sq
            genL_q.append(qframes)
        genL = []
        for i in range(FREE_RUN):
            img = np.zeros((H, H))
            for qi, (r, c) in enumerate(QUADS):
                img[r:r + Q, c:c + Q] = genL_q[qi][i]
            genL.append(img)

        name = "".join(map(str, combo))
        save_gif(genG, OUTPUT_DIR / f"gen_G_{name}.gif", scale=14)
        save_gif(genGq, OUTPUT_DIR / f"gen_Gq_{name}.gif", scale=14)
        save_gif(genH, OUTPUT_DIR / f"gen_H_{name}.gif", scale=14)
        save_gif(genA, OUTPUT_DIR / f"gen_A_{name}.gif", scale=14)
        save_gif(genJ, OUTPUT_DIR / f"gen_J_{name}.gif", scale=14)
        save_gif(genL, OUTPUT_DIR / f"gen_L_{name}.gif", scale=14)
        strips.append((name, genG, genL))

        for arm, gen in [("G", genG), ("Gq", genGq), ("H", genH),
                         ("A", genA), ("J", genJ), ("L", genL)]:
            bits = []
            for qi, (r, c) in enumerate(QUADS):
                d = direction_of([g[r:r + Q, c:c + Q] for g in gen])
                bits.append(1 if d == sign_cw else (0 if d == -sign_cw else -1))
            ok = sum(int(b == cb) for b, cb in zip(bits, combo))
            results.append((name, arm, bits, ok))
            print(f"TEST {name} arm {arm}: decoded bits="
                  f"{''.join(str(b) if b >= 0 else 'x' for b in bits)} "
                  f"correct {ok}/4", flush=True)

    # Relevance-discovery diagnostic: per column, variance across its
    # units of each stored label dim — own bit should dominate.
    for qi in range(4):
        labW = dhs[qi].W[:, 2 * K1:2 * K1 + 4]
        var = labW.var(axis=0)
        print(f"H column {qi}: label-dim variance across units = "
              + " ".join(f"{v:.4f}" for v in var)
              + f"  (own bit = dim {qi})", flush=True)
    for qi in range(4):
        An = Aassoc[qi] / np.maximum(Awins[qi], 1.0)[:, None]
        act = Awins[qi] > 0
        var = An[act].var(axis=0)
        print(f"A column {qi}: win-average label variance across units = "
              + " ".join(f"{v:.4f}" for v in var)
              + f"  (own bit = dim {qi})", flush=True)

    fig, axes = plt.subplots(4, 12, figsize=(12.5, 4.6))
    for s, (name, genG, genL) in enumerate(strips):
        for i in range(12):
            axes[2 * s, i].imshow(genG[i * 3], cmap="gray", vmin=0, vmax=1)
            axes[2 * s + 1, i].imshow(genL[i * 3], cmap="gray", vmin=0, vmax=1)
            axes[2 * s, i].axis("off")
            axes[2 * s + 1, i].axis("off")
        for r, lbl in [(2 * s, f"G {name}"), (2 * s + 1, f"L {name}")]:
            axes[r, 0].axis("on")
            axes[r, 0].set_xticks([])
            axes[r, 0].set_yticks([])
            for sp in axes[r, 0].spines.values():
                sp.set_visible(False)
            axes[r, 0].set_ylabel(lbl, fontsize=7, rotation=0, ha="right",
                                  va="center")
    fig.suptitle("Held-out combos: global arm (G) vs local arm (L), "
                 "every 3rd frame")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "filmstrip.png", dpi=110)
    plt.close(fig)

    (OUTPUT_DIR / "report.md").write_text("\n".join(
        ["# Compositional generalization: 4 lines x 4 direction bits",
         "",
         f"Trained on <=2-bit combos ({len(TRAIN_COMBOS)} of 16); "
         f"held out: {', '.join(''.join(map(str, c)) for c in TEST_COMBOS)}.",
         "",
         "| test | arm | decoded | correct |",
         "|---|---|---|---|"]
        + [f"| {n} | {a} | {''.join(str(b) if b >= 0 else 'x' for b in bits)} "
           f"| {ok}/4 |" for n, a, bits, ok in results]
        + ["", "Files: input_1100.gif, gen_{G,L}_*.gif, filmstrip.png"]) + "\n")
    print("Report written.")


if __name__ == "__main__":
    main()
