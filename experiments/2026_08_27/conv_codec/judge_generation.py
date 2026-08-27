"""Re-judge the conv-codec free runs with a MATCHED judge.

The nearest-class-mean gate scores 0.87 on real MNIST but much worse on
renders (feathered strokes, different contrast) — it undercounts the
generations. Fair instrument: train a probe on renders of REAL frames
through this arm's own L1 round trip, i.e. the same distribution the
emissions come from, then judge the free run with it.

Reports both judges side by side plus cycle coverage.

Run:  CODEC_ARM=w4k36 .venv/bin/python experiments/2026_08_27/conv_codec/judge_generation.py
"""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import run_conv_codec as R   # reads CODEC_ARM at import

ARM = R.ARM
DIR = R.OUTPUT_DIR
w = np.load(DIR / "weights.npz")

l1 = R.L1()
l1.bank.W = w["W1"]
l1.bank.n_boot = R.K1
W2 = w["W2"]
basis = l1.cargo


def successor(T2):
    q = R.cn1(np.concatenate([np.zeros(R.CODE_D, np.float32), R.cn1(T2)]))
    w2 = int(np.argmax(W2 @ q))
    return w2, np.maximum(W2[w2, :R.CODE_D], 0.0).reshape(R.NPOS, R.K1)


# ---- matched judge: renders of real frames, this arm's own round trip ----
l1.reset()
Xr, yr = [], []
for t in range(1500):
    _, cs, _, _, _ = l1.encode(R.frame_for(t), False)
    if t >= 20:
        Xr.append(R.render(cs, basis, "committed").ravel())
        yr.append(t % 10)
Xr, yr = np.array(Xr), np.array(yr)
cut = int(0.8 * len(Xr))
probe = LogisticRegression(max_iter=2000, C=1.0).fit(Xr[:cut], yr[:cut])
probe_val = float(probe.score(Xr[cut:], yr[cut:]))
oldjudge_on_renders = float(np.mean([R.classify(x) == l for x, l in
                                     zip(Xr[cut:], yr[cut:])]))

out = {"arm": ARM, "probe_val_acc_on_real_renders": round(probe_val, 3),
       "old_judge_acc_on_real_renders": round(oldjudge_on_renders, 3)}

for mode in ("graded", "committed"):
    l1.reset()
    T2 = np.zeros(R.CODE_D, np.float32)
    for t in range(R.PRIME):
        _, cs, _, _, _ = l1.encode(R.frame_for(t), False)
        T2 = cs.ravel() + R.G2 * T2
    gen = []
    for _ in range(R.FREE):
        _, prof = successor(T2)
        em = R.render(prof, basis, mode)
        gen.append(em)
        _, cs, _, _, _ = l1.encode(em.astype(np.float32), False)
        T2 = cs.ravel() + R.G2 * T2
    G = np.array([g.ravel() for g in gen])
    new = probe.predict(G)
    old = np.array([R.classify(g) for g in gen])
    intended = np.arange(R.PRIME, R.PRIME + R.FREE) % 10
    for name, cls in (("probe", new), ("meanjudge", old)):
        out[f"{mode}_{name}_advance"] = round(
            float(np.mean((np.diff(cls) % 10) == 1)), 3)
        out[f"{mode}_{name}_timeline"] = round(
            float(np.mean(cls == intended)), 3)
        out[f"{mode}_{name}_distinct_classes"] = int(len(set(cls.tolist())))
    out[f"{mode}_probe_first40"] = "".join(map(str, new[:40]))

print(json.dumps(out), flush=True)
(DIR / "judged.json").write_text(json.dumps(out, indent=2))
