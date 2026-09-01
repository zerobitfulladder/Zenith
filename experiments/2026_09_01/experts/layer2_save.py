"""Train layer 2 once at K2=200 and SAVE it: templates, counts, and the table
in the same log-ratio form layer 1 uses.

    T2[t, class] = log P(t | class) / P(t)

With one active L2 unit per image this is a single lookup rather than a sum, and
on balanced classes its argmax agrees with the raw count argmax -- but it is the
form the downward path needs, because it keeps the SOFT distribution. A template
that is 90% 4s and 10% 9s should bias layer 1 differently from one that is
certain.
"""
import json
from pathlib import Path
import numpy as np
import experts as E, single as SG, pressure as PR, layer2 as L2

OUT = Path(__file__).resolve().parent / "results"
K2, NL, ALPHA = 200, 10, 1.0

Xtr, ytr, Xte, yte = E.load()
W1 = np.load(OUT / "coadapt.npz")["W"].astype(np.float64)
Qtr = L2.cn(PR.pooled(SG.winners(W1, Xtr), len(W1))).astype(np.float32)
Qte = L2.cn(PR.pooled(SG.winners(W1, Xte), len(W1))).astype(np.float32)
W2, n = L2.train(Qtr, K2, np.random.default_rng(3))

wtr, wte = (Qtr @ W2.T).argmax(1), (Qte @ W2.T).argmax(1)
C = np.zeros((K2, NL))
np.add.at(C, (wtr, ytr), 1.0)

p_t_given_c = (C + ALPHA) / (C.sum(0, keepdims=True) + ALPHA * K2)      # P(t | class)
p_t = (C.sum(1, keepdims=True) + ALPHA * NL) / (C.sum() + ALPHA * K2 * NL)
T2 = np.log(p_t_given_c) - np.log(p_t)

acc_argmax = float((C.argmax(1)[wte] == yte).mean())
acc_table = float((T2[wte].argmax(1) == yte).mean())
soft = T2[wte] - T2[wte].max(1, keepdims=True)
q = np.exp(soft); q /= q.sum(1, keepdims=True)
print(f"  K2={K2}  raw-count argmax {acc_argmax:.4f}   log-ratio table {acc_table:.4f}")
print(f"  mean confidence of the winning class: {q.max(1).mean():.3f}")
print(f"  templates whose top class is under 90% pure: "
      f"{int(((C.max(1)/np.maximum(C.sum(1),1)) < 0.9).sum())}/{K2}")

np.savez_compressed(OUT / "layer2_k200.npz", W2=W2.astype(np.float32),
                    counts=C, T2=T2.astype(np.float32))
(OUT / "layer2_k200.json").write_text(json.dumps(
    {"K2": K2, "acc_raw_argmax": acc_argmax, "acc_log_ratio": acc_table,
     "mean_confidence": float(q.max(1).mean()),
     "params_templates": int(W2.size), "params_table": int(T2.size)}, indent=2))
print(f"  saved: {W2.size:,} template numbers + {T2.size:,} table numbers")
