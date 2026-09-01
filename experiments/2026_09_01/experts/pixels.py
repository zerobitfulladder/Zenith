"""The control we should have run first: no layer at all, just count pixels.

With 400 templates over 5x5 patches, layer 1 is describing very local pixel
structure. So how much of the 0.9707 is the templates, and how much would you
get by counting raw pixels with the same machinery?

Same scoring rule, one level lower:

    T[position, intensity bin, class] = log P(bin | position, class)
                                          / P(bin | position)
    score(class) = sum over all 784 positions

That is naive Bayes on quantised pixels. If it lands near 0.97, the template
layer is decoration. If it lands well below, the templates are doing real work
and we can say how much.
"""

import json, sys, time
from pathlib import Path
import numpy as np
import experts as E

OUT = Path(__file__).resolve().parent / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "mnist"
NL, ALPHA = 10, 1.0
BINS = [2, 4, 8, 16]


def run(Xtr, ytr, Xte, yte, B):
    n, d = len(Xtr), Xtr.shape[1]
    qtr = np.clip((Xtr * B).astype(np.int64), 0, B - 1)
    qte = np.clip((Xte * B).astype(np.int64), 0, B - 1)
    N = np.zeros((d, B, NL))
    pos = np.tile(np.arange(d), n)
    np.add.at(N, (pos, qtr.ravel(), np.repeat(ytr, d)), 1.0)
    pc = (N + ALPHA) / (N.sum(1, keepdims=True) + ALPHA * B)          # P(bin | pos, class)
    pm = ((N.sum(2, keepdims=True) + ALPHA * NL)
          / (N.sum((1, 2), keepdims=True) + ALPHA * B * NL))          # P(bin | pos)
    T = (np.log(pc) - np.log(pm)).astype(np.float32)
    P = np.arange(d)
    sc = np.zeros((len(qte), NL))
    for a in range(0, len(qte), 500):
        b = qte[a:a + 500]
        sc[a:a + len(b)] = T[P[None, :], b].sum(1)
    return float((sc.argmax(1) == yte).mean()), int(T.size)


def main():
    t0 = time.time()
    Xtr, ytr, Xte, yte = E.load(DS)
    Xtr = Xtr.reshape(len(Xtr), -1); Xte = Xte.reshape(len(Xte), -1)
    print(f"{DS}: counting raw pixels, no layer, no templates\n", flush=True)
    res = {}
    for B in BINS:
        acc, size = run(Xtr, ytr, Xte, yte, B)
        res[f"bins={B}"] = {"acc": acc, "table": size}
        print(f"  {B:>2} intensity bins   accuracy {acc:.4f}   table {size:,}", flush=True)
    ref = ({"L1 400 templates + table": 0.9707, "logistic on pixels": 0.9074}
           if DS == "mnist" else
           {"L1 180 templates + table": 0.8293, "logistic on pixels": 0.8512})
    res["reference"] = ref
    print("\n  for comparison: " + "   ".join(f"{k} {v:.4f}" for k, v in ref.items()))
    (OUT / f"pixels_{DS}.json").write_text(json.dumps(res, indent=2))
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
