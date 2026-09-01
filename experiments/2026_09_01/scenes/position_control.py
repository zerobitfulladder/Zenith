"""Is the recognition loss the digit's POSITION?

Finer pooling made single-digit recognition worse, not better (0.8465 -> 0.7715),
which refutes the resolution hypothesis and points the other way: an identity map
ties what to WHERE, so the same digit at a new place is a new pattern, and finer
blocks make that worse. This isolates it -- the identical encoder and readout,
one digit, position fixed vs free.
"""
import json, time
from pathlib import Path
import numpy as np
import scenes as S, readouts as R

OUT = Path(__file__).resolve().parent / "results"
N_TR, N_TE, EP = 6000, 2000, 25


def fixed(n, X, bucket, seed):
    rng = np.random.default_rng(seed)
    cls = rng.integers(0, 10, n)
    imgs = np.zeros((n, S.H, S.W), np.float32)
    y0, x0 = S.H // 2 - 14, S.W // 2 - 14           # dead centre, every scene
    for i in range(n):
        t = imgs[i, y0:y0 + 28, x0:x0 + 28]
        np.maximum(t, X[rng.choice(bucket[cls[i]])], out=t)
    return imgs, cls.astype(np.int64)


def main():
    t0 = time.time(); X, ptr, pte = S.load_digits(); W1 = S.load_w1(); res = {}
    for name, gen in (("fixed", fixed), ("free", S.make_single)):
        a, ya = gen(N_TR, X, ptr, 21); b, yb = gen(N_TE, X, pte, 22)
        A, B = S.l1_pooled(W1, a), S.l1_pooled(W1, b)
        r = {}
        for lab, hid in (("logistic", 0), ("mlp", 256)):
            n = R.fit_net(A, np.arange(len(A)), None, ya, 10, "softmax", hidden=hid, epochs=EP)
            r[lab] = float((R.predict_net(n, B, np.arange(len(B)), None).argmax(1) == yb).mean())
        res[name] = r
        print(f"  position {name:<6} logistic {r['logistic']:.4f}  mlp {r['mlp']:.4f}", flush=True)
    res["reference"] = {"yesterday, 28x28 centred": 0.9492}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "position_control.json").write_text(json.dumps(res, indent=2))
    print(f"done in {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
