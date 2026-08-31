"""Is L2 the bottleneck? Sweep how many templates it has, hold everything else.

If stacking two hard quantisers is what costs the three-rung stack its accuracy,
then giving L2 more templates -- so it discards less -- should recover it toward
the two-rung 0.9492.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint, stack, three

OUT = Path(__file__).resolve().parent / "results"
K2S = [128, 256, 512]


def main():
    rng = np.random.default_rng(1)
    W1 = np.load(OUT / "km_mnist.npz")["64"].astype(np.float64)
    k1 = len(W1)
    Xtr, ytr, Xte, yte = joint.load("mnist")
    itr, mtr = stack.l1_map(W1, Xtr)
    ite, mte = stack.l1_map(W1, Xte)
    sub = rng.choice(len(Xtr), 6000, replace=False)
    Qw, nv = three.unit(three.windows(three.pool_map(itr[sub], mtr[sub], k1)))
    Qw = Qw[nv > 0]
    Qw = Qw[rng.choice(len(Qw), 300000, False)].astype(np.float64)
    res = {}
    for K2 in K2S:
        three.K2 = K2
        W2t, n2 = three.kmeans(Qw, K2, np.random.default_rng(2), epochs=8)

        def enc(idx, mag, chunk=2000):
            I, M = [], []
            for a in range(0, len(idx), chunk):
                i, m = three.l2_code(W2t, three.pool_map(idx[a:a+chunk], mag[a:a+chunk], k1))
                I.append(i); M.append(m)
            return np.concatenate(I), np.concatenate(M)

        i2tr, m2tr = enc(itr, mtr); i2te, m2te = enc(ite, mte)
        A = np.concatenate([three.l3_input(i2tr[s:s+2000], m2tr[s:s+2000],
                                           ytr[s:s+2000]).astype(np.float32)
                            for s in range(0, len(i2tr), 2000)])
        W3, n3 = three.kmeans(A.astype(np.float64), three.K3,
                              np.random.default_rng(3), epochs=8)
        P = three.P2 * three.P2 * K2
        lab = W3[:, P:].argmax(1)
        acc = []
        for s in range(0, len(i2te), 2000):
            B = three.l3_input(i2te[s:s+2000], m2te[s:s+2000])
            acc.append(lab[(B @ W3.T).argmax(1)] == yte[s:s+2000])
        a = float(np.concatenate(acc).mean())
        p = int(K2 * three.W2 * three.W2 * k1 + three.K3 * (P + 10))
        res[str(K2)] = {"acc": a, "params": p, "l3_input": int(P + 10),
                        "dead_l2": int((n2 == 0).sum()), "dead_l3": int((n3 == 0).sum())}
        print(f"  K2={K2:<5} L3 input {P+10:>6}  params {p:>9,}  "
              f"accuracy {a:.4f}  dead L2 {int((n2==0).sum())}", flush=True)
        del A
    res["reference"] = {"two rungs": 0.9492, "one rung on pixels": 0.9210}
    (OUT / "three_sweep.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
