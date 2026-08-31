"""Three ways for a unit to speak, and whether the third one loses the identity.

Per patch position:
    1  identity only     H floats, one nonzero (the winner, with its magnitude)
    2  identity + dense  H*K floats, the winner's K coefficients in its slot
    3  dense only        K floats, the winner's coefficients with NO indication
                         of who produced them

Format 3 is only safe if the identity is implicitly recoverable from the code,
because different hypercolumns hold different bases. So the first test is
literally that: predict the winner from its coefficients alone. Chance is 1/H.
"""

import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv1 as c

OUT = Path(__file__).resolve().parent / "results"
KEY = sys.argv[1] if len(sys.argv) > 1 else "H32_K8"
DS, GRID = "mnist", 2


def collect(W, X, gridn=GRID):
    H, K, _ = W.shape
    n = len(X)
    F = {"1 identity only": np.zeros((n, gridn * gridn * H), np.float32),
         "2 identity+dense": np.zeros((n, gridn * gridn * H * K), np.float32),
         "3 dense only": np.zeros((n, gridn * gridn * K), np.float32)}
    codes, owners = [], []
    for a in range(0, n, c.CHUNK_IMG):
        Q, keep = c.prep(c.grid(X[a:a + c.CHUNK_IMG]))
        npos = Q.shape[1]; side = int(np.sqrt(npos))
        kf = keep.reshape(-1)
        e, S = c.errors(W, Q.reshape(-1, c.PS * c.PS)[kf])
        win = e.argmin(1)
        fit = 1.0 - e[np.arange(len(win)), win]
        code = np.abs(S[np.arange(len(win)), win])
        idx = np.nonzero(kf)[0]
        img = a + idx // npos
        pos = idx % npos
        cell = (pos // side * gridn // side) * gridn + (pos % side * gridn // side)
        np.maximum.at(F["1 identity only"], (img, cell * H + win), fit)
        cols = win[:, None] * K + np.arange(K)[None, :]
        np.maximum.at(F["2 identity+dense"],
                      (img[:, None].repeat(K, 1), cell[:, None] * H * K + cols), code)
        np.maximum.at(F["3 dense only"],
                      (img[:, None].repeat(K, 1),
                       cell[:, None] * K + np.arange(K)[None, :]), code)
        if len(codes) < 40:
            codes.append(code); owners.append(win)
    return F, np.concatenate(codes), np.concatenate(owners)


def main():
    from sklearn.linear_model import LogisticRegression
    W = np.load(OUT / f"conv1_{DS}.npz")[KEY].astype(np.float64)
    H, K, _ = W.shape
    Xtr, ytr, Xte, yte = c.load(DS)
    Ftr, code_tr, own_tr = collect(W, Xtr[:6000])
    Fte, code_te, own_te = collect(W, Xte)
    print(f"{KEY}: {H} hypercolumns x {K} minicolumns\n")

    # can the identity be read back out of the code alone?
    m = LogisticRegression(max_iter=600).fit(code_tr[:150000], own_tr[:150000])
    rec = float((m.predict(code_te[:60000]) == own_te[:60000]).mean())
    print(f"  who spoke, recovered from the code alone: {rec:.4f}   "
          f"(chance {1/H:.4f})")
    maj = float(np.bincount(own_te[:60000], minlength=H).max() / 60000)
    print(f"  always-guess-the-commonest baseline:       {maj:.4f}\n")

    res = {"identity_recoverable": rec, "chance": 1 / H, "majority": maj}
    for k in sorted(Ftr):
        mm = LogisticRegression(max_iter=500).fit(Ftr[k], ytr[:6000])
        acc = float((mm.predict(Fte[k]) == yte).mean())
        res[k] = {"acc": acc, "dims": int(Ftr[k].shape[1])}
        print(f"  {k:<20} dims {Ftr[k].shape[1]:>5}   probe {acc:.4f}")
    (OUT / f"formats_{KEY}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
