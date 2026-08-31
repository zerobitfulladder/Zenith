"""What does the unit send up: identity only, or identity + the dense code?

Per position the unit produces one winner. Two message formats:

    identity only    H floats: the winner's slot holds how well it fit, rest 0
    identity + dense H*K floats: the winner's K coefficients in its own slot

Pooled two ways (global, and on a 2x2 spatial grid so some layout survives) and
read by a linear probe. The probe is a stand-in for "can a next rung use this",
not part of the architecture.
"""

import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv1 as c
from common import EPS

OUT = Path(__file__).resolve().parent / "results"
KEY = sys.argv[1] if len(sys.argv) > 1 else "H8_K6"
DS = "mnist"


def messages(W, X, gridn=2):
    """Returns dict of pooled feature matrices for each format."""
    H, K, _ = W.shape
    n = len(X)
    feats = {"identity, global": np.zeros((n, H), np.float32),
             "identity, 2x2": np.zeros((n, gridn * gridn * H), np.float32),
             "identity+dense, global": np.zeros((n, H * K), np.float32),
             "identity+dense, 2x2": np.zeros((n, gridn * gridn * H * K), np.float32),
             "counts, global": np.zeros((n, H), np.float32)}
    for a in range(0, n, c.CHUNK_IMG):
        Xc = X[a:a + c.CHUNK_IMG]
        Q, keep = c.prep(c.grid(Xc))
        npos = Q.shape[1]
        side = int(np.sqrt(npos))
        kf = keep.reshape(-1)
        e, S = c.errors(W, Q.reshape(-1, c.PS * c.PS)[kf])
        win = e.argmin(1)
        fit = 1.0 - e[np.arange(len(win)), win]              # how well it fit
        code = np.abs(S[np.arange(len(win)), win])           # (m, K)
        idx = np.nonzero(kf)[0]
        img = a + idx // npos
        pos = idx % npos
        cell = (pos // side * gridn // side) * gridn + (pos % side * gridn // side)
        np.maximum.at(feats["identity, global"], (img, win), fit)
        np.add.at(feats["counts, global"], (img, win), 1.0)
        np.maximum.at(feats["identity, 2x2"], (img, cell * H + win), fit)
        cols = win[:, None] * K + np.arange(K)[None, :]
        np.maximum.at(feats["identity+dense, global"],
                      (img[:, None].repeat(K, 1), cols), code)
        np.maximum.at(feats["identity+dense, 2x2"],
                      (img[:, None].repeat(K, 1), cell[:, None] * H * K + cols), code)
    return feats


def main():
    z = np.load(OUT / f"conv1_{DS}.npz")
    W = z[KEY].astype(np.float64)
    H, K, _ = W.shape
    Xtr, ytr, Xte, yte = c.load(DS)
    Ftr = messages(W, Xtr[:6000])
    Fte = messages(W, Xte)
    from sklearn.linear_model import LogisticRegression
    res = {}
    print(f"{KEY}  ({H} hypercolumns x {K} minicolumns)")
    for k in Ftr:
        m = LogisticRegression(max_iter=500).fit(Ftr[k], ytr[:6000])
        acc = float((m.predict(Fte[k]) == yte).mean())
        res[k] = {"acc": acc, "dims": int(Ftr[k].shape[1])}
        print(f"  {k:<26} dims {Ftr[k].shape[1]:>5}   probe {acc:.4f}")
    # raw pixels, same probe, for scale
    m = LogisticRegression(max_iter=500).fit(Xtr[:6000].reshape(6000, -1), ytr[:6000])
    res["raw pixels"] = {"acc": float((m.predict(Xte.reshape(len(Xte), -1)) == yte).mean()),
                         "dims": 784}
    print(f"  {'raw pixels (reference)':<26} dims   784   probe {res['raw pixels']['acc']:.4f}")
    (OUT / f"message_{KEY}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
