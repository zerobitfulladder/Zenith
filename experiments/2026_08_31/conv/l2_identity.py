"""L2 again, on the identity-only message.

At each of the 24x24 positions the L1 unit says WHICH of its 32 hypercolumns
won and how strongly (the norm of that winner's coefficients). That map is
pooled onto a 6x6 grid -- max within each 4x4 block, per hypercolumn channel --
so L2 sees 36 cells x 32 channels = 1152 numbers, against raw pixels' 784.

Now every coordinate means one fixed thing ("hypercolumn 7 fired in cell 12"),
which is what the dense-only message could not promise. Same L2 unit, same
teacher-free rule, same control.
"""

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fashion"))
import conv1 as c                                                  # noqa: E402
from l2 import train, report, H2, K2, SEED, N_TR, N_TE             # noqa: E402

L1KEY, DS, GRIDN = "H32_K8", "mnist", 6


def identity_map(W1, X, gridn=GRIDN):
    """(n, gridn*gridn*H) -- winner per position, magnitude = its code norm."""
    H, K, _ = W1.shape
    F = np.zeros((len(X), gridn * gridn * H), np.float32)
    for a in range(0, len(X), c.CHUNK_IMG):
        Q, keep = c.prep(c.grid(X[a:a + c.CHUNK_IMG]))
        npos = Q.shape[1]; side = int(np.sqrt(npos))
        kf = keep.reshape(-1)
        e, S = c.errors(W1, Q.reshape(-1, c.PS * c.PS)[kf])
        win = e.argmin(1)
        mag = np.linalg.norm(S[np.arange(len(win)), win], axis=1)
        idx = np.nonzero(kf)[0]
        img, pos = a + idx // npos, idx % npos
        cell = (pos // side * gridn // side) * gridn + (pos % side * gridn // side)
        np.maximum.at(F, (img, cell * H + win), mag)
    return F.astype(np.float64)


def main():
    t0 = time.time()
    W1 = np.load(OUT / f"conv1_{DS}.npz")[L1KEY].astype(np.float64)
    Xtr, ytr, Xte, yte = c.load(DS)
    Xtr, ytr, Xte, yte = Xtr[:N_TR], ytr[:N_TR], Xte[:N_TE], yte[:N_TE]
    Ftr, Fte = identity_map(W1, Xtr), identity_map(W1, Xte)
    print(f"L2 input {Ftr.shape[1]} numbers "
          f"({float((Ftr != 0).mean())*100:.0f}% nonzero)   "
          f"[raw pixels would be 784]", flush=True)
    W, wins = train(Ftr, ytr, np.random.default_rng(SEED + 1), "identity 6x6")
    r = report(W, wins, Fte, yte, "identity 6x6")
    prev = json.loads((OUT / "l2.json").read_text())["results"]
    print(f"\n  for comparison:")
    for k, v in prev.items():
        print(f"    {k:<16} accuracy {v['acc']:.4f}   "
              f"rebuild(input) {v['rebuild_input']:.4f}")
    (OUT / "l2_identity.json").write_text(json.dumps(
        {"L1": L1KEY, "grid": GRIDN, "H2": H2, "K2": K2, "dims": Ftr.shape[1],
         "result": r, "previous": prev,
         "seconds": round(time.time() - t0, 1)}, indent=2))
    np.savez_compressed(OUT / "l2_identity.npz", W=W.astype(np.float32), wins=wins)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
