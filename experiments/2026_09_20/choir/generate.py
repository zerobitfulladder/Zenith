"""Can an image be rebuilt from the single top-level chord?  No labels anywhere.

    up    image -> L1 -> pool 3x3 -> L2 -> pool global -> L3 -> h        (B angles)
    down  h -> D3 -> 25 chords -> D2 -> 121 cell chords -> D1 -> 121 patches -> image

h is B=64 angles standing for a 784-pixel image.  The reference is PCA at the same budget.
Every level is trained together on patch reconstruction; the assembled picture is made by
averaging the overlapping 7x7 patches.
"""
import argparse, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import layer1 as L1M
from twotrack import cluster, Voices, atan2d, P, B, L, NG, WIN, WSTEP

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
dev = "cuda" if torch.cuda.is_available() else "cpu"
POS = list(range(0, 28 - P + 1, 2))


class AE(nn.Module):
    def __init__(s, Wc, Ws, rho_px, e2e=False):
        super().__init__()
        if e2e: s.Wc, s.Ws = nn.Parameter(Wc.clone()), nn.Parameter(Ws.clone())
        else:   s.register_buffer("Wc", Wc); s.register_buffer("Ws", Ws)
        s.register_buffer("rbase", torch.stack([rho_px[3], rho_px[2]]))
        s.nw = len(range(0, NG - WIN + 1, WSTEP))**2                  # 25 windows
        s.rho2 = nn.Parameter(torch.randn(s.nw, B) * .3)
        s.L2, s.L3 = Voices(B*L), Voices(B*L)
        s.D3 = nn.Linear(2*B, s.nw*2*B)                               # h  -> 25 chords
        s.D2 = nn.Linear(2*B, 9*2*B)                                  # win-> 9 cell chords
        s.D1 = nn.Linear(2*B, P*P)                                    # cell -> patch

    def r1(s, dy, dx): return dy*s.rbase[0] + dx*s.rbase[1]
    @staticmethod
    def ang(z):
        a, c = z.chunk(2, -1); return atan2d(c, a)

    def up(s, patches):
        n = len(patches)
        th = atan2d(patches @ s.Ws.T, patches @ s.Wc.T).view(n, NG, NG, B)
        out = []
        for i in range(0, NG - WIN + 1, WSTEP):
            for j in range(0, NG - WIN + 1, WSTEP):
                out.append(sum(cluster(th[:, i+1+dy, j+1+dx] + s.r1(dy, dx))
                               for dy in (-1, 0, 1) for dx in (-1, 0, 1)))
        w = s.L2(torch.stack(out, 1).flatten(2))                      # (n, 25, B)
        z = sum(cluster(w[:, k] + s.rho2[k]) for k in range(s.nw))
        return s.L3(z.flatten(1)), th.view(n, NG*NG, B)               # h, and L1 chords

    def down(s, h):
        n = len(h)
        cs = lambda t: torch.cat([torch.cos(t), torch.sin(t)], -1)
        w = s.ang(s.D3(cs(h)).view(n, s.nw, 2*B))                     # 25 window chords
        c = s.ang(s.D2(cs(w)).view(n, s.nw, 9, 2*B))                  # 9 cells each
        # scatter the 25x9 predictions back onto the 11x11 cell grid, averaging overlaps
        acc = h.new_zeros(n, NG, NG, 2*B); cnt = h.new_zeros(1, NG, NG, 1)
        k = 0
        for i in range(0, NG - WIN + 1, WSTEP):
            for j in range(0, NG - WIN + 1, WSTEP):
                for q, (dy, dx) in enumerate([(u, v) for u in (-1,0,1) for v in (-1,0,1)]):
                    acc[:, i+1+dy, j+1+dx] += cs(c[:, k, q])
                    cnt[:, i+1+dy, j+1+dx] += 1
                k += 1
        cell = s.ang(acc / cnt.clamp(min=1))                          # (n, NG, NG, B)
        return s.D1(cs(cell)).view(n, NG*NG, P*P)                     # 121 patches


def load(split, n):
    X, y = L1M.mnist_labelled(split, n)
    pa = np.stack([[im[i:i+P, j:j+P].ravel() for i in POS for j in POS] for im in X])
    pa = pa - pa.mean(-1, keepdims=True)
    return torch.tensor(pa, dtype=torch.float32), torch.tensor(np.stack(X))


def assemble(patches):
    """121 mean-centred 7x7 patches -> a 28x28 picture, averaging the overlaps."""
    n = len(patches); img = np.zeros((n, 28, 28)); cnt = np.zeros((28, 28))
    k = 0
    for i in POS:
        for j in POS:
            img[:, i:i+P, j:j+P] += patches[:, k].reshape(n, P, P); cnt[i:i+P, j:j+P] += 1; k += 1
    return img / cnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000); ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--e2e", type=int, default=1)
    a = ap.parse_args()

    W = np.load(ROOT / "results" / "W_coherent.npy"); rho = np.load(ROOT / "results" / "rho_coherent.npy")
    Xtr, Itr = load("train", a.train); Xte, Ite = load("t10k", a.test)
    m = AE(torch.tensor(W[0], dtype=torch.float32), torch.tensor(W[1], dtype=torch.float32),
           torch.tensor(rho, dtype=torch.float32), e2e=bool(a.e2e)).to(dev)
    print(f"{a.train} train / {a.test} test digits.  top code = {B} angles for a 784-px image, "
          f"L1 {'trained' if a.e2e else 'frozen'}, dev={dev}\n")

    opt = torch.optim.Adam(m.parameters(), a.lr); t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr) - 32, 32):
            x = Xtr[perm[i:i+32]].to(dev)
            h, _ = m.up(x)
            loss = F.mse_loss(m.down(h), x)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        if ep % 5 == 4:
            print(f"    epoch {ep+1:>3}  train mse {loss.item():.5f}   ({time.time()-t0:.0f}s)")

    with torch.no_grad():
        xt = Xte.to(dev); h, _ = m.up(xt); rec = m.down(h)
        r2 = 1 - ((rec - xt)**2).sum().item() / (xt**2).sum().item()
    print(f"\n  top-code reconstruction R2 (patch level): {r2:.3f}")

    F_ = Ite.reshape(len(Ite), -1).numpy(); Ftr = Itr.reshape(len(Itr), -1).numpy()
    mu = Ftr.mean(0); C = np.cov((Ftr - mu).T); ev, Q = np.linalg.eigh(C); Q = Q[:, ::-1][:, :B]
    pca = (F_ - mu) @ Q @ Q.T + mu
    print(f"  PCA at the same budget ({B} numbers), whole image:  "
          f"{1 - ((pca - F_)**2).sum()/((F_ - F_.mean())**2).sum():.3f}")

    rec_img = assemble(rec.cpu().numpy()); org_img = assemble(xt.cpu().numpy())
    fig, ax = plt.subplots(3, 12, figsize=(12*.8, 3*.95))
    for j in range(12):
        for r, (im, lbl) in enumerate(((Ite[j].numpy(), "digit"),
                                       (org_img[j], "patches"), (rec_img[j], "from h"))):
            ax[r, j].imshow(im, cmap="gray"); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            if j == 0: ax[r, j].set_ylabel(lbl, fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(f"Rebuilt from a single {B}-angle top code, no labels   (R2 = {r2:.3f})", fontsize=10)
    fig.tight_layout(rect=[.03, 0, 1, .92]); fig.savefig(OUT / "generate.png", dpi=150)
    print(f"  figure -> {OUT}/generate.png")
