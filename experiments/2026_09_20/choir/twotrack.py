"""Two tracks into one choir: image (3 layers, pooled) and label (3 layers, no position).

    image  -> L1 -> [pool 3x3] -> L2 -> [pool global] -> L3 -> h_img
    label  -> M1 -> M2 -> M3 ------------------------------> h_lab
    top choir:   Z = cluster(h_img) + cluster(h_lab)      (label gets NO transposition)

Read the label back by cleanup: score all ten label chords against the top polychord.
At test time only the image track runs, so Z = cluster(h_img) alone.

  --mask p   drop the label track from Z with probability p during training.
             p=0 lets the loss be satisfied by copying label->label without consulting
             the image; p>0 forces the route through the image.

L1 is the coherence-trained bank from coherent.py, frozen: no gradient crosses a quantiser,
and clusters are used above it so gradients do flow through the upper poolings.
"""
import argparse, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
import layer1 as L1M, coherent as CO

P, B, L = 7, 64, 41
NG, WIN, WSTEP = 11, 3, 2                      # 11x11 cells -> 5x5 windows, full cover
dev = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ pieces
class _Atan2(torch.autograd.Function):
    """atan2 with the 1/r^2 backward damped, exactly as layer1.py does it in numpy.

    torch's own atan2 backward carries 1/(a^2+c^2) unprotected; a faintly driven voice
    sends r -> 0 and the gradient detonates into NaN.  This is the same guard, in the
    place autograd actually uses."""
    EPS = 1e-3
    @staticmethod
    def forward(ctx, c, a):
        ctx.save_for_backward(c, a); return torch.atan2(c, a)
    @staticmethod
    def backward(ctx, g):
        c, a = ctx.saved_tensors
        r2 = a*a + c*c + _Atan2.EPS
        return g * (a / r2), g * (-c / r2)


def atan2d(c, a):
    return _Atan2.apply(c, a)


def cluster(th, w=0.7):
    """angles (..., B) -> soft one-hot over L notes (..., B, L).  Differentiable."""
    slot = torch.arange(L, device=th.device, dtype=th.dtype) * (2*np.pi/L)
    d = th.unsqueeze(-1) - slot
    d = torch.atan2(torch.sin(d), torch.cos(d)) * (L/(2*np.pi))     # circular distance in slots
    return torch.softmax(-d*d/(2*w*w), dim=-1)


class Voices(nn.Module):
    """linear -> atan2.  The only nonlinearity."""
    def __init__(s, nin, b=B):
        super().__init__(); s.f = nn.Linear(nin, 2*b, bias=False)
        nn.init.normal_(s.f.weight, std=nin**-0.5)
    def forward(s, x):
        z = s.f(x); a, c = z.chunk(2, -1)
        return atan2d(c, a)


class TwoTrack(nn.Module):
    def __init__(s, Wc, Ws, rho_px, e2e=False, depth=2):
        super().__init__()
        if e2e:   s.Wc, s.Ws = nn.Parameter(Wc.clone()), nn.Parameter(Ws.clone())
        else:     s.register_buffer("Wc", Wc); s.register_buffer("Ws", Ws)
        s.depth = depth
        # pool-1 intervals are COMPOSED from the learned pixel intervals (cell step = 2 px):
        #   rho(dy,dx) = dy*rho(2,0) + dx*rho(0,2).   Poses add.
        r = torch.stack([rho_px[k] for k in range(4)])          # (0,1) (1,0) (0,2) (2,0)
        if e2e:   s.rbase = nn.Parameter(torch.stack([r[3], r[2]]))
        else:     s.register_buffer("rbase", torch.stack([r[3], r[2]]))
        # every later pooling gets free intervals, sized to its own window
        s.rho = nn.ParameterList([nn.Parameter(torch.randn(64, B) * 0.3)
                                  for _ in range(max(depth - 1, 1))])
        s.enc = nn.ModuleList([Voices(B*L) for _ in range(depth)])
        s.M1, s.M2, s.M3 = Voices(10), Voices(B), Voices(B)
        s.match = Voices(B*L); s.g = nn.Parameter(torch.randn(B) * 0.3)

    def rho1(s, dy, dx):
        return dy * s.rbase[0] + dx * s.rbase[1]

    def _pool(s, th, g, win, step, rho):
        """3x3 (or global) windows over a g x g map of chords; transpose, render, sum."""
        if win >= g:                                             # global
            c = g // 2
            return torch.stack([sum(cluster(th[:, i, j] + rho[i*g + j])
                                    for i in range(g) for j in range(g))], 1), 1
        out = []
        for i in range(0, g - win + 1, step):
            for j in range(0, g - win + 1, step):
                out.append(sum(cluster(th[:, i+1+dy, j+1+dx] + rho[k])
                               for k, (dy, dx) in enumerate(
                                   [(u, v) for u in (-1,0,1) for v in (-1,0,1)])))
        return torch.stack(out, 1), int(len(out) ** .5)

    def image(s, patches):                                       # (n, NG*NG, 49)
        n = patches.shape[0]
        th = atan2d(patches @ s.Ws.T, patches @ s.Wc.T).view(n, NG, NG, B)   # L1
        if s.depth == 1:                                         # one global stage
            c = NG // 2
            z = sum(cluster(th[:, i, j] + s.rho1(i - c, j - c))
                    for i in range(NG) for j in range(NG))
            return s.enc[0](z.flatten(1))
        # stage 1: 3x3 cell windows, composed intervals
        out = []
        for i in range(0, NG - WIN + 1, WSTEP):
            for j in range(0, NG - WIN + 1, WSTEP):
                out.append(sum(cluster(th[:, i+1+dy, j+1+dx] + s.rho1(dy, dx))
                               for dy in (-1, 0, 1) for dx in (-1, 0, 1)))
        z = torch.stack(out, 1); g = int(len(out) ** .5)
        th = s.enc[0](z.flatten(2))
        for d in range(1, s.depth):                              # pool then encode
            th = th.view(n, g, g, B)
            win = WIN if g > WIN else g
            z, g = s._pool(th, g, win, WSTEP, s.rho[d-1])
            th = s.enc[d](z.flatten(2))
        return th.reshape(n, -1)[:, :B] if th.dim() == 3 else th

    def label(s, y):
        return s.M3(s.M2(s.M1(F.one_hot(y, 10).float())))


def score(Z, lab_th):
    """bi-encoder: inner product of the image polychord with each label chord."""
    return torch.einsum("nbl,kbl->nk", Z, cluster(lab_th))


def score_match(m, Zimg, lab_th):
    """cross-encoder: build each combination and ask how CONSISTENT it is.

    A linear head would give f(a+b) = f(a)+f(b) -- purely additive, no interaction, and it
    would collapse back to the bi-encoder.  So the combination is encoded to a chord (atan2,
    nonlinear) and scored against a learned 'consistent' chord g."""
    n = len(Zimg)
    Zk = Zimg.unsqueeze(1) + cluster(lab_th).unsqueeze(0)           # (n, 10, B, L)
    t = m.match(Zk.flatten(2))                                      # (n, 10, B)
    return torch.cos(t - m.g).sum(-1)


# ------------------------------------------------------------------ data
def load(split, n):
    X, y = L1M.mnist_labelled(split, n)
    pos = range(0, 28 - P + 1, 2)
    pa = np.stack([[im[i:i+P, j:j+P].ravel() for i in pos for j in pos] for im in X])
    pa = pa - pa.mean(-1, keepdims=True)
    return torch.tensor(pa, dtype=torch.float32), torch.tensor(y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--train", type=int, default=6000); ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--e2e", type=int, nargs="+", default=[0, 1], help="0 = L1 frozen, 1 = end to end")
    ap.add_argument("--depth", type=int, nargs="+", default=[2], help="learned stages above L1")
    ap.add_argument("--readout", default="dot", choices=["dot", "match"])
    a = ap.parse_args()

    W = np.load(ROOT / "results" / "W_coherent.npy")
    X0, XS = CO.pair_data(n_img=1500)
    _, _, _, _, rho = CO.train(X0, XS, *L1M.random_bank(B), 0.05, epochs=1)   # shape only
    rho = np.load(ROOT / "results" / "rho_coherent.npy") if (ROOT / "results" / "rho_coherent.npy").exists() else rho
    Wc = torch.tensor(W[0], dtype=torch.float32); Ws = torch.tensor(W[1], dtype=torch.float32)
    rho_px = torch.tensor(rho, dtype=torch.float32)

    Xtr, ytr = load("train", a.train); Xte, yte = load("t10k", a.test)
    print(f"{len(Xtr)} train / {len(Xte)} test digits, {NG*NG} cells, B={B}, dev={dev}\n")
    import shift as SH
    A_, B_ = SH.pairs(1)
    def l1_coh(mm):
        return float(SH.coherence(A_, B_, mm.Wc.detach().cpu().numpy(),
                                  mm.Ws.detach().cpu().numpy()).mean())
    print(f"  {'L1':>11}{'depth':>7}{'mask p':>8}{'train':>9}{'TEST':>8}{'L1 coherence':>14}")

    with torch.no_grad():
        Ftr = Xtr.reshape(len(Xtr), -1); Fte = Xte.reshape(len(Xte), -1)
    lin = nn.Linear(Ftr.shape[1], 10).to(dev); o = torch.optim.Adam(lin.parameters(), 1e-3)
    for _ in range(12):
        pm = torch.randperm(len(Ftr))
        for i in range(0, len(Ftr)-256, 256):
            k = pm[i:i+256]
            l = F.cross_entropy(lin(Ftr[k].to(dev)), ytr[k].to(dev))
            o.zero_grad(); l.backward(); o.step()
    with torch.no_grad():
        ref = (lin(Fte.to(dev)).argmax(1).cpu() == yte).float().mean().item()
    print(f"  {'linear on the same patch features (reference)':<44}{ref:>10.3f}\n")

    for e2e, dep, p in [(e, d, q) for e in a.e2e for d in a.depth for q in a.mask]:
        torch.manual_seed(0)
        m = TwoTrack(Wc, Ws, rho_px, e2e=bool(e2e), depth=dep).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=a.lr)
        for ep in range(a.epochs):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr) - 64, 64):
                idx = perm[i:i+64]
                x, y = Xtr[idx].to(dev), ytr[idx].to(dev)
                lab_th = m.label(torch.arange(10, device=dev))
                Z = cluster(m.image(x))
                keep = (torch.rand(len(x), device=dev) >= p).float()[:, None, None]
                Z = Z + keep * cluster(m.label(y))
                sc = score_match(m, cluster(m.image(x)), lab_th) if a.readout == "match" \
                     else score(Z, lab_th)
                loss = F.cross_entropy(sc, y)
                opt.zero_grad(); loss.backward(); opt.step()

        @torch.no_grad()
        def acc(X, Y, bs=256):
            m.eval(); lab_th = m.label(torch.arange(10, device=dev)); ok = 0
            for i in range(0, len(X), bs):
                x = X[i:i+bs].to(dev)
                zi = cluster(m.image(x))
                sc = score_match(m, zi, lab_th) if a.readout == "match" else score(zi, lab_th)
                ok += (sc.argmax(1).cpu() == Y[i:i+bs]).sum().item()
            m.train(); return ok / len(X)

        print(f"  {('end-to-end' if e2e else 'frozen'):>11}{dep:>7}{p:>8}"
              f"{acc(Xtr[:2000], ytr[:2000]):>9.3f}{acc(Xte, yte):>8.3f}{l1_coh(m):>14.3f}")
