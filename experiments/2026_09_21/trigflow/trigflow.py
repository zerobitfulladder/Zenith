"""An invertible torus network with capped sine/cosine shears, and what it generates.

Every pixel is an angle in [-pi, pi); pixel x maps to span*(x - 1/2).  A layer leaves half
the angles alone (checkerboard, alternating) and shifts the other half by a weighted sum of
the cosines and sines of the untouched half, with the shift capped at one turn:

    s        =  W [cos theta_A, sin theta_A] + b
    s        =  cap * tanh(s / cap)                cap = 2pi: at most one turn per layer
    theta_B  <- theta_B + s   (mod 2pi)
    theta_A  <- theta_A

Inverse is subtraction of the same shift.  Every gradient is circular, because the state
is only ever read through cos and sin.  The cap makes a runaway impossible.

Readout: mean cosine of all 784 output angles against ten fixed random chords, softmax.
Trained for classification only.  Then run backwards from  chord_y + noise  to see what a
label plus some noise looks like as an image.

    uv run python experiments/2026_09_21/trigflow/trigflow.py                 # train, then generate
    uv run python experiments/2026_09_21/trigflow/trigflow.py --generate      # from the saved checkpoint
"""
import argparse, gzip, json, struct, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent / "data"
OUT  = ROOT / "results"
TAU, PI = 2 * np.pi, np.pi
dev  = "cuda" if torch.cuda.is_available() else "cpu"


def wrap(b):
    return torch.remainder(b + PI, TAU) - PI


def mnist(split, data="mnist"):
    tag = "train" if split == "train" else "t10k"
    with gzip.open(DATA / data / f"{tag}-images-idx3-ubyte.gz") as f:
        _, cnt, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), np.uint8).reshape(cnt, r * c)
    with gzip.open(DATA / data / f"{tag}-labels-idx1-ubyte.gz") as f:
        f.read(8)
        y = np.frombuffer(f.read(), np.uint8)
    return torch.tensor(X.astype(np.float32) / 255.0), torch.tensor(y.astype(np.int64))


def to_angle(x, span):
    return (x - 0.5) * span


def to_pixel(th, span):
    """nearest point of the data arc, back to [0, 1]."""
    return (wrap(th).clamp(-span / 2, span / 2) / span + 0.5)


# ---------------------------------------------------------------- model

class Shear(nn.Module):
    def __init__(self, A, B, cap):
        super().__init__()
        self.register_buffer("A", A); self.register_buffer("B", B)
        self.f = nn.Linear(2 * len(A), len(B)); self.cap = cap
        nn.init.zeros_(self.f.weight); nn.init.zeros_(self.f.bias)      # start at the identity

    def shift(self, th):
        a = th[:, self.A]
        s = self.f(torch.cat([a.cos(), a.sin()], 1))
        return self.cap * torch.tanh(s / self.cap) if self.cap else s

    def forward(self, th):
        out = th.clone(); out[:, self.B] = wrap(th[:, self.B] + self.shift(th)); return out

    def inverse(self, th):
        out = th.clone(); out[:, self.B] = wrap(th[:, self.B] - self.shift(th)); return out


class TrigFlow(nn.Module):
    def __init__(self, d, depth, cap, n_cls=10, kappa=10.0, seed=0):
        super().__init__()
        par = torch.tensor(((np.arange(d) // 28) + (np.arange(d) % 28)) % 2)
        self.layers = nn.ModuleList([Shear(torch.where(par == l % 2)[0], torch.where(par != l % 2)[0], cap)
                                     for l in range(depth)])
        g = torch.Generator().manual_seed(seed + 1)
        self.register_buffer("chords", torch.rand(n_cls, d, generator=g) * TAU - PI)
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(kappa))))

    def forward(self, th):
        for l in self.layers: th = l(th)
        return th

    def inverse(self, th):
        for l in reversed(self.layers): th = l.inverse(th)
        return th

    def logits(self, th):
        return self.log_kappa.exp() * torch.cos(th[:, None, :] - self.chords[None]).mean(-1)


# ---------------------------------------------------------------- train

def accuracy(fn, X, y, bs=2048):
    with torch.no_grad():
        return sum((fn(X[i:i+bs]).argmax(1) == y[i:i+bs]).sum().item() for i in range(0, len(X), bs)) / len(X)


def train(a):
    torch.manual_seed(a.seed)
    Xtr, ytr = mnist("train", a.data); Xte, yte = mnist("test", a.data)
    Xtr, Xte = to_angle(Xtr, a.span).to(dev), to_angle(Xte, a.span).to(dev); ytr, yte = ytr.to(dev), yte.to(dev)
    net = TrigFlow(784, a.depth, a.cap, seed=a.seed).to(dev)
    fwd = lambda x: net.logits(net(x))
    opt = torch.optim.Adam(net.parameters(), a.lr)
    print(f"{a.tag}   {sum(p.numel() for p in net.parameters())} params   {dev}")
    hist = {"test": [], "train": [], "loss": []}; t0 = time.time()
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(Xtr), device=dev); tot = 0.0
        for i in range(0, len(Xtr), a.bs):
            idx = perm[i:i+a.bs]; opt.zero_grad()
            z = net(Xtr[idx])
            loss = nn.functional.cross_entropy(net.logits(z), ytr[idx])
            if a.pull:                                  # be NEAR the chord, not just nearer than the others
                loss = loss + a.pull * (1 - torch.cos(z - net.chords[ytr[idx]]).mean())
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        net.eval(); te, tr = accuracy(fwd, Xte, yte), accuracy(fwd, Xtr[:10000], ytr[:10000])
        with torch.no_grad():
            turns = [(l.shift(Xte[:1024]).abs() / TAU).mean().item() for l in net.layers]
            own = torch.cos(net(Xte[:2048]) - net.chords[yte[:2048]]).mean().item()
        hist["test"].append(te); hist["train"].append(tr); hist["loss"].append(tot / len(Xtr))
        print(f"  ep {ep+1:3d}  loss {tot/len(Xtr):.4f}  train {tr:.4f}  test {te:.4f}  cos to own chord {own:.3f}  "
              f"turns/layer {' '.join(f'{t:.2f}' for t in turns)}")
        if te < max(hist["test"]) - 0.25: print("  COLLAPSED, stopping"); break
    with torch.no_grad():
        err = wrap(net.inverse(net(Xte[:512])) - Xte[:512]).abs().max().item()
    print(f"  -> {hist['test'][-1]:.4f} test   roundtrip max {err:.1e}   {time.time()-t0:.0f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "args": vars(a), "hist": hist, "roundtrip": err}, OUT / f"{a.tag}.pt")
    return net


# ---------------------------------------------------------------- generate

def generate(a, net=None):
    ck = torch.load(OUT / f"{a.tag}.pt", map_location=dev)
    args = ck["args"]
    if net is None:
        net = TrigFlow(784, args["depth"], args["cap"]).to(dev); net.load_state_dict(ck["net"])
    net.eval()
    sds, per = [0.0, 0.1, 0.3, 0.6, 1.0], 3
    torch.manual_seed(0)
    fig, axes = plt.subplots(10, 1 + len(sds) * per, figsize=((1 + len(sds) * per) * .55, 10 * .58))
    Xte, yte = mnist("test", args["data"])
    with torch.no_grad():
        for y in range(10):
            real = Xte[(yte == y).nonzero()[0, 0]].numpy().reshape(28, 28)
            axes[y, 0].imshow(real, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            for k, sd in enumerate(sds):
                th = net.chords[y][None].repeat(per, 1) + torch.randn(per, 784, device=dev) * sd
                img = to_pixel(net.inverse(wrap(th)), args["span"]).cpu().numpy()
                for j in range(per):
                    axes[y, 1 + k * per + j].imshow(img[j].reshape(28, 28), cmap="gray", vmin=0, vmax=1,
                                                    interpolation="nearest")
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("real", fontsize=7)
    for k, sd in enumerate(sds): axes[0, 1 + k * per + 1].set_title(f"sd = {sd}", fontsize=7)
    for y in range(10): axes[y, 0].set_ylabel(str(y), rotation=0, fontsize=8, ha="right", va="center")
    fig.suptitle(f"chord_y + N(0, sd) on all 784 angles, run backwards   ({a.tag}, "
                 f"test {ck['hist']['test'][-1]:.3f})", fontsize=9)
    fig.tight_layout(rect=[0.02, 0, 1, 0.95]); fig.savefig(OUT / f"{a.tag}_generated.png", dpi=150)
    # how far out of the data arc do the decoded angles land?
    with torch.no_grad():
        th = net.chords + torch.randn(10, 784, device=dev) * 0.3
        phi = wrap(net.inverse(wrap(th)))
        oob = (phi.abs() > args["span"] / 2).float().mean().item()
    print(f"  decoded angles outside the data arc at sd=0.3: {oob:.1%}   -> {OUT / f'{a.tag}_generated.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mnist")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--span", type=float, default=PI)
    ap.add_argument("--cap", type=float, default=TAU, help="max shift per layer, radians; 0 = none")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pull", type=float, default=0.0, help="weight of the (1 - cos to own chord) term; 0 = off")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--generate", action="store_true", help="skip training, use the checkpoint")
    a = ap.parse_args()
    a.tag = a.tag or f"{a.data}_L{a.depth}_span{a.span/PI:g}pi_cap{a.cap/PI:g}pi" + (f"_pull{a.pull:g}" if a.pull else "")
    net = None if a.generate else train(a)
    generate(a, net)
