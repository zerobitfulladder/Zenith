"""A conventional MLP on MNIST: ReLU against sine-and-cosine, nothing else changed.

    784 -> H -> H -> 10,  cross-entropy,  Adam,  same seeds, same epochs.

    relu      h = relu(Wx + b)                     H units,  H features
    trig      h = [cos(Wx + b), sin(Wx + b)]       H/2 units, H features  (half the weights)
    trig2     h = [cos(Wx + b), sin(Wx + b)]       H units,  2H features (more weights)

No torus, no wrap, no invertibility.  The question is only: what does swapping the
activation do.  cos and sin of the same pre-activation is the circle's own coordinate pair,
so "trig" is a ReLU network whose neurons are angles read as hand positions.

    uv run python experiments/2026_09_21/trigmlp/trigmlp.py --act relu --seed 0
    uv run python experiments/2026_09_21/trigmlp/trigmlp.py --plot
"""
import argparse, gzip, json, struct, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATASETS = {"mnist": ROOT.parent.parent / "data" / "mnist", "fashion": ROOT.parent.parent / "data" / "fashion"}
OUT  = ROOT / "results"; RUNS = OUT / "runs"
dev  = "cuda" if torch.cuda.is_available() else "cpu"


def mnist(split, data="mnist"):
    tag = "train" if split == "train" else "t10k"
    DATA = DATASETS[data]
    with gzip.open(DATA / f"{tag}-images-idx3-ubyte.gz") as f:
        _, cnt, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), np.uint8).reshape(cnt, r * c)
    with gzip.open(DATA / f"{tag}-labels-idx1-ubyte.gz") as f:
        f.read(8)
        y = np.frombuffer(f.read(), np.uint8)
    return torch.tensor(X.astype(np.float32) / 255.0 - 0.5), torch.tensor(y.astype(np.int64))


class Trig(nn.Module):
    """cos and sin of the pre-activation.  With a bound B, the pre-activation is first
    squashed to (-B, B) by B*tanh(z/B): identity for small z, so a unit can never turn more
    than B/2pi turns.  B = pi means at most half a turn either way."""
    def __init__(self, bound=0.0):
        super().__init__(); self.bound = bound
    def forward(self, z):
        if self.bound: z = self.bound * torch.tanh(z / self.bound)
        return torch.cat([z.cos(), z.sin()], 1)


class Phase(nn.Module):
    """Torus arithmetic only.  Each connection j->i is a transposition phi_ij and a vote w_ij:

        z_i  =  sum_j  w_ij * exp(i (theta_j + phi_ij))  =  sum_j C_ij * exp(i theta_j),   C = w e^{i phi}

    output angle = arg z_i (the consensus), agreement r_i = |z_i| / sum_j |C_ij| in [0, 1].
    Scale-free: multiplying all weights changes nothing, so nothing can run away.
    Input and output are unit vectors [cos, sin] (or, with amp=True, r * [cos, sin]:
    the agreement rides along as amplitude and a voice with no consensus goes quiet)."""
    def __init__(self, n_in, n_out, amp=False):
        super().__init__()
        self.A = nn.Parameter(torch.randn(n_out, n_in) / np.sqrt(n_in))
        self.B = nn.Parameter(torch.randn(n_out, n_in) / np.sqrt(n_in))
        self.amp = amp
    def forward(self, u):
        c, s_ = u.chunk(2, 1)                                     # cos and sin (times r if amp)
        zr = c @ self.A.T - s_ @ self.B.T                          # Re(C u)
        zi = s_ @ self.A.T + c @ self.B.T                          # Im(C u)
        if self.amp == "rms":                                      # expected |z| for incoherent inputs
            norm = torch.sqrt((self.A ** 2 + self.B ** 2).sum(1))  # sqrt(sum_j |C_ij|^2)
            zr, zi = zr / norm, zi / norm
            mag = torch.sqrt(zr ** 2 + zi ** 2 + 1e-8)
            return torch.cat([zr / (1 + mag) , zi / (1 + mag)], 1) # squash: r = |z|/(1+|z|) < 1
        if self.amp:
            norm = torch.sqrt(self.A ** 2 + self.B ** 2).sum(1)    # sum_j |C_ij|
            return torch.cat([zr / norm, zi / norm], 1)            # r * (cos, sin), r <= 1
        mag = torch.sqrt(zr ** 2 + zi ** 2 + 1e-8)
        return torch.cat([zr / mag, zi / mag], 1)                  # unit vector: the consensus angle


class PhaseNet(nn.Module):
    """pixels -> angles on half the circle -> two Phase layers -> cosine cleanup against learned chords."""
    def __init__(self, voices, n_cls=10, amp=False, kappa=10.0):
        super().__init__()
        self.l1, self.l2 = Phase(784, voices, amp), Phase(voices, voices, amp)
        self.chords = nn.Parameter(torch.rand(n_cls, voices) * 2 * np.pi - np.pi)
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(kappa))))
    def forward(self, x):
        th = np.pi * x                                             # x in [-0.5, 0.5] -> [-pi/2, pi/2]
        u = torch.cat([th.cos(), th.sin()], 1)
        u = self.l2(self.l1(u))
        c, s_ = u.chunk(2, 1)                                      # mean_i cos(theta_i - chord_yi)
        return self.log_kappa.exp() * (c @ self.chords.cos().T + s_ @ self.chords.sin().T) / c.shape[1]


def mlp(act, H, bound=0.0):
    if act in ("phase", "phaseamp", "phaserms"):
        return PhaseNet(H // 2, amp={"phase": False, "phaseamp": True, "phaserms": "rms"}[act])
    if act == "relu":
        units, feats, nl = H, H, nn.ReLU
    elif act == "trig":
        units, feats, nl = H // 2, H, lambda: Trig(bound)
    elif act == "trig2":
        units, feats, nl = H, 2 * H, lambda: Trig(bound)
    else:
        raise ValueError(act)
    return nn.Sequential(nn.Linear(784, units), nl(), nn.Linear(feats, units), nl(), nn.Linear(feats, 10))


def accuracy(net, X, y, bs=4096):
    with torch.no_grad():
        return sum((net(X[i:i+bs]).argmax(1) == y[i:i+bs]).sum().item() for i in range(0, len(X), bs)) / len(X)


def run(a):
    torch.manual_seed(a.seed)
    Xtr, ytr = mnist("train", a.data); Xte, yte = mnist("test", a.data)
    Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
    net = mlp(a.act, a.hidden, a.bound).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), a.lr)
    name = f"{a.data}_{a.act}{f'_b{a.bound:g}' if a.bound else ''}_H{a.hidden}_lr{a.lr:g}_s{a.seed}"
    print(f"{name}  {n_params} params")
    hist = {"test": [], "train": [], "loss": []}; collapsed_at = None; t0 = time.time()
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(Xtr), device=dev); tot = 0.0
        for i in range(0, len(Xtr), a.bs):
            idx = perm[i:i+a.bs]; opt.zero_grad()
            loss = nn.functional.cross_entropy(net(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        net.eval(); te, tr = accuracy(net, Xte, yte), accuracy(net, Xtr[:10000], ytr[:10000])
        hist["test"].append(te); hist["train"].append(tr); hist["loss"].append(tot / len(Xtr))
        print(f"  ep {ep+1:3d}  loss {tot/len(Xtr):.4f}  train {tr:.4f}  test {te:.4f}")
        if te < max(hist["test"]) - 0.25 or not np.isfinite(hist["loss"][-1]):
            collapsed_at = ep + 1; print(f"  COLLAPSED at epoch {collapsed_at}"); break
    RUNS.mkdir(parents=True, exist_ok=True)
    res = {"name": name, "args": vars(a), "n_params": n_params, "secs": time.time() - t0,
           "test": hist["test"][-1], "train": hist["train"][-1], "best_test": max(hist["test"]),
           "collapsed_at": collapsed_at, "hist": hist}
    (RUNS / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(f"  -> {res['test']:.4f} test   {res['secs']:.0f}s")


def plot(data):
    rs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json")) if json.loads(p.read_text())["args"].get("data", "mnist") == data]
    arms = {"relu": ("ReLU, H units", "C2"), "trig": ("cos+sin, H/2 units (half the weights)", "C1"),
            "trig2": ("cos+sin, H units (2H features)", "C0"),
            "trig_b3.14159": ("cos+sin, H/2, bounded to half a turn", "C3"),
            "trig_b6.28319": ("cos+sin, H/2, bounded to one turn", "C4"),
            "phase": ("phase net: shift + consensus, angle only", "C5"),
            "phaseamp": ("phase net: shift + consensus, with agreement", "C6"),
            "phaserms": ("phase net, agreement normalised by rms", "C8")}
    key = lambda r: r["args"]["act"] + (f"_b{r['args']['bound']:g}" if r["args"].get("bound") else "")
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    print(f"{'arm':<14} {'params':>8} {'seeds':>5} {'test mean':>10} {'sd':>7} {'train mean':>11}")
    for act, (lab, col) in arms.items():
        sel = [r for r in rs if key(r) == act]
        if not sel: continue
        cs = [r["hist"]["test"] for r in sel if not r["collapsed_at"]]
        L = min(map(len, cs)) if cs else 0
        curves = np.array([c[:L] for c in cs])                    # runs of different length: truncate
        if len(curves):
            m = curves.mean(0); ax.plot(np.arange(1, len(m) + 1), m, color=col, label=lab)
            ax.fill_between(np.arange(1, len(m) + 1), curves.min(0), curves.max(0), color=col, alpha=.15)
        te = np.array([r["test"] for r in sel]); tr = np.array([r["train"] for r in sel])
        print(f"{act:<14} {sel[0]['n_params']:>8} {len(sel):>5} {te.mean():10.4f} {te.std():7.4f} {tr.mean():11.4f}"
              + (f"   collapsed: {sum(bool(r['collapsed_at']) for r in sel)}" if any(r["collapsed_at"] for r in sel) else ""))
    lo, hi = (0.95, 0.99) if data == "mnist" else (0.84, 0.90)
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.set_ylim(lo, hi); ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="lower right"); ax.set_title(f"784-H-H-10 on {data}, mean of seeds, band = min..max", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / f"curves_{data}.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=list(DATASETS), default="mnist")
    ap.add_argument("--act", choices=["relu", "trig", "trig2", "phase", "phaseamp", "phaserms"], default="relu")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--bound", type=float, default=0.0, help="cap |pre-activation| at this many radians via B*tanh(z/B); 0 = off")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    plot(a.data) if a.plot else run(a)
