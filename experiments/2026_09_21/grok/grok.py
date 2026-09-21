"""Modular addition, the grokking task.  Does a network made of angles have anything to grok?

    inputs (a, b) in Z_p x Z_p,  target (a + b) mod p,  p = 97
    train on a random 30% of the p*p pairs, test on the other 70%, full batch, AdamW.

Arms:
    relu    embed(a) ++ embed(b) -> Linear -> ReLU -> Linear -> p logits      weight decay 1 (groks)
    relu0   the same, weight decay 0                                            (memorises, never generalises)
    trig    embed(a) ++ embed(b) -> Linear -> cos,sin capped at one turn -> Linear -> p logits,  wd 1
    torus   each residue r has k learned ANGLES phi_r.  theta = phi_a + phi_b (mod 2pi).
            readout: cleanup, kappa * mean_i cos(theta_i - c_{y,i}) against p learned chords c_y.
            The group operation is built in; only which angle each residue gets is learned.
            No weight decay: decaying an angle towards 0 means nothing.  [asymmetry, noted]

Logged every 50 steps: train and test accuracy.  Reported: the step at which test accuracy
first passes 0.5 and 0.99, and the gap between train and test crossing, which is the plateau.

    uv run python experiments/2026_09_21/grok/grok.py --arm relu
    uv run python experiments/2026_09_21/grok/grok.py --plot
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"; RUNS = OUT / "runs"
TAU = 2 * np.pi
dev = "cuda" if torch.cuda.is_available() else "cpu"


class Trig(nn.Module):
    def __init__(self, bound=TAU):
        super().__init__(); self.bound = bound
    def forward(self, z):
        z = self.bound * torch.tanh(z / self.bound)
        return torch.cat([z.cos(), z.sin()], 1)


class MLP(nn.Module):
    def __init__(self, p, d, h, act):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        if act == "relu":
            self.net = nn.Sequential(nn.Linear(2 * d, h), nn.ReLU(), nn.Linear(h, p))
        else:
            self.net = nn.Sequential(nn.Linear(2 * d, h // 2), Trig(), nn.Linear(h, p))
    def forward(self, a, b):
        return self.net(torch.cat([self.emb(a), self.emb(b)], 1))


class Torus(nn.Module):
    """residue -> k angles; add; cleanup against p chords."""
    def __init__(self, p, k, kappa=10.0):
        super().__init__()
        self.phi = nn.Parameter(torch.rand(p, k) * TAU - np.pi)        # input angles
        self.chords = nn.Parameter(torch.rand(p, k) * TAU - np.pi)     # output chords
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(kappa))))
    def forward(self, a, b):
        th = self.phi[a] + self.phi[b]
        return self.log_kappa.exp() * torch.cos(th[:, None, :] - self.chords[None]).mean(-1)


def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    p = a.p
    A, B = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    A, B = A.ravel(), B.ravel(); Y = (A + B) % p
    perm = torch.randperm(p * p); n_tr = int(a.frac * p * p)
    tr, te = perm[:n_tr], perm[n_tr:]
    A, B, Y = A.to(dev), B.to(dev), Y.to(dev)

    if a.arm in ("relu", "relu0"):
        net = MLP(p, a.d, a.h, "relu"); wd = 1.0 if a.arm == "relu" else 0.0
    elif a.arm == "trig":
        net = MLP(p, a.d, a.h, "trig"); wd = 1.0
    elif a.arm == "torus":
        net = Torus(p, a.k); wd = 0.0
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=wd, betas=(0.9, 0.98))
    name = f"{a.arm}_p{p}_frac{a.frac:g}_s{a.seed}"
    print(f"{name}  {sum(q.numel() for q in net.parameters())} params  wd {wd}  train {n_tr} pairs, test {p*p-n_tr}")

    hist = {"step": [], "train": [], "test": [], "loss": []}; t0 = time.time()
    first = {}
    for step in range(a.steps + 1):
        net.train(); opt.zero_grad()
        loss = nn.functional.cross_entropy(net(A[tr], B[tr]), Y[tr]); loss.backward(); opt.step()
        if step % 50 == 0:
            net.eval()
            with torch.no_grad():
                tra = (net(A[tr], B[tr]).argmax(1) == Y[tr]).float().mean().item()
                tea = (net(A[te], B[te]).argmax(1) == Y[te]).float().mean().item()
            hist["step"].append(step); hist["train"].append(tra); hist["test"].append(tea); hist["loss"].append(loss.item())
            for key, v, thr in (("train50", tra, .5), ("train99", tra, .99), ("test50", tea, .5), ("test99", tea, .99)):
                if key not in first and v >= thr: first[key] = step
            if step % 1000 == 0:
                print(f"  step {step:6d}  loss {loss.item():.4f}  train {tra:.3f}  test {tea:.3f}")
            if tea >= 0.99 and tra >= 0.99 and step - first.get("test99", step) >= 1000:
                break                                        # generalised and stayed there
    RUNS.mkdir(parents=True, exist_ok=True)
    res = {"name": name, "args": vars(a), "wd": wd, "first": first, "final_train": hist["train"][-1],
           "final_test": hist["test"][-1], "secs": time.time() - t0, "hist": hist}
    (RUNS / f"{name}.json").write_text(json.dumps(res))
    torch.save(net.state_dict(), RUNS / f"{name}.pt")
    print(f"  -> train 99% at step {first.get('train99', '-')}, test 50% at {first.get('test50', '-')}, "
          f"test 99% at {first.get('test99', '-')}    final train {res['final_train']:.3f} test {res['final_test']:.3f}   {res['secs']:.0f}s")


def plot():
    rs = [json.loads(q.read_text()) for q in sorted(RUNS.glob("*.json"))]
    style = {"relu": ("ReLU MLP, wd 1", "C2"), "relu0": ("ReLU MLP, wd 0", "C7"),
             "trig": ("cos+sin MLP, wd 1", "C1"), "torus": ("angles + addition + cleanup", "C0")}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    print(f"{'arm':<8} {'params':>7} {'train 99%':>10} {'test 50%':>9} {'test 99%':>9} {'plateau':>8} {'final test':>11}")
    for r in rs:
        arm = r["args"]["arm"]; lab, col = style[arm]; h = r["hist"]; s = np.array(h["step"]) + 1
        ax[0].plot(s, h["train"], color=col, label=lab); ax[1].plot(s, h["test"], color=col, label=lab)
        f = r["first"]; plateau = (f["test99"] - f["train99"]) if "test99" in f and "train99" in f else None
        n = sum(1 for _ in [0]);
        print(f"{arm:<8} {'':>7} {str(f.get('train99','-')):>10} {str(f.get('test50','-')):>9} {str(f.get('test99','-')):>9} "
              f"{str(plateau if plateau is not None else '-'):>8} {r['final_test']:11.3f}")
    for k, t in enumerate(("train accuracy", "test accuracy")):
        ax[k].set_xscale("log"); ax[k].set_xlabel("step"); ax[k].set_title(t, fontsize=10); ax[k].grid(alpha=.3)
    ax[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("(a + b) mod 97, 30% of pairs for training, full batch", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "grok.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["relu", "relu0", "trig", "torus"], default="relu")
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--d", type=int, default=128, help="embedding width for the MLPs")
    ap.add_argument("--h", type=int, default=512, help="hidden features for the MLPs")
    ap.add_argument("--k", type=int, default=32, help="angles per residue for the torus arm")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    plot() if a.plot else run(a)
