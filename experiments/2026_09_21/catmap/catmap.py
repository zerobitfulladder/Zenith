"""Is the torus wrap enough nonlinearity to classify?

Every pixel is an angle.  Every layer is a shear of the torus: pick half the angles (A),
leave them alone, and shift the other half (B) by a LINEAR map of A, then wrap:

    theta_B  <-  theta_B + W theta_A + b      (mod 2pi)
    theta_A  <-  theta_A

Inverse is subtraction, since A came through unchanged.  Arnold's cat map is this on two
angles with W = 1; here W is real and learned.  Nowhere in the network is there a ReLU,
a sigmoid, a softmax, or a cos/sin embedding.  The only thing that is not linear is the
wrap.  So if this stack beats a single linear map, the wrap did it.

Readout: cosine of all 784 output angles against ten fixed random chords, softmax.

Angles live in [-pi, pi) and pixel x maps to span*(x - 1/2).  Centring matters: at span=pi
the data sits in [-pi/2, pi/2], well clear of the seam at +-pi, so no pixel value is
ambiguous between 0 and 2pi.  (With the domain [0, 2pi) every black pixel sits ON the seam,
and the inverse of a real-weighted shear then depends on which side it comes back on: a
deterministic failure, not rounding.)  At span=tau, black and white meet at the seam by
design; that is the arm that pays for treating intensity as a genuine angle.

Arms, all with the same optimiser, batch, epochs and depth, so only the arm differs:

    --shift linear             the claim.  wrap is the only nonlinearity
    --shift linear --nowrap    control.  same weights, no wrap between layers, so the L
                               shears collapse to ONE linear map before the cosine readout
    --shift mlp                ceiling.  the shift is  W2 relu(W1 theta_A + b1) + b2,
                               i.e. the linear arm plus a ReLU.  This is what NICE does.
    --shift trig               periodic.  the shift is  A cos(theta_A) + B sin(theta_A) + b, so
                               every gradient in the network is circular.  The trig IS a
                               nonlinearity (Nanda's quadrature embedding), but it is the torus's own
    --logreg                   linear readout on raw pixels, the classical floor

The confound that would make the whole thing meaningless: if the weights stay small, no
angle ever crosses 2pi and the "wrap" arm is linear in practice.  So every run records,
per layer, the fraction of shifted angles that actually wrapped.

    uv run python experiments/2026_09_21/catmap/catmap.py --depth 8 --span pi
    uv run python experiments/2026_09_21/catmap/catmap.py --plot
"""
import argparse, gzip, json, struct, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent / "data" / "mnist"
OUT  = ROOT / "results"
RUNS = OUT / "runs"
TAU  = 2 * np.pi
PI   = np.pi
dev  = "cuda" if torch.cuda.is_available() else "cpu"


def wrap(b):
    """into [-pi, pi).  d/db = 1 almost everywhere: the wrap is invisible to backprop."""
    return torch.remainder(b + PI, TAU) - PI


# ---------------------------------------------------------------- data

def mnist(split):
    tag = "train" if split == "train" else "t10k"
    with gzip.open(DATA / f"{tag}-images-idx3-ubyte.gz") as f:
        _, cnt, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), np.uint8).reshape(cnt, r * c)
    with gzip.open(DATA / f"{tag}-labels-idx1-ubyte.gz") as f:
        f.read(8)
        y = np.frombuffer(f.read(), np.uint8)
    return torch.tensor(X.astype(np.float32) / 255.0), torch.tensor(y.astype(np.int64))


# ---------------------------------------------------------------- splits

def splits(d, L, mix, seed):
    """which angles are untouched (A) and which are shifted (B), per layer."""
    out = []
    if mix == "checker":
        rc = np.arange(d); par = ((rc // 28) + (rc % 28)) % 2
        for l in range(L):
            A = np.where(par == l % 2)[0]; B = np.where(par != l % 2)[0]
            out.append((A, B))
    elif mix == "half":
        h = d // 2
        for l in range(L):
            A, B = (np.arange(h), np.arange(h, d)) if l % 2 == 0 else (np.arange(h, d), np.arange(h))
            out.append((A, B))
    elif mix == "shuffle":
        g = torch.Generator().manual_seed(seed)
        for l in range(L):
            p = torch.randperm(d, generator=g).numpy()
            out.append((p[: d // 2], p[d // 2:]))
    else:
        raise ValueError(mix)
    return [(torch.tensor(A), torch.tensor(B)) for A, B in out]


# ---------------------------------------------------------------- model

class Shear(nn.Module):
    """theta_B += f(theta_A), wrap.  f linear (the claim) or a one-hidden-layer ReLU MLP."""

    def __init__(self, A, B, shift, hidden, init):
        super().__init__()
        self.register_buffer("A", A); self.register_buffer("B", B)
        nA, nB = len(A), len(B)
        if shift == "linear":
            self.f = nn.Linear(nA, nB)
            nn.init.normal_(self.f.weight, 0, init / np.sqrt(nA)); nn.init.zeros_(self.f.bias)
        elif shift == "mlp":
            self.f = nn.Sequential(nn.Linear(nA, hidden), nn.ReLU(), nn.Linear(hidden, nB))
            nn.init.normal_(self.f[-1].weight, 0, init / np.sqrt(hidden)); nn.init.zeros_(self.f[-1].bias)
        elif shift == "trig":
            self.f = nn.Linear(2 * nA, nB)
            nn.init.normal_(self.f.weight, 0, init / np.sqrt(2 * nA)); nn.init.zeros_(self.f.bias)
        else:
            raise ValueError(shift)
        self.trig = shift == "trig"

    def read(self, a):
        """how the shift sees the untouched half: as numbers, or periodically."""
        return torch.cat([a.cos(), a.sin()], 1) if self.trig else a

    def forward(self, th, do_wrap, stats=None):
        s = self.f(self.read(th[:, self.A]))
        b = th[:, self.B] + s
        if stats is not None:                       # how often the wrap fires, and how far
            stats.append(((b.abs() >= PI).float().mean().item(), (s.abs() / TAU).mean().item()))
        if do_wrap:
            b = wrap(b)
        out = th.clone(); out[:, self.B] = b
        return out

    def inverse(self, th):
        s = self.f(self.read(th[:, self.A]))
        out = th.clone(); out[:, self.B] = wrap(th[:, self.B] - s)
        return out


class CatNet(nn.Module):
    def __init__(self, d, depth, mix, shift, hidden, init, wrap, n_cls=10, kappa=10.0, seed=0):
        super().__init__()
        self.wrap = wrap
        self.layers = nn.ModuleList([Shear(A, B, shift, hidden, init) for A, B in splits(d, depth, mix, seed)])
        g = torch.Generator().manual_seed(seed + 1)
        self.register_buffer("chords", torch.rand(n_cls, d, generator=g) * TAU)   # fixed
        self.log_kappa = nn.Parameter(torch.tensor(float(np.log(kappa))))

    def forward(self, th, stats=None):
        for l in self.layers:
            th = l(th, self.wrap, stats)
        return th

    def inverse(self, th):
        for l in reversed(self.layers):
            th = l.inverse(th)
        return th

    def logits(self, th):
        """cleanup: mean cosine agreement with each chord, scaled."""
        return self.log_kappa.exp() * torch.cos(th[:, None, :] - self.chords[None]).mean(-1)


# ---------------------------------------------------------------- train

def accuracy(fn, X, y, bs=2048):
    hit = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            hit += (fn(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return hit / len(X)


def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    Xtr, ytr = mnist("train"); Xte, yte = mnist("test")
    span = {"pi": np.pi, "tau": TAU}[a.span]
    Xtr, ytr, Xte, yte = ((Xtr - .5) * span).to(dev), ytr.to(dev), ((Xte - .5) * span).to(dev), yte.to(dev)
    d = Xtr.shape[1]

    if a.logreg:
        net = nn.Linear(d, 10).to(dev)
        fwd = lambda x: net(x)
        name = f"logreg_{a.span}"
    else:
        net = CatNet(d, a.depth, a.mix, a.shift, a.hidden, a.init, not a.nowrap, seed=a.seed).to(dev)
        fwd = lambda x: net.logits(net(x))
        name = a.name or f"{a.shift}{'_nowrap' if a.nowrap else ''}_{a.mix}_L{a.depth}_{a.span}_init{a.init:g}_s{a.seed}"
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), a.lr)
    hist = {"test": [], "train": [], "wrapfrac": [], "turns": [], "loss": []}
    print(f"{name}   {n_params} params   {dev}")
    t0 = time.time(); collapsed_at = None
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(Xtr), device=dev); tot = 0.0
        for i in range(0, len(Xtr), a.bs):
            idx = perm[i:i + a.bs]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(fwd(Xtr[idx]), ytr[idx])
            loss.backward()
            if a.clip: nn.utils.clip_grad_norm_(net.parameters(), a.clip)
            opt.step(); tot += loss.item() * len(idx)
        net.eval()
        te, tr = accuracy(fwd, Xte, yte), accuracy(fwd, Xtr[:10000], ytr[:10000])
        wf = []
        if not a.logreg:
            with torch.no_grad():
                net(Xte[:2048], stats=wf)
        hist["test"].append(te); hist["train"].append(tr); hist["loss"].append(tot / len(Xtr))
        hist["wrapfrac"].append([w for w, _ in wf]); hist["turns"].append([t for _, t in wf])
        kap = net.log_kappa.exp().item() if not a.logreg else float('nan')
        print(f"  ep {ep + 1:3d}  loss {tot / len(Xtr):.4f}  train {tr:.4f}  test {te:.4f}  kappa {kap:5.1f}"
              + (f"  wrapped/layer {' '.join(f'{w:.2f}' for w, _ in wf)}  turns/layer {' '.join(f'{t:.1f}' for _, t in wf)}" if wf else ""))
        # collapse: accuracy fell far below the run's own best, or the loss is no longer a number
        if te < max(hist["test"]) - 0.25 or not np.isfinite(hist["loss"][-1]):
            collapsed_at = ep + 1
            print(f"  COLLAPSED at epoch {collapsed_at}: test {te:.3f} vs best {max(hist['test']):.3f}.  stopping.")
            break

    res = {"name": name, "args": vars(a), "n_params": n_params, "secs": time.time() - t0,
           "test": hist["test"][-1], "train": hist["train"][-1], "best_test": max(hist["test"]),
           "collapsed_at": collapsed_at, "hist": hist}
    if not a.logreg:
        with torch.no_grad():                       # invertibility: exact in exact arithmetic, chaotic in float32
            rt = {}
            for dt in (torch.float32, torch.float64):
                m = net.to(dt); x = Xte[:512].to(dt)
                err = wrap(m.inverse(m(x)) - x).abs()
                rt[str(dt).split(".")[-1]] = {"max": err.max().item(), "frac_bad": (err > 1e-3).float().mean().item()}
            net.to(torch.float32); res["roundtrip"] = rt
            # what the wraps are worth on the trained weights: same net, wrap switched off
            net.wrap = False; res["test_wrap_off"] = accuracy(fwd, Xte, yte); net.wrap = not a.nowrap
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / f"{name}.json").write_text(json.dumps(res, indent=1))
    rt = res.get("roundtrip", {})
    print(f"  -> {res['test']:.4f} test   (wrap switched off after training: {res.get('test_wrap_off', float('nan')):.4f})"
          + (f"   roundtrip f32 max {rt['float32']['max']:.1e} bad {rt['float32']['frac_bad']:.1e}"
             f"  f64 max {rt['float64']['max']:.1e} bad {rt['float64']['frac_bad']:.1e}" if rt else "")
          + f"   {res['secs']:.0f}s")


# ---------------------------------------------------------------- plot

def plot():
    rs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]
    arms = {"linear": ("wrap (the claim)", "C0", "o"), "linear_nowrap": ("no wrap (control)", "C3", "s"),
            "mlp": ("ReLU MLP shift (ceiling)", "C2", "^"), "trig": ("sin/cos shift (periodic)", "C1", "D")}

    def arm(r):
        if r["args"].get("logreg"): return "logreg"
        return r["args"]["shift"] + ("_nowrap" if r["args"]["nowrap"] else "")

    spans = ["pi", "tau"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for k, sp in enumerate(spans):
        for key, (lab, col, mk) in arms.items():
            sel = sorted((r["args"]["depth"], r) for r in rs
                         if arm(r) == key and r["args"]["span"] == sp and r["args"]["mix"] == "checker"
                         and r["args"]["init"] == 0 and r["args"]["seed"] == 0)
            ok = [(L, r["test"]) for L, r in sel if not r.get("collapsed_at") and r["test"] > 0.9]
            bad = [(L, r) for L, r in sel if r.get("collapsed_at") or r["test"] <= 0.9]
            if ok: ax[k].plot(*zip(*ok), marker=mk, color=col, label=lab)
            for L, r in bad:                      # collapsed: hollow marker on the floor, labelled with the epoch
                y = {"linear": 0.909, "mlp": 0.903, "trig": 0.915}.get(key, 0.906)
                ax[k].plot(L, y, marker=mk, mfc="none", color=col, ls="none")
                txt = f"collapsed ep {r['collapsed_at']}" if r.get("collapsed_at") else "ran away"
                ax[k].annotate(txt, (L, y), fontsize=5.5, ha="left", va="center",
                               xytext=(6, 0), textcoords="offset points", color=col)
        lr = [r["test"] for r in rs if arm(r) == "logreg" and r["args"]["span"] == sp]
        if lr:
            ax[k].axhline(lr[0], color="k", ls=":", lw=1, label="logistic regression on pixels")
        ax[k].set_xscale("log", base=2); ax[k].set_xlabel("depth (shear layers)")
        ax[k].set_ylim(0.9, 0.99)
        ax[k].set_title(f"span = {sp}" + ("   (no seam)" if sp == "pi" else "   (black meets white)"), fontsize=10)
        ax[k].grid(alpha=.3)
    ax[0].set_ylabel("test accuracy"); ax[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Does the wrap classify?  checkerboard split, identity init, 15 epochs", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "accuracy.png", dpi=150)

    # wrap traffic over training, the confound check
    sel = [r for r in rs if arm(r) == "linear" and r["args"]["mix"] == "checker" and r["args"]["init"] == 0 and r["args"]["seed"] == 0]
    if sel:
        fig, ax = plt.subplots(1, len(sel), figsize=(2.6 * len(sel), 2.8), sharey=True, squeeze=False)
        for k, r in enumerate(sorted(sel, key=lambda r: (r["args"]["span"], r["args"]["depth"]))):
            wf = np.array(r["hist"]["wrapfrac"])            # epochs x layers
            for l in range(wf.shape[1]):
                ax[0, k].plot(wf[:, l], lw=1, color=plt.cm.viridis(l / max(1, wf.shape[1] - 1)))
            ax[0, k].set_title(f"L={r['args']['depth']}  span={r['args']['span']}", fontsize=8)
            ax[0, k].set_xlabel("epoch", fontsize=8); ax[0, k].grid(alpha=.3)
        ax[0, 0].set_ylabel("fraction of shifted angles that wrapped", fontsize=8)
        fig.suptitle("Is the wrap actually firing?  one line per layer, dark = first", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / "wrapfrac.png", dpi=150)

    print(f"{'run':<44} {'params':>8} {'test':>7} {'train':>7} {'wrap off':>9}  {'wrapped/layer (last epoch)'}")
    for r in sorted(rs, key=lambda r: r["name"]):
        wf = r["hist"]["wrapfrac"][-1] if r["hist"]["wrapfrac"] else []
        flag = f"  COLLAPSED ep {r['collapsed_at']}" if r.get("collapsed_at") else ""
        print(f"{r['name']:<44} {r['n_params']:>8} {r['test']:7.4f} {r['train']:7.4f} "
              f"{r.get('test_wrap_off', float('nan')):9.4f}  {' '.join(f'{w:.2f}' for w in wf)}{flag}")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--span", choices=["pi", "tau"], default="pi")
    ap.add_argument("--mix", choices=["checker", "half", "shuffle"], default="checker")
    ap.add_argument("--shift", choices=["linear", "mlp", "trig"], default="linear")
    ap.add_argument("--nowrap", action="store_true")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--init", type=float, default=0.0, help="std of the initial shift, radians. 0 = identity start")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)   # 1e-3 diverges in every arm, see results/diag
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip, 0 = off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logreg", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    plot() if a.plot else run(a)
