"""An invertible torus network: image <-> (label, residual).

Classification destroys information -- 6000 threes must all become "3", and no bijection
does 6000 -> 1.  So we don't throw the rest away, we KEEP it:

    forward   image -> T^784 -> flow -> (head: T^K , residual: T^(784-K))
              read the class off the head, ignore the residual
    backward  pick a class, SAMPLE a residual, run the flow in reverse -> an image

Everything lives on the torus.  Each pixel is an angle (theta = 2*pi*x), and every layer
is a bijection of T^784.

Two ingredients, both unimodular (|det| = 1) so the map is volume preserving:

    permutation     a fixed random shuffle.  Integer matrix, det +-1.  The benign member
                    of the unimodular family -- a cat-map-style shear is equally legal but
                    large integer entries multiply the angle, which aliases and kills the
                    gradient.
    coupling        split the voices; pass half through untouched, shift the other half by
                    theta_B += f(theta_A)  (mod 2pi).  Invertible by subtraction whatever f
                    is, so f can be an arbitrary MLP.  Addition mod 2pi is the torus's own
                    group operation, so this is more native here than in R^n.

Because |det J| = 1 exactly,  log p(x) = log p_base(f(x))  with NO log-det term.  Training
is therefore exact maximum likelihood, and the base carries the whole story:

    base  =  vonMises(head ; chord_y , kappa)        10 fixed chords, one per class
             x  vonMises(residual ; mu, kappa)       learned, per coordinate

The head term is what makes it classify; the residual term is what makes it generate.
Classification is Bayes: the residual factor does not depend on y, so argmax_y reduces to
scoring the head against the ten chords -- a cleanup, exactly as in the choir.
"""
import argparse, gzip, struct, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent / "data" / "mnist"
OUT  = ROOT / "results"
TAU  = 2 * np.pi
dev  = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- data

def mnist(split, n=None):
    """uint8 images, flattened."""
    tag = "train" if split == "train" else "t10k"
    with gzip.open(DATA / f"{tag}-images-idx3-ubyte.gz") as f:
        _, cnt, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), np.uint8).reshape(cnt, r * c)
    with gzip.open(DATA / f"{tag}-labels-idx1-ubyte.gz") as f:
        f.read(8)
        y = np.frombuffer(f.read(), np.uint8)
    return (X[:n], y[:n]) if n else (X, y)


SPAN = TAU            # how much of the circle the intensity range is stretched over


def to_angles(X, rng, span):
    """Dequantise (the pixels are integers; a density on them is otherwise ill-defined)
    and map onto the circle.

    span = 2pi  puts 0 and 255 at the SAME angle -- black meets white.  That seam is the
                price of treating intensity as a genuine torus coordinate.
    span = pi   uses only half the circle, so there is no seam.  The data then occupies a
                band and the flow may push samples into the empty half; we count that."""
    u = (X.astype(np.float64) + rng.random(X.shape)) / 256.0
    return torch.tensor(u * span, dtype=torch.float32)


def to_x(th):
    """angle -> pixel value.

    The data occupies the ARC [0, SPAN] of the circle.  An angle outside it decodes to the
    NEAREST point of that arc -- which matters enormously at the wrap, because an angle
    just below 0 is a hair away from black, not a hair away from white.  Dividing by SPAN
    and clipping gets that backwards and renders the background as pure white."""
    c = SPAN / 2
    d = (th - c + np.pi) % TAU - np.pi                 # signed offset from the arc centre
    d = d.clamp(-c, c) if isinstance(d, torch.Tensor) else np.clip(d, -c, c)
    return (d + c) / SPAN


# ---------------------------------------------------------------- the flow

class Coupling(nn.Module):
    """theta_B += f(theta_A) mod 2pi.  Jacobian is unit triangular, so det = 1 exactly."""

    def __init__(self, d, d_a, hidden):
        super().__init__()
        self.d_a = d_a
        self.net = nn.Sequential(
            nn.Linear(2 * d_a, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d - d_a))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  # start at identity

    def shift(self, a):
        return self.net(torch.cat([a.cos(), a.sin()], 1))

    def forward(self, th):
        a, b = th[:, :self.d_a], th[:, self.d_a:]
        return torch.cat([a, (b + self.shift(a)) % TAU], 1)

    def inverse(self, th):
        a, b = th[:, :self.d_a], th[:, self.d_a:]
        return torch.cat([a, (b - self.shift(a)) % TAU], 1)


class TorusFlow(nn.Module):
    """Stack of coupling blocks, each preceded by a fixed permutation.

    The permutations are NOT free random draws.  A block only shifts the coordinates that
    land in its second half, so with L independent permutations a coordinate escapes every
    block with probability 2^-L -- at L=6 that left 22 of 784 pixels passing through the
    network bit-identical, stuck at the chord's arbitrary value and visible as specks in
    every generated image.  Resampling cannot fix it (all-covered has probability ~5e-6),
    so coverage is built in: the coordinates are partitioned into L groups and group i is
    forced into block i's shifted half.  The rest of each permutation stays random."""

    def __init__(self, d, n_layers, hidden, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        d_a = d // 2
        self.couplings = nn.ModuleList([Coupling(d, d_a, hidden) for _ in range(n_layers)])
        groups = torch.randperm(d, generator=g).chunk(n_layers)
        for i in range(n_layers):
            forced = groups[i]                                   # must be shifted in block i
            mask = torch.ones(d, dtype=torch.bool); mask[forced] = False
            rest = torch.masked_select(torch.arange(d), mask)
            rest = rest[torch.randperm(len(rest), generator=g)]
            n_extra = (d - d_a) - len(forced)                    # fill the rest of the b half
            assert n_extra >= 0, "too few layers for this split"
            p = torch.cat([rest[n_extra:], forced, rest[:n_extra]])
            self.register_buffer(f"perm{i}", p)
            self.register_buffer(f"iperm{i}", torch.argsort(p))

    def forward(self, th):
        for i, c in enumerate(self.couplings):
            th = c(th[:, getattr(self, f"perm{i}")])
        return th

    def inverse(self, th):
        for i in reversed(range(len(self.couplings))):
            th = self.couplings[i].inverse(th)[:, getattr(self, f"iperm{i}")]
        return th


# ---------------------------------------------------------------- the base

def vm_logpdf(th, mu, kappa):
    """log vonMises.  log I0(k) = log i0e(k) + k keeps large kappa finite."""
    return kappa * torch.cos(th - mu) - np.log(TAU) - torch.log(torch.special.i0e(kappa)) - kappa


class Base(nn.Module):
    """Ten fixed chords for the head; a learned product of vonMises for the residual."""

    def __init__(self, d, K, n_cls=10, kappa_head=8.0, learn_kappa=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.K, self.learn_kappa = K, learn_kappa
        self.register_buffer("chords", torch.rand(n_cls, K, generator=g) * TAU)
        self.raw_kh = nn.Parameter(torch.tensor(float(np.log(np.exp(kappa_head) - 1))),
                                   requires_grad=learn_kappa)
        self.mu_r  = nn.Parameter(torch.zeros(d - K))
        self.raw_k = nn.Parameter(torch.zeros(d - K))          # softplus(0) = 0.69

    def kappa_r(self):
        return nn.functional.softplus(self.raw_k) + 1e-3

    def kappa_head(self):
        return nn.functional.softplus(self.raw_kh) + 1e-3

    def log_head(self, head, y):
        return vm_logpdf(head, self.chords[y], self.kappa_head()).sum(1)

    def log_res(self, res):
        return vm_logpdf(res, self.mu_r, self.kappa_r()).sum(1)

    def scores(self, head):
        """cleanup: score the head against all ten chords."""
        return torch.cos(head[:, None, :] - self.chords[None]).sum(-1)


# ---------------------------------------------------------------- reference

def logreg(Xtr, ytr, Xte, yte, epochs=12):
    m = nn.Linear(784, 10).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xtr = (Xtr.float() / 255).to(dev); ytr = ytr.to(dev)
    for _ in range(epochs):
        for i in range(0, len(Xtr), 256):
            opt.zero_grad()
            nn.functional.cross_entropy(m(Xtr[i:i+256]), ytr[i:i+256]).backward(); opt.step()
    with torch.no_grad():
        return (m((Xte.float() / 255).to(dev)).argmax(1).cpu() == yte).float().mean().item()


# ---------------------------------------------------------------- figures

def wrap(d):
    return (d + np.pi) % TAU - np.pi


def _grid(ax, img, title=None):
    ax.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=7, pad=2)


def fig_arch(K, D, n_layers, path):
    """What the network is, drawn."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 4.6); ax.axis("off")
    box = dict(boxstyle="round,pad=0.32", fc="#eef3fa", ec="#39506e", lw=1.3)
    dash = dict(boxstyle="round,pad=0.32", fc="#fdf1e2", ec="#b4762a", lw=1.3)

    ax.text(0.85, 3.5, "image\n28x28", ha="center", va="center", fontsize=9, bbox=box)
    ax.text(0.85, 2.0, r"$\theta = 2\pi x$" + "\neach pixel\nan angle", ha="center", va="center",
            fontsize=8.5, bbox=dash)
    ax.annotate("", (0.85, 2.75), (0.85, 3.12), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(0.85, 0.75, r"$T^{784}$", ha="center", va="center", fontsize=11)

    xs = [2.7, 4.5, 6.3]
    for i, x in enumerate(xs):
        lbl = "permute\n" + r"$|\det|=1$" + "\n\ncoupling\n" + r"$\theta_B \!+\!=\! f(\theta_A)$"
        ax.text(x, 2.6, lbl, ha="center", va="center", fontsize=8, bbox=box)
        ax.annotate("", (x - 0.42, 3.55), (x - 1.32, 3.55),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#39506e"))
        ax.annotate("", (x - 1.32, 1.65), (x - 0.42, 1.65),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#b4762a"))
    ax.text(xs[1], 4.15, f"x {n_layers} blocks   (every step a bijection of the torus)",
            ha="center", fontsize=9, style="italic")
    ax.text(1.75, 3.75, "forward", ha="center", fontsize=8, color="#39506e")
    ax.text(1.75, 1.32, "inverse", ha="center", fontsize=8, color="#b4762a")

    ax.annotate("", (7.85, 3.55), (6.75, 3.55), arrowprops=dict(arrowstyle="->", lw=1.5, color="#39506e"))
    ax.annotate("", (6.75, 1.65), (7.85, 1.65), arrowprops=dict(arrowstyle="->", lw=1.5, color="#b4762a"))
    ax.text(8.7, 3.3, f"head\n{K} angles", ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#e6f4ea", ec="#2f7d4f", lw=1.3))
    ax.text(8.7, 1.65, f"residual\n{D-K} angles", ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4e8f4", ec="#7d2f6e", lw=1.3))
    ax.text(10.6, 3.3, "cleanup vs\n10 chords\n" + r"$\to$ label", ha="center", va="center",
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#e6f4ea", ec="#2f7d4f", lw=1.3))
    ax.annotate("", (9.9, 3.3), (9.35, 3.3), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(10.6, 1.65, "sample it\n(or keep a\nreal one)", ha="center", va="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f4e8f4", ec="#7d2f6e", lw=1.3))
    ax.annotate("", (9.35, 1.65), (9.9, 1.65), arrowprops=dict(arrowstyle="<-", lw=1.4))
    ax.text(9.65, 0.55, "image  <->  (label, residual)", ha="center", fontsize=10.5, style="italic")
    fig.suptitle("Invertible torus network", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_curves(hist, path):
    ep = np.arange(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(ep, hist["loss"], lw=1.6, color="#39506e")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("nats"); ax[0].set_title("negative log-likelihood", fontsize=10)
    ax[0].axhline(784 * np.log(TAU), ls="--", lw=1, color="#999")
    ax[0].text(len(ep) * .55, 784 * np.log(TAU) * .97, "uniform on $T^{784}$", fontsize=7.5, color="#777")
    ax[1].plot(ep, hist["test"], lw=1.6, color="#2f7d4f", label="test")
    ax[1].plot(ep, hist["train"], lw=1.2, color="#8fbf9f", label="train")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].set_ylim(0, 1)
    ax[1].set_title("classification (cleanup on the head)", fontsize=10); ax[1].legend(fontsize=8)
    for a in ax: a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_roundtrip(flow, Th, path, n=8):
    flow.eval()
    with torch.no_grad():
        x = Th[:n].to(dev); back = flow.inverse(flow(x))
        err = (back - x).abs(); err = torch.minimum(err, TAU - err).max().item()
    fig, axes = plt.subplots(2, n, figsize=(n * .78, 2.1))
    for j in range(n):
        _grid(axes[0, j], to_x(x[j]).cpu().numpy())
        _grid(axes[1, j], to_x(back[j]).cpu().numpy())
    axes[0, 0].set_ylabel("in", fontsize=8, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("back", fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(f"$f^{{-1}}(f(x))$  -  max error {err:.1e} rad  (float32 is ~1e-7)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88]); fig.savefig(path, dpi=150); plt.close(fig)


def fig_samples(flow, base, path, per_class=8):
    """Backward from the FITTED base: the honest generative test."""
    flow.eval()
    with torch.no_grad():
        rows = []
        for y in range(10):
            head = base.chords[y].repeat(per_class, 1) + \
                   torch.randn(per_class, base.K, device=dev) / np.sqrt(base.kappa_head().item())
            k = base.kappa_r()
            sd = torch.clamp(1.0 / torch.sqrt(k), max=TAU)
            res = base.mu_r + torch.randn(per_class, len(k), device=dev) * sd
            rows.append(to_x(flow.inverse(torch.cat([head % TAU, res % TAU], 1))).cpu().numpy())
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * .62, 10 * .62))
    for y in range(10):
        for j in range(per_class): _grid(axes[y, j], rows[y][j])
        axes[y, 0].set_ylabel(str(y), fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Sampled residual from the fitted base", fontsize=10)
    fig.tight_layout(rect=[0.02, 0, 1, 0.96]); fig.savefig(path, dpi=150); plt.close(fig)


def fig_transfer(flow, base, Th, y, path, n_src=8):
    """Keep a REAL image's residual, swap the head to each class chord, invert.

    This separates the two failure modes.  If the factorisation is real, the digit should
    change and the style should stay.  If generation only fails because the base does not
    fit, this still works -- the residual is a genuine one."""
    flow.eval()
    src = [int(np.where(y.numpy() == c)[0][0]) for c in range(n_src)]
    with torch.no_grad():
        th = flow(Th[src].to(dev))
        res = th[:, base.K:]
        cols = [to_x(Th[src]).numpy()]
        for c in range(10):
            head = base.chords[c].repeat(len(src), 1)
            cols.append(to_x(flow.inverse(torch.cat([head, res], 1))).cpu().numpy())
    fig, axes = plt.subplots(len(src), 11, figsize=(11 * .62, len(src) * .62))
    for r in range(len(src)):
        for c in range(11):
            _grid(axes[r, c], cols[c][r], title=("orig" if c == 0 else f"->{c-1}") if r == 0 else None)
    fig.suptitle("Real residual kept, head swapped to each class chord", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(path, dpi=150); plt.close(fig)


def fig_interp(flow, Th, y, path, steps=9):
    """Walk the short way round the torus between two real images, in latent space."""
    flow.eval()
    pairs = [(int(np.where(y.numpy() == a)[0][0]), int(np.where(y.numpy() == b)[0][0]))
             for a, b in [(0, 1), (3, 8), (4, 9), (5, 6), (2, 7)]]
    fig, axes = plt.subplots(len(pairs), steps, figsize=(steps * .62, len(pairs) * .62))
    with torch.no_grad():
        for r, (i, j) in enumerate(pairs):
            t1, t2 = flow(Th[[i]].to(dev)), flow(Th[[j]].to(dev))
            d = torch.tensor(wrap((t2 - t1).cpu().numpy()), device=dev)
            for s, t in enumerate(np.linspace(0, 1, steps)):
                img = to_x(flow.inverse((t1 + t * d) % TAU)).cpu().numpy()[0]
                _grid(axes[r, s], img, title=f"{t:.2f}" if r == 0 else None)
    fig.suptitle("Interpolation between two real images, in torus latent space", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(path, dpi=150); plt.close(fig)


def fig_seam(flow, base, Th, path):
    """Where do the pixel values land?  The circle glues black to white, so noise near
    theta=0 comes back as salt-and-pepper.  Is that what we are seeing?"""
    flow.eval()
    with torch.no_grad():
        k = base.kappa_r(); sd = torch.clamp(1.0 / torch.sqrt(k), max=TAU)
        head = base.chords[0].repeat(256, 1)
        res = base.mu_r + torch.randn(256, len(k), device=dev) * sd
        gen = to_x(flow.inverse(torch.cat([head, res % TAU], 1))).cpu().numpy().ravel()
    real = to_x(Th[:256]).numpy().ravel()
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for a, v, t in ((ax[0], real, "real MNIST"), (ax[1], gen, "sampled from base")):
        a.hist(v, bins=64, range=(0, max(1.0, float(v.max()))), color="#39506e"); a.set_title(t, fontsize=10)
        a.set_xlabel("pixel value"); a.set_yscale("log")
    fig.suptitle(f"Pixel marginals   (span = {SPAN/np.pi:.0f}pi)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_chords(flow, base, path, noises=(0.0, 0.15, 0.35, 0.8)):
    """Invert each class chord.  With no residual this is the ONLY image the class has --
    the inverse is a function, not a sample.  Rows add angular noise around the chord to
    show what the neighbourhood of a chord decodes to."""
    flow.eval()
    fig, axes = plt.subplots(len(noises), 10, figsize=(10 * .62, len(noises) * .68))
    with torch.no_grad():
        for r, sd in enumerate(noises):
            th = base.chords.clone()
            if sd > 0: th = (th + torch.randn_like(th) * sd) % TAU
            img = to_x(flow.inverse(th)).cpu().numpy()
            for c in range(10):
                _grid(axes[r, c], img[c], title=str(c) if r == 0 else None)
            axes[r, 0].set_ylabel(f"sd {sd:g}", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Each class chord, run backwards", fontsize=10)
    fig.tight_layout(rect=[0.03, 0, 1, 0.93]); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--span", choices=["tau", "pi"], default="tau")
    ap.add_argument("--headkappa", choices=["fix", "learn"], default="fix")
    ap.add_argument("--tag", default="")
    ap.add_argument("--load", action="store_true", help="skip training, reuse the checkpoint")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0); torch.manual_seed(0)
    SPAN = TAU if a.span == "tau" else np.pi
    tag = a.tag or a.span
    CKPT = OUT / f"torusflow_{tag}.pt"

    Xtr, ytr = mnist("train", a.n); Xte, yte = mnist("test", 5000)
    Thtr, Thte = to_angles(Xtr, rng, SPAN), to_angles(Xte, rng, SPAN)
    ytr_t, yte_t = torch.tensor(ytr.astype(np.int64)), torch.tensor(yte.astype(np.int64))
    D = 784
    print(f"{a.n} train / {len(Xte)} test on T^{D}, {a.layers} coupling layers, "
          f"head K={a.K}, residual {D - a.K}, span {a.span}, device {dev}\n")

    flow = TorusFlow(D, a.layers, a.hidden).to(dev)
    base = Base(D, a.K, learn_kappa=(a.headkappa == 'learn')).to(dev)
    Xg, yg = Thtr.to(dev), ytr_t.to(dev)
    hist = {"loss": [], "test": [], "train": []}

    if a.load and CKPT.exists():
        ck = torch.load(CKPT, map_location=dev)
        flow.load_state_dict(ck["flow"])
        missing, _ = base.load_state_dict(ck["base"], strict=False)
        if missing: print(f"  (older checkpoint: {list(missing)} left at init)")
        hist = ck["hist"]
        print("  loaded checkpoint\n")
    else:
        opt = torch.optim.Adam(list(flow.parameters()) + list(base.parameters()), 1e-3)
        t0 = time.time()
        for ep in range(a.epochs):
            flow.train()
            idx = torch.randperm(len(Xg), device=dev); tot = 0.0
            for i in range(0, len(idx), a.batch):
                j = idx[i:i+a.batch]
                opt.zero_grad()
                th = flow(Xg[j])
                loss = -(base.log_head(th[:, :a.K], yg[j]) + base.log_res(th[:, a.K:])).mean()
                loss.backward(); opt.step(); tot += loss.item() * len(j)
            flow.eval()
            with torch.no_grad():
                te = (base.scores(flow(Thte.to(dev))[:, :a.K]).argmax(1).cpu() == yte_t).float().mean().item()
                tr = (base.scores(flow(Xg[:5000])[:, :a.K]).argmax(1).cpu() == ytr_t[:5000]).float().mean().item()
            hist["loss"].append(tot/len(idx)); hist["test"].append(te); hist["train"].append(tr)
            if (ep+1) % 5 == 0 or ep == 0:
                print(f"  epoch {ep+1:3d}   -logp {tot/len(idx):8.1f} nats   train {tr:.4f}   test {te:.4f}")
        torch.save({"flow": flow.state_dict(), "base": base.state_dict(), "hist": hist}, CKPT)
        print(f"  ({time.time() - t0:.0f}s)")

    with torch.no_grad():
        x = Thte[:1000].to(dev)
        err = (flow.inverse(flow(x)) - x).abs(); err = torch.minimum(err, TAU - err)
        te = (base.scores(flow(Thte.to(dev))[:, :a.K]).argmax(1).cpu() == yte_t).float().mean().item()
        tr = (base.scores(flow(Xg)[:, :a.K]).argmax(1).cpu() == ytr_t).float().mean().item()
    print(f"\n  round-trip |f^-1(f(x)) - x|   max {err.max():.3e}   mean {err.mean():.3e} rad")
    print(f"  train acc {tr:.4f}   test acc {te:.4f}")
    print(f"  logistic regression on raw pixels   "
          f"{logreg(torch.tensor(Xtr.copy()), ytr_t, torch.tensor(Xte.copy()), yte_t):.4f}")
    if a.K < D:
        k = base.kappa_r().detach()
        print(f"  residual kappa: min {k.min():.2f}  median {k.median():.2f}  max {k.max():.2f}"
              f"   (0 = uniform)", end="")
    print(f"   -logp {hist['loss'][-1]:.0f} vs uniform {784*np.log(TAU):.0f} nats")

    # how far did the images actually move toward their chord?
    with torch.no_grad():
        th = flow(Thte.to(dev))[:, :a.K]
        cs = torch.cos(th[:, None, :] - base.chords[None]).mean(-1)      # per-candidate, per-dim
        own = cs[torch.arange(len(cs)), yte_t.to(dev)]
        other = (cs.sum(1) - own) / 9
    print(f"  cos to own chord {own.mean():.3f}   to the other nine {other.mean():.3f}"
          f"   (1.0 = landed exactly on it)   kappa_head {base.kappa_head().item():.2f}")

    # how entangled are head and residual?  swap the head, see what it does to the pixels.
    if a.K < D:
      with torch.no_grad():
        th = flow(Thte[:512].to(dev))
        keep = flow.inverse(th)
        wrong = (yte_t[:512].to(dev) + 1) % 10
        swapped = flow.inverse(torch.cat([base.chords[wrong], th[:, a.K:]], 1))
        d = (to_x(swapped) - to_x(keep)).abs().mean().item()
        k = base.kappa_r(); sd = torch.clamp(1.0 / torch.sqrt(k), max=TAU)
        res = (base.mu_r + torch.randn(512, len(k), device=dev) * sd) % TAU
        phi = flow.inverse(torch.cat([base.chords[yte_t[:512].to(dev)], res], 1)) % TAU
        far = torch.where(phi <= SPAN, torch.zeros_like(phi),
                          torch.minimum(phi - SPAN, TAU - phi))
        oob = (far > 1e-6).float().mean().item()
        farm = far[far > 1e-6].median().item() if (far > 1e-6).any() else 0.0
      print(f"  head swap moves the average pixel by {d:.3f}  (0 = head and residual factorise)")
      print(f"  sampled angles outside the arc  {oob:.1%}, median {farm:.3f} rad past the end"
            f"  (the arc is {SPAN:.2f} rad long)")

    OUT_ARM = OUT / f"arm-{tag}"; OUT_ARM.mkdir(parents=True, exist_ok=True)
    fig_arch(a.K, D, a.layers, OUT_ARM / "architecture.png")
    fig_curves(hist, OUT_ARM / "curves.png")
    fig_roundtrip(flow, Thte, OUT_ARM / "roundtrip.png")
    if a.K == D:
        fig_chords(flow, base, OUT_ARM / "chords.png")
    else:
        fig_samples(flow, base, OUT_ARM / "samples.png")
        fig_transfer(flow, base, Thte, yte_t, OUT_ARM / "transfer.png")
    fig_interp(flow, Thte, yte_t, OUT_ARM / "interpolate.png")
    fig_seam(flow, base, Thte, OUT_ARM / "seam.png")
    print(f"  figures -> {OUT_ARM}/")
