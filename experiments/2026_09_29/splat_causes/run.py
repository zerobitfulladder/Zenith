"""Causes as trees of splats, MNIST: explain the whole digit, name the label,
draw from the label, fill in what is hidden. Nothing fitted by gradient.

The idea under test (2026-09-29 conversation):

    splat      a Gaussian fitted to some ink: centre, shape (length, thickness,
               angle, kept as the log of its covariance so round splats and angle
               wrap-around cause no trouble) and how much ink it carries
    cause      one row of the table: a binary tree of splats. The root is the
               whole digit; each child is kept in its parent's frame (offset in
               units of the parent's length, shape and ink relative to the
               parent's). For every node and each of its 6 numbers: a typical
               value and a spread, learned by counting. Plus label counts.
    explain    fit the root to all the ink; for a shortlisted cause, walk down
               its tree: where the cause splits, split the ink the same way,
               children placed where the cause expects them, then settled on the
               ink (a few rounds of: each pixel to the likelier child, refit).
    cost       (nats) how rare the cause is
             + for every node and number: (value - typical)^2 / (2 spread^2)
               + log spread + the price of writing a number to precision DELTA
             + the ink the leaves fail to draw: sum (image - drawing)^2 / (2 SIG_PIX^2)
               The cheapest cause is the explanation; its label counts the answer.
    learn      right answer: the winning cause takes in this image's numbers
               (running average and spread). Wrong answer: hire a new cause
               from this image. The label acts only here, as the check.
    grow       a new cause splits its first image all the way down, then keeps a
               split only where the whole subtree under it draws more ink than
               its nodes cost (grow, then prune). Afterwards each leaf of a
               cause tries a split on every image it explains; when the split
               has paid on average over N_GROW images, it becomes part of the cause.
    retire     a cause that is picked and keeps being wrong is switched off.
               A cause that is merely unused is kept.
    draw       from a label: pick a cause by its label count, walk down its tree
               with typical values (or values drawn from the spreads), paint leaves.
    fill       with part of the image hidden: hidden pixels start as the cause's
               own drawing, the cause is fitted to visible + imagined ink, the
               drawing replaces the imagined ink, four rounds. Only visible
               pixels are priced, and a node pays only for the share of its ink
               that is visible: missing is not contradicting.

Split: the 60k training file, first 50k to learn, next 2k to choose SIG_PIX on
a 10k-image pilot, the 10k test file for every reported number.
Writes results/: summary.md, metrics.json, run.log, and the figures.
"""
import json, math, os, time
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

D_MAX = 6                         # a safety cap on depth (64 leaves), reported if reached
NN = 2 ** (D_MAX + 1) - 1         # node slots: root 0, children of k at 2k+1 and 2k+2
NK = (NN - 1) // 2                # nodes that may have children
N_EM = 4                          # rounds of settling per split
M_MIN = 0.5                       # least ink a splat may carry
K0 = 2.0                          # the starting spread counts as K0 images
SIG0_FRAC = 0.5                   # starting spread = this x the spread over all images
DELTA = 0.1                       # each number is written to this precision
C0 = 0.5 * math.log(2 * math.pi) - math.log(DELTA)
N_GROW = 10                       # trial splits before a leaf may grow
S_SHORT = 16                      # causes explained in full per image
CH = 32                           # images per training step
CAP = 16384                       # most causes the table can hold
N_LEARN = 50000
PILOT_N, PILOT_VAL = 10000, 2000
SIG_PIX_GRID = [0.2, 0.3, 0.5]
N_FILL = 1000                     # test images for the hidden-part tests
FILL_ITERS = 4
SIG_PIX = 0.3                     # set by the pilot
SMOKE = bool(os.environ.get("SMOKE"))
if SMOKE:                         # a one-minute run of every step, nothing reported
    N_LEARN, PILOT_N, PILOT_VAL, SIG_PIX_GRID, N_FILL = 3000, 1500, 300, [0.3], 100
    OUT = HERE / "smoke"; OUT.mkdir(exist_ok=True)
LOG = open(OUT / "run.log", "w") if __name__ == "__main__" else open(os.devnull, "w")
T0 = time.time()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# ----------------------------------------------------------------- data
def read_idx(path):
    b = Path(path).read_bytes(); nd = b[3]
    dims = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(b, np.uint8, offset=4 + 4 * nd).reshape(dims)


RAW = ROOT / "data/mnist/digits/raw"
Xall = torch.as_tensor(read_idx(RAW / "train-images-idx3-ubyte").astype(np.float32) / 255.0).reshape(-1, 784).to(DEV)
yall = torch.as_tensor(read_idx(RAW / "train-labels-idx1-ubyte").astype(np.int64)).to(DEV)
Xte = torch.as_tensor(read_idx(RAW / "t10k-images-idx3-ubyte").astype(np.float32) / 255.0).reshape(-1, 784).to(DEV)
yte = torch.as_tensor(read_idx(RAW / "t10k-labels-idx1-ubyte").astype(np.int64)).to(DEV)
if SMOKE:
    Xte, yte = Xte[:1000], yte[:1000]
Xtr, ytr = Xall[:N_LEARN], yall[:N_LEARN]
Xva, yva = Xall[N_LEARN:N_LEARN + PILOT_VAL], yall[N_LEARN:N_LEARN + PILOT_VAL]
say(f"data: learn {len(Xtr)}, choose {len(Xva)}, test {len(Xte)}; device {DEV}")

# pixel centres, measured from the middle of the image (x = column, y = row)
_r, _c = torch.meshgrid(torch.arange(28.), torch.arange(28.), indexing="ij")
P = torch.stack([_c.reshape(-1), _r.reshape(-1)], -1).to(DEV) - 13.5
DEPTH = torch.tensor([int(math.log2(k + 1)) for k in range(NN)], device=DEV)
PARENT = torch.tensor([max((k - 1) // 2, 0) for k in range(NN)], device=DEV)
KA = 2 * torch.arange(NK, device=DEV) + 1
KB = KA + 1
PIX_VAR = torch.tensor([1 / 12, 0.0, 1 / 12], device=DEV)   # a pixel is a unit square


# ----------------------------------------------------------------- 2x2 shapes
def eig(cov):
    a, b, c = cov.unbind(-1)
    t = (a + c) / 2; r = torch.sqrt(((a - c) / 2) ** 2 + b ** 2)
    return t + r, 0.5 * torch.atan2(2 * b, a - c)          # largest spread, its angle


def logm(cov):
    a, b, c = cov.unbind(-1)
    t = (a + c) / 2; r = torch.sqrt(((a - c) / 2) ** 2 + b ** 2)
    l1, l2 = torch.log(t + r), torch.log((t - r).clamp_min(1e-4))
    al = (l1 + l2) / 2
    be = torch.where(r > 1e-5, (l1 - l2) / (2 * r).clamp_min(1e-5), 1 / t)
    return torch.stack([al + be * (a - t), be * b, al + be * (c - t)], -1)


def expm(lc):
    p, q, s = lc.unbind(-1)
    t = (p + s) / 2; r = torch.sqrt(((p - s) / 2) ** 2 + q ** 2)
    e = torch.exp(t)
    sh = torch.where(r > 1e-5, torch.sinh(r) / r.clamp_min(1e-5), torch.ones_like(r))
    return torch.stack([e * (torch.cosh(r) + sh * (p - t)), e * sh * q, e * (torch.cosh(r) + sh * (s - t))], -1)


def loggauss(d, cov):
    a, b, c = cov.unbind(-1)
    det = (a * c - b * b).clamp_min(1e-6)
    q = (d[..., 0] ** 2 * c - 2 * d[..., 0] * d[..., 1] * b + d[..., 1] ** 2 * a) / det
    return -0.5 * q - 0.5 * torch.log(det) - math.log(2 * math.pi)


def moments(F, idx):
    S = torch.zeros(F.shape[0], NN, 6, device=DEV).scatter_add_(1, idx[..., None].expand(-1, -1, 6), F)
    m = S[..., 0]; mm = m.clamp_min(1e-6)[..., None]
    mu = S[..., 1:3] / mm
    cov = S[..., 3:6] / mm - torch.stack([mu[..., 0] ** 2, mu[..., 0] * mu[..., 1], mu[..., 1] ** 2], -1) + PIX_VAR
    return m, mu, cov


def feats_of(W):
    x, y = P[:, 0], P[:, 1]
    return torch.stack([W, W * x, W * y, W * x * x, W * x * y, W * y * y], -1)


def has_kids(tree):
    h = torch.zeros_like(tree); h[:, :NK] = tree[:, KA]; return h


def add_kids(tree, where):
    t = tree.clone(); t[:, KA] |= where[:, :NK]; t[:, KB] |= where[:, :NK]; return t


def render(mass, mu, cov, leaf, clamp=True):
    R = torch.zeros(mass.shape[0], 784, device=DEV)
    for d in range(D_MAX + 1):
        k0, k1 = 2 ** d - 1, 2 ** (d + 1) - 1
        lf = leaf[:, k0:k1]
        if not lf.any():
            continue
        dens = loggauss(P[None, None] - mu[:, k0:k1, None], cov[:, k0:k1, None]).exp()
        R += torch.where(lf[..., None], dens * mass[:, k0:k1, None], 0.).sum(1)
    return R.clamp(0, 1) if clamp else R


# ----------------------------------------------------------------- explain one image with one cause
def _explain(W, tree, init, canon):
    """W [B,784] ink; tree [B,NN] the cause's nodes; init [B,NN,6] where the cause
    expects each child (in its parent's frame); canon [B,NN] children placed by the
    fixed rule instead (split along the parent's long axis, upper or left child first)."""
    B = W.shape[0]
    F = feats_of(W)
    cur = torch.zeros(B, 784, dtype=torch.long, device=DEV)
    mass = torch.zeros(B, NN, device=DEV); mu = torch.zeros(B, NN, 2, device=DEV)
    cov = PIX_VAR.expand(B, NN, 3).clone()
    m, u, c = moments(F, cur)
    mass[:, 0], mu[:, 0], cov[:, 0] = m[:, 0].clamp_min(M_MIN), u[:, 0], c[:, 0]
    for d in range(D_MAX):
        k0, k1 = 2 ** d - 1, 2 ** (d + 1) - 1
        ka = torch.arange(2 * k0 + 1, 2 * k1, 2, device=DEV); kb = ka + 1
        split = tree[:, ka]
        if not split.any():
            break
        pu, pc = mu[:, k0:k1], cov[:, k0:k1]
        lam1, phi = eig(pc)
        L = lam1.sqrt(); logL2 = 2 * L.log()
        v = torch.stack([phi.cos(), phi.sin()], -1)
        s = torch.where(v[..., 1].abs() >= 0.38, -torch.sign(v[..., 1]), -torch.ones_like(L))
        off = (0.87 * s * L)[..., None] * v
        ccov = pc - 0.75 * lam1[..., None] * torch.stack([v[..., 0] ** 2, v[..., 0] * v[..., 1], v[..., 1] ** 2], -1)
        guess = []
        for kk, sign in ((ka, 1.0), (kb, -1.0)):
            rel = init[:, kk]
            gu = pu + L[..., None] * rel[..., 0:2]
            lc = rel[..., 2:5].clone(); lc[..., 0] += logL2; lc[..., 2] += logL2
            gc = expm(lc.clamp(-6, 8))
            gs = rel[..., 5].clamp(-6, 0).exp()
            cn = canon[:, kk]
            guess.append((torch.where(cn[..., None], pu + sign * off, gu),
                          torch.where(cn[..., None], ccov, gc),
                          torch.where(cn, torch.full_like(gs, 0.5), gs)))
        sel = (DEPTH[cur] == d) & tree.gather(1, (2 * cur + 1).clamp_max(NN - 1))
        j = (cur - k0).clamp(0, k1 - k0 - 1)
        J2, J3 = j[..., None].expand(-1, -1, 2), j[..., None].expand(-1, -1, 3)
        (au, ac, ash), (bu, bc, bsh) = guess
        for _ in range(N_EM):
            la = loggauss(P - au.gather(1, J2), ac.gather(1, J3)) + ash.gather(1, j).log()
            lb = loggauss(P - bu.gather(1, J2), bc.gather(1, J3)) + bsh.gather(1, j).log()
            idx = torch.where(sel, 2 * cur + 1 + (lb > la).long(), cur)
            m, u, c = moments(F, idx)
            ma, mb = m[:, ka], m[:, kb]
            oka, okb = (ma >= M_MIN)[..., None], (mb >= M_MIN)[..., None]
            tot = (ma + mb).clamp_min(1e-6)
            au = torch.where(oka, u[:, ka], guess[0][0]); ac = torch.where(oka, c[:, ka], guess[0][1])
            bu = torch.where(okb, u[:, kb], guess[1][0]); bc = torch.where(okb, c[:, kb], guess[1][1])
            ash = torch.where(oka[..., 0], ma / tot, guess[0][2]).clamp_min(1e-3)
            bsh = torch.where(okb[..., 0], mb / tot, guess[1][2]).clamp_min(1e-3)
        cur = idx
        sp = split[..., None]
        mass[:, ka] = torch.where(split, ma.clamp_min(M_MIN), 0.)
        mass[:, kb] = torch.where(split, mb.clamp_min(M_MIN), 0.)
        mu[:, ka] = torch.where(sp, au, 0.); mu[:, kb] = torch.where(sp, bu, 0.)
        cov[:, ka] = torch.where(sp, ac, PIX_VAR); cov[:, kb] = torch.where(sp, bc, PIX_VAR)
    leaf = tree & ~has_kids(tree)
    massf = torch.where(tree, mass, 1.)
    covf = torch.where(tree[..., None], cov, torch.tensor([1., 0., 1.], device=DEV))
    lam1p, _ = eig(covf[:, PARENT[1:]])
    Lp = lam1p.clamp_min(1e-3).sqrt()
    dxy = (mu[:, 1:] - mu[:, PARENT[1:]]) / Lp[..., None]
    lg = logm(covf[:, 1:]); lg[..., 0] -= 2 * Lp.log(); lg[..., 2] -= 2 * Lp.log()
    lm = (massf[:, 1:] / massf[:, PARENT[1:]]).log()
    rel = torch.cat([torch.cat([mu[:, :1], logm(covf[:, :1]), massf[:, :1, None].log()], -1),
                     torch.cat([dxy, lg, lm[..., None]], -1)], 1)
    rel = torch.where(tree[..., None], rel, 0.)
    Rraw = render(massf, mu, covf, leaf, clamp=False)
    return dict(mass=massf, mu=mu, cov=covf, rel=rel, own=cur, leaf=leaf, R=Rraw.clamp(0, 1), Rraw=Rraw)


def explain(W, tree, init, canon, sub=256):
    outs = [_explain(W[i:i + sub], tree[i:i + sub], init[i:i + sub], canon[i:i + sub]) for i in range(0, len(W), sub)]
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def compose(rel, tree):
    """Walk down a tree of relative numbers to splats on the page; paint the leaves."""
    B = rel.shape[0]
    mass = torch.ones(B, NN, device=DEV); mu = torch.zeros(B, NN, 2, device=DEV)
    cov = PIX_VAR.expand(B, NN, 3).clone()
    mu[:, 0] = rel[:, 0, 0:2]; cov[:, 0] = expm(rel[:, 0, 2:5].clamp(-6, 8)); mass[:, 0] = rel[:, 0, 5].exp()
    for d in range(1, D_MAX + 1):
        k0, k1 = 2 ** d - 1, 2 ** (d + 1) - 1
        par = PARENT[k0:k1]
        lam1, _ = eig(cov[:, par]); L = lam1.clamp_min(1e-3).sqrt()
        mu[:, k0:k1] = mu[:, par] + L[..., None] * rel[:, k0:k1, 0:2]
        lc = rel[:, k0:k1, 2:5].clone(); lc[..., 0] += 2 * L.log(); lc[..., 2] += 2 * L.log()
        cov[:, k0:k1] = expm(lc.clamp(-6, 8))
        mass[:, k0:k1] = mass[:, par] * rel[:, k0:k1, 5].clamp(-8, 0).exp()
    leaf = tree & ~has_kids(tree)
    return render(mass, mu, cov, leaf), (mass, mu, cov, leaf)


# ----------------------------------------------------------------- the table
class Table:
    def __init__(self):
        z = lambda *s, **k: torch.zeros(*s, device=DEV, **k)
        self.C = 0
        self.active = z(CAP, dtype=torch.bool); self.tree = z(CAP, NN, dtype=torch.bool)
        self.n = z(CAP, NN); self.mean = z(CAP, NN, 6); self.M2 = z(CAP, NN, 6)
        self.counts = z(CAP, 10); self.uses = z(CAP)
        self.picks = z(CAP); self.ema = z(CAP); self.retired = 0
        self.tgain = z(CAP, NN); self.tcnt = z(CAP, NN)
        self.rend = z(CAP, 784)
        self.grown = 0; self.hired = 0; self.full = False

    def var(self, c):
        return (self.M2[c] + K0 * SIG0V) / (self.n[c] + K0)[..., None]

    def prior(self):
        u = self.uses[:self.C]
        return -torch.log((u + 1) / (u.sum() + self.C))

    def rerender(self, idx):
        idx = torch.unique(idx)
        self.rend[idx] = compose(self.mean[idx], self.tree[idx])[0]


T = None
SIG0V = None


def pose_cost(rel, tree, cidx):
    mean, n = T.mean[cidx], T.n[cidx]
    var = T.var(cidx)
    diff = torch.where((n > 0)[..., None], rel - mean, 0.)
    per = (diff ** 2 / (2 * var) + 0.5 * var.log() + C0).sum(-1)
    return per * tree


def root_pose(W):
    ex = _explain(W, torch.zeros(len(W), NN, dtype=torch.bool, device=DEV) | (torch.arange(NN, device=DEV) == 0),
                  torch.zeros(len(W), NN, 6, device=DEV), torch.zeros(len(W), NN, dtype=torch.bool, device=DEV))
    return ex["rel"][:, 0]


def shortlist(W, vis, S, rr=None):
    C = T.C
    R = T.rend[:C]
    d2 = (vis * W * W).sum(1, keepdim=True) - 2 * (vis * W) @ R.T + vis @ (R * R).T
    sc = d2 / (2 * SIG_PIX ** 2) + T.prior()[None]
    if rr is not None:
        var = T.var(torch.arange(C, device=DEV))[:, 0]
        diff = rr[:, None] - T.mean[:C, 0][None]
        sc = sc + (diff ** 2 / (2 * var[None]) + 0.5 * var.log()[None]).sum(-1)
    sc[:, ~T.active[:C]] = float("inf")
    k = min(S, int(T.active[:C].sum()))
    return sc.topk(k, 1, largest=False).indices


def seen_share(own, Wimg, vis):
    """Per node, the share of its ink that falls on visible pixels."""
    w = Wimg + 1e-3
    tot = torch.zeros(len(own), NN, device=DEV); sv = torch.zeros_like(tot)
    for d in range(D_MAX + 1):
        anc = ancestors(own, d); a = anc.clamp_min(0); ok = anc >= 0
        tot.scatter_add_(1, a, torch.where(ok, w, 0.)); sv.scatter_add_(1, a, torch.where(ok, w * vis, 0.))
    return sv / tot.clamp_min(1e-6)


def total_cost(ex, W, cidx, vis, Wimg=None):
    tree = T.tree[cidx].float()
    if Wimg is not None:                  # missing is not contradicting: hidden nodes cost nothing
        tree = tree * seen_share(ex["own"], Wimg, vis)
    return (T.prior()[cidx] + pose_cost(ex["rel"], tree, cidx).sum(1)
            + (vis * (W - ex["R"]) ** 2).sum(1) / (2 * SIG_PIX ** 2))


# ----------------------------------------------------------------- learning
def pix_gain(W, ex, R0, tree):
    """Ink drawn better by splitting each leaf: pixel error saved, per leaf of `tree`."""
    g = ((W - R0) ** 2 - (W - ex["R"]) ** 2) / (2 * SIG_PIX ** 2)
    own = ex["own"]
    own = torch.where(tree.gather(1, own), own, PARENT[own])
    return torch.zeros(len(W), NN, device=DEV).scatter_add_(1, own, g)


def ancestors(own, d):
    """The node at depth d above each pixel's owner (-1 where the owner is shallower)."""
    Dk = DEPTH[own]
    return torch.where(Dk >= d, ((own + 1) >> (Dk - d).clamp_min(0)) - 1, -1)


def hire(W, y):
    """Grow, then prune: split the image's ink all the way down, then keep a split
    only where the whole subtree under it draws more ink than its nodes cost."""
    B = len(W)
    if T.C + B > CAP:
        if not T.full: say("  the table is full; no more causes are hired")
        T.full = True
        return 0
    zero_init = torch.zeros(B, NN, 6, device=DEV)
    full = torch.ones(B, NN, dtype=torch.bool, device=DEV)
    ex = explain(W, full, zero_init, full)
    var_new = K0 * SIG0V / (1 + K0)
    node_cost = (0.5 * var_new.log() + C0).sum(-1)                  # a node at its own values
    as_leaf = torch.zeros(B, NN, device=DEV)                        # ink error if the node drew alone
    for d in range(D_MAX + 1):
        anc = ancestors(ex["own"], d)
        a = anc.clamp_min(0)
        r = (ex["mass"].gather(1, a) * loggauss(P - ex["mu"].gather(1, a[..., None].expand(-1, -1, 2)),
                                                 ex["cov"].gather(1, a[..., None].expand(-1, -1, 3))).exp()).clamp(0, 1)
        as_leaf.scatter_add_(1, a, torch.where(anc >= 0, (W - r) ** 2, 0.) / (2 * SIG_PIX ** 2))
    best = as_leaf.clone(); keep = torch.zeros(B, NN, dtype=torch.bool, device=DEV)
    for d in range(D_MAX - 1, -1, -1):
        k0, k1 = 2 ** d - 1, 2 ** (d + 1) - 1
        ka = torch.arange(2 * k0 + 1, 2 * k1, 2, device=DEV)
        split = best[:, ka] + best[:, ka + 1] + node_cost[ka] + node_cost[ka + 1]
        keep[:, k0:k1] = split < as_leaf[:, k0:k1]
        best[:, k0:k1] = torch.minimum(split, as_leaf[:, k0:k1])
    tr = torch.zeros(B, NN, dtype=torch.bool, device=DEV); tr[:, 0] = True
    for d in range(D_MAX):
        k0, k1 = 2 ** d - 1, 2 ** (d + 1) - 1
        ka = torch.arange(2 * k0 + 1, 2 * k1, 2, device=DEV)
        go = tr[:, k0:k1] & keep[:, k0:k1]
        tr[:, ka] = go; tr[:, ka + 1] = go
    ex = explain(W, tr, zero_init, tr)
    idx = torch.arange(T.C, T.C + B, device=DEV)
    T.tree[idx] = tr; T.n[idx] = tr.float(); T.mean[idx] = ex["rel"]; T.M2[idx] = 0
    T.counts[idx] = 0; T.counts[idx, y] = 1; T.uses[idx] = 1; T.active[idx] = True
    T.picks[idx] = 0; T.ema[idx] = 0; T.tgain[idx] = 0; T.tcnt[idx] = 0
    T.C += B; T.hired += B
    T.rerender(idx)
    return B


def update(W, y, c):
    tr = T.tree[c]
    leaf = tr & ~has_kids(tr)
    growable = leaf & (DEPTH < D_MAX)[None]
    ext = add_kids(tr, growable)
    trial = ext & ~tr
    canon = trial & (T.n[c] == 0)
    ex = explain(W, ext, T.mean[c], canon)
    R0 = render(ex["mass"], ex["mu"], ex["cov"], leaf)
    gain = pix_gain(W, ex, R0, tr)
    kc = pose_cost(ex["rel"], trial, c)
    kidcost = torch.zeros_like(gain); kidcost[:, :NK] = kc[:, KA] + kc[:, KB]
    net = torch.where(growable, gain - kidcost, 0.)
    rel = ex["rel"]
    for i in range(len(W)):
        ci = int(c[i]); upd = ext[i]
        n = T.n[ci] + upd.float()
        delta = rel[i] - T.mean[ci]
        mean = T.mean[ci] + torch.where(upd[:, None], delta / n.clamp_min(1)[:, None], 0.)
        T.M2[ci] += torch.where(upd[:, None], delta * (rel[i] - mean), 0.)
        T.mean[ci] = mean; T.n[ci] = n
        T.counts[ci, y[i]] += 1; T.uses[ci] += 1
        T.tgain[ci] += net[i]; T.tcnt[ci] += growable[i].float()
        now_leaf = T.tree[ci] & ~has_kids(T.tree[ci][None])[0]
        grow = now_leaf & growable[i] & (T.tcnt[ci] >= N_GROW) & (T.tgain[ci] > 0)
        if grow.any():
            T.tree[ci] = add_kids(T.tree[ci][None], grow[None])[0]
            T.tgain[ci, grow] = 0; T.tcnt[ci, grow] = 0
            T.grown += int(grow.sum())
    T.rerender(c)


def train(X, y, log_every=5000):
    global T
    T = Table()
    hire(X[:1], y[:1])
    right, seen, t0 = 0, 0, time.time()
    ones = torch.ones(1, 784, device=DEV)
    for s in range(1, len(X), CH):
        W, yy = X[s:s + CH], y[s:s + CH]
        B = len(W)
        cand = shortlist(W, ones, S_SHORT, root_pose(W))
        k = cand.shape[1]
        cp = cand.reshape(-1)
        Wp = W.repeat_interleave(k, 0)
        ex = explain(Wp, T.tree[cp], T.mean[cp], torch.zeros(len(cp), NN, dtype=torch.bool, device=DEV))
        cost = total_cost(ex, Wp, cp, ones).view(B, k)
        best = cand[torch.arange(B, device=DEV), cost.argmin(1)]
        ok = T.counts[best].argmax(1) == yy
        right += int(ok.sum()); seen += B
        for i in range(B):
            bi = int(best[i])
            T.picks[bi] += 1; T.ema[bi] = 0.9 * T.ema[bi] + 0.1 * float(~ok[i])
        if ok.any():
            update(W[ok], yy[ok], best[ok])
        if (~ok).any():
            hire(W[~ok], yy[~ok])
        bad = T.active[:T.C] & (T.picks[:T.C] >= 10) & (T.ema[:T.C] > 0.5)
        if bad.any():
            T.active[:T.C] &= ~bad; T.retired += int(bad.sum())
        if (s + B) // log_every != s // log_every:
            say(f"  {s + B:6d} images  right before learning {right / seen:.3f}  causes {int(T.active[:T.C].sum())} "
                f"(hired {T.hired}, retired {T.retired})  leaves grown {T.grown}  {time.time() - t0:.0f}s")
            right, seen = 0, 0
    return T


# ----------------------------------------------------------------- reading
ALPHA = 0.1


def read(X, vis=None, iters=0, chunk=128):
    """Label hidden. Returns answers (cheapest cause, blend over the shortlist, the
    fast path alone), the drawing of the cheapest cause and its index."""
    out = dict(pred=[], blend=[], fast=[], R=[], best=[], cost=[])
    for s in range(0, len(X), chunk):
        W = X[s:s + chunk]; B = len(W)
        v = torch.ones_like(W) if vis is None else vis[s:s + chunk]
        cand = shortlist(W * v, v, S_SHORT, root_pose(W) if vis is None else None)
        k = cand.shape[1]; cp = cand.reshape(-1)
        Wp, vp = W.repeat_interleave(k, 0), v.repeat_interleave(k, 0)
        none = torch.zeros(len(cp), NN, dtype=torch.bool, device=DEV)
        Wimg = None
        if iters == 0:
            ex = explain(Wp, T.tree[cp], T.mean[cp], none)
        else:
            R = T.rend[cp]
            for _ in range(iters):
                Wimg = Wp * vp + R * (1 - vp)
                ex = explain(Wimg, T.tree[cp], T.mean[cp], none)
                R = ex["Rraw"]                # imagined ink keeps its full weight
        cost = total_cost(ex, Wp, cp, vp, Wimg).view(B, k)
        a = cost.argmin(1); rows = torch.arange(B, device=DEV)
        best = cand[rows, a]
        pyc = (T.counts[cand] + ALPHA) / (T.counts[cand].sum(-1, keepdim=True) + 10 * ALPHA)
        w = torch.softmax(-cost, 1)
        out["pred"].append(T.counts[best].argmax(1))
        out["blend"].append((w[..., None] * pyc).sum(1).argmax(1))
        out["fast"].append(T.counts[cand[:, 0]].argmax(1))
        out["R"].append(ex["R"].view(B, k, 784)[rows, a])
        out["best"].append(best); out["cost"].append(cost[rows, a])
    return {k: torch.cat(v) for k, v in out.items()}


def acc(p, y):
    return float((p == y).float().mean())


# ----------------------------------------------------------------- starting spreads
def starting_spreads():
    W = Xtr[:2000]
    full = torch.ones(len(W), NN, dtype=torch.bool, device=DEV)
    ex = explain(W, full, torch.zeros(len(W), NN, 6, device=DEV), full)
    sig = torch.zeros(D_MAX + 1, 6, device=DEV)
    for d in range(D_MAX + 1):
        k0, k1 = 2 ** d - 1, 2 ** (d + 1) - 1
        sig[d] = ex["rel"][:, k0:k1].reshape(-1, 6).std(0)
    return (SIG0_FRAC * sig[DEPTH]) ** 2, sig


# ================================================================= run
if __name__ == "__main__":
    SIG0V, sig_all = starting_spreads()
    say("spread over all images, per depth (dx, dy, shape x3, ink share):")
    for d in range(D_MAX + 1):
        say(f"  depth {d}: " + " ".join(f"{x:.3f}" for x in sig_all[d].tolist()))

    metrics = dict(pilot={})
    say(f"\npilot: learn {PILOT_N}, choose on {PILOT_VAL}")
    for sp in SIG_PIX_GRID:
        SIG_PIX = sp
        t = time.time()
        train(Xtr[:PILOT_N], ytr[:PILOT_N], log_every=PILOT_N)
        r = read(Xva)
        a = acc(r["pred"], yva)
        metrics["pilot"][sp] = dict(acc=a, causes=int(T.active[:T.C].sum()), seconds=time.time() - t)
        say(f"SIG_PIX {sp}: choose-set accuracy {a:.4f}, causes {int(T.active[:T.C].sum())}, {time.time() - t:.0f}s")
    SIG_PIX = max(metrics["pilot"], key=lambda s: metrics["pilot"][s]["acc"])
    say(f"chosen SIG_PIX = {SIG_PIX}")

    say(f"\nfull run: learn {N_LEARN}")
    t = time.time()
    train(Xtr, ytr)
    train_s = time.time() - t
    act = T.active[:T.C]
    leaves = (T.tree[:T.C] & ~has_kids(T.tree[:T.C])).sum(1)[act].float()
    depth = (T.tree[:T.C] * DEPTH[None]).amax(1)[act]
    metrics["table"] = dict(causes=int(act.sum()), hired=T.hired, retired=T.retired, leaves_grown=T.grown,
                            leaves_median=float(leaves.median()), leaves_max=int(leaves.max()),
                            at_depth_cap=int((depth == D_MAX).sum()),
                            depth_hist={int(d): int((depth == d).sum()) for d in range(D_MAX + 1)},
                            numbers=int(T.tree[:T.C][act].sum()) * 12 + int(act.sum()) * 10,
                            train_seconds=train_s)
    say(json.dumps(metrics["table"]))
    torch.save({k: getattr(T, k)[:T.C].cpu() for k in ("active", "tree", "n", "mean", "M2", "counts", "uses")}, OUT / "table.pt")

    # ---- classification
    t = time.time()
    r = read(Xte)
    cls = dict(cheapest=acc(r["pred"], yte), blend=acc(r["blend"], yte), fast=acc(r["fast"], yte),
               seconds=time.time() - t)
    # nearest training image by pixels, the same 50k
    nn_pred = []
    for s in range(0, len(Xte), 1000):
        d = (Xte[s:s + 1000] ** 2).sum(1, keepdim=True) - 2 * Xte[s:s + 1000] @ Xtr.T + (Xtr ** 2).sum(1)[None]
        nn_pred.append(ytr[d.argmin(1)])
    cls["nearest_image"] = acc(torch.cat(nn_pred), yte)
    metrics["classify"] = cls
    say("classify:", json.dumps(cls))
    test_best = r["best"]

    # ---- drawing from the label
    torch.manual_seed(1)
    rows = ["typical, most-used cause", "drawn from the spreads", "drawn from the spreads"]
    gen = torch.zeros(len(rows), 10, 784)
    for yv in range(10):
        w = T.counts[:T.C, yv] * T.active[:T.C]
        c0 = int(w.argmax())
        gen[0, yv] = compose(T.mean[c0:c0 + 1], T.tree[c0:c0 + 1])[0][0].cpu()
        for rr in (1, 2):
            c = int(torch.multinomial(w, 1))
            sd = T.var(torch.tensor([c], device=DEV)).sqrt()
            rel = T.mean[c:c + 1] + sd * torch.randn_like(sd)
            gen[rr, yv] = compose(rel, T.tree[c:c + 1])[0][0].cpu()
    fig, ax = plt.subplots(len(rows), 10, figsize=(10, 3.4))
    for i in range(len(rows)):
        for yv in range(10):
            ax[i, yv].imshow(gen[i, yv].view(28, 28), cmap="gray_r", vmin=0, vmax=1); ax[i, yv].axis("off")
            if i == 0: ax[i, yv].set_title(str(yv))
        ax[i, 0].text(-4, 14, rows[i], ha="right", va="center", fontsize=8)
    plt.subplots_adjust(left=0.2, wspace=0.05, hspace=0.05)
    plt.savefig(OUT / "draw_from_label.png", dpi=110, bbox_inches="tight"); plt.close()

    # ---- the tree of a few test digits, coarse to fine
    sel = [int((yte == d).nonzero()[0]) for d in (2, 4, 7)]
    W = Xte[sel]; c = test_best[sel]
    ex = explain(W, T.tree[c], T.mean[c], torch.zeros(len(sel), NN, dtype=torch.bool, device=DEV))
    maxd = int((T.tree[c] * DEPTH[None]).amax())
    fig, ax = plt.subplots(len(sel), maxd + 2, figsize=(1.6 * (maxd + 2), 1.7 * len(sel)))
    for i in range(len(sel)):
        tr = T.tree[c[i]]
        for d in range(maxd + 1):
            a = ax[i, d]; a.imshow(W[i].view(28, 28).cpu(), cmap="gray_r", vmin=0, vmax=2); a.axis("off")
            show = tr & ((DEPTH == d) | ((DEPTH < d) & ~has_kids(tr[None])[0]))
            for k in show.nonzero().flatten().tolist():
                lam1, phi = eig(ex["cov"][i, k]); a_, b_, c_ = ex["cov"][i, k].tolist()
                lam2 = a_ + c_ - float(lam1)
                x0, y0 = (ex["mu"][i, k] + 13.5).tolist()
                a.add_patch(Ellipse((x0, y0), 4 * float(lam1) ** 0.5, 4 * max(lam2, 1e-3) ** 0.5,
                                    angle=math.degrees(float(phi)), fill=False, color="C3", lw=1))
            if i == 0: a.set_title(f"depth {d}", fontsize=8)
        ax[i, maxd + 1].imshow(ex["R"][i].view(28, 28).cpu(), cmap="gray_r", vmin=0, vmax=1); ax[i, maxd + 1].axis("off")
        if i == 0: ax[i, maxd + 1].set_title("its drawing", fontsize=8)
    plt.savefig(OUT / "trees.png", dpi=110, bbox_inches="tight"); plt.close()

    # ---- hidden parts: name the digit and fill it in
    g = torch.Generator(device="cpu").manual_seed(2)
    fill_idx = torch.randperm(len(Xte), generator=g)[:N_FILL].to(DEV)
    Xf, yf = Xte[fill_idx], yte[fill_idx]
    masks = {}
    v = torch.ones(N_FILL, 28, 28, device=DEV); v[:, 14:] = 0; masks["bottom half hidden"] = v.reshape(N_FILL, 784)
    v = torch.ones(N_FILL, 28, 28, device=DEV)
    for i in range(N_FILL):
        ink = (Xf[i] > 0.5).nonzero().flatten()
        p = int(ink[torch.randint(len(ink), (1,), generator=g)])
        r0, c0 = min(max(p // 28 - 6, 0), 16), min(max(p % 28 - 6, 0), 16)
        v[i, r0:r0 + 12, c0:c0 + 12] = 0
    masks["12x12 box over ink hidden"] = v.reshape(N_FILL, 784)
    mean_img = Xtr.mean(0)
    metrics["fill"] = {}
    for name, vis in masks.items():
        hid = 1 - vis
        rf = read(Xf, vis, iters=FILL_ITERS)
        fill = vis * Xf + hid * rf["R"]
        d = ((vis * Xf) ** 2).sum(1, keepdim=True) - 2 * (vis * Xf) @ Xtr.T + vis @ (Xtr ** 2).T
        nn_i = d.argmin(1)
        mse = lambda F_: float((((F_ - Xf) * hid) ** 2).sum() / hid.sum())
        m = dict(label_cheapest=acc(rf["pred"], yf), label_blend=acc(rf["blend"], yf),
                 label_nearest_image=acc(ytr[nn_i], yf), label_full_image=acc(r["pred"][fill_idx], yf),
                 mse_cause=mse(rf["R"]), mse_mean_image=mse(mean_img[None].expand_as(Xf)),
                 mse_nearest_image=mse(Xtr[nn_i]), mse_blank=mse(torch.zeros_like(Xf)))
        metrics["fill"][name] = m
        say(f"{name}:", json.dumps(m))
        ex8 = list(range(8))
        fig, ax = plt.subplots(4, 8, figsize=(8, 4.4))
        labels = ["hidden part grey", "filled by the cause", "truth", "nearest training image"]
        shown = [vis * Xf + hid * 0.25, fill, Xf, vis * Xf + hid * Xtr[nn_i]]
        for rr in range(4):
            for j in ex8:
                ax[rr, j].imshow(shown[rr][j].view(28, 28).cpu(), cmap="gray_r", vmin=0, vmax=1); ax[rr, j].axis("off")
                if rr == 1: ax[rr, j].set_title(f"says {int(rf['pred'][j])}", fontsize=7, pad=1)
            ax[rr, 0].text(-4, 14, labels[rr], ha="right", va="center", fontsize=8)
        plt.subplots_adjust(left=0.22, wspace=0.05, hspace=0.25)
        fn = "fill_bottom.png" if "bottom" in name else "fill_box.png"
        plt.savefig(OUT / fn, dpi=110, bbox_inches="tight"); plt.close()

    metrics["settings"] = dict(D_MAX=D_MAX, N_EM=N_EM, M_MIN=M_MIN, K0=K0, SIG0_FRAC=SIG0_FRAC, DELTA=DELTA,
                               N_GROW=N_GROW, S_SHORT=S_SHORT, CH=CH, SIG_PIX=SIG_PIX, FILL_ITERS=FILL_ITERS)
    metrics["seconds"] = time.time() - T0
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)
    say(f"done in {time.time() - T0:.0f}s")
