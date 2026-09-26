"""What and where as two codes: how coarse can the where be, and does the backward path
render from content plus a content-free pointer?

Forward network: as in feedback_render (conv3x3 + ReLU + 2x2 max pool, three levels), trained
on the labels and frozen.  It gives, per level, the pre-pool map a_l and the pooled map y_l.
The top map y3 is the what: a code with the positions inside every pool thrown away.

Where codes, all computed by summing across channels, so that none of them knows what is
where, only that something is:
    energy    per level, the pre-pool activity summed over channels, divided by its max
    energy3   the level-3 energy map (8x8) only, upsampled to the other levels
    switch1   per pool, the one position where the summed activity peaks (2 bits per pool)
    box       centre and extent of the level-1 energy, as a Gaussian bump at every level
    none      a map of ones: content alone

Backward network: per level, an unpool that takes the spread content and the where map, then
a transposed conv.  Two unpools:
    product   a_hat = spread(y) * where            (no parameters: content times gain)
    learned   a_hat = ReLU(conv3x3([spread(y), where]))  trained against the true pre-pool map
The transposed conv is trained against the true pooled map below, from a_hat detached.
Every loss is local to its level.

Checks (judge = separate classifier trained on real digits):
    recon        the digit's own what and where
    where_alone  the all-class-mean what, the digit's own where.  Should be at chance if
                 the where code is content-free.
    swap         the digit's own where (class c) with the class mean what of class c+1.
                 Does the judge say c+1 (content decides) or c (where decides)?
    concept      the class-mean what with the class-mean where, ten renders
    own          the forward stack's own head on the renders: does the network that imagined
                 the digit recognise its own imagination?  And the cosine between the what
                 that was rendered and the what recomputed from the render (against the
                 cosine to a shuffled digit's what as the floor).

The what is the top map y3 (4x4x64) by default.  With --what global it is the top map
max-pooled over its 4x4 grid to one 64-vector and broadcast back, so that no position at all
survives in the content and every position has to come from the where code.

Usage:  python run.py                  # what = the 4x4 top map, results in results/
        python run.py --what global    # what = one 64-vector, results in results_global/
"""
import argparse, json, struct, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--what", choices=("grid", "global"), default="grid")
ARGS = ap.parse_args()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data/mnist/digits/raw"
OUT = HERE / ("results" if ARGS.what == "grid" else "results_global"); OUT.mkdir(exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CH = (16, 32, 64)
KINDS = ("energy", "energy3", "switch1", "box", "none")
LOG = open(OUT / "run.log", "w")


def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def mnist(split):
    tag = "train" if split == "train" else "t10k"
    with open(DATA / f"{tag}-images-idx3-ubyte", "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), np.uint8).reshape(n, 1, r, c).astype(np.float32) / 255
    with open(DATA / f"{tag}-labels-idx1-ubyte", "rb") as f:
        f.read(8); y = np.frombuffer(f.read(), np.uint8).astype(np.int64)
    X = np.pad(X, ((0, 0), (0, 0), (2, 2), (2, 2)))
    return torch.tensor(X).to(dev), torch.tensor(y).to(dev)


def batches(n, bs, shuffle=True):
    idx = torch.randperm(n, device=dev) if shuffle else torch.arange(n, device=dev)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


# ---------------------------------------------------------------- forward, judge

class Forward(nn.Module):
    def __init__(s):
        super().__init__()
        cins = (1,) + CH[:-1]
        s.convs = nn.ModuleList([nn.Conv2d(ci, co, 3, padding=1) for ci, co in zip(cins, CH)])
        s.head = nn.Linear(CH[-1] * 16, 10)

    def forward(s, x):
        ys, As, h = [x], [], x
        for conv in s.convs:
            a = F.relu(conv(h)); h = F.max_pool2d(a, 2)
            As.append(a); ys.append(h)
        return ys, As                      # ys[0] = image, ys[l] = pooled map of level l; As[l-1] = pre-pool map of level l

    def logits(s, x):
        return s.head(s(x)[0][-1].flatten(1))


def top(ys):
    """The what.  The 4x4 top map, or its max over the grid broadcast back to 4x4."""
    y3 = ys[3]
    if ARGS.what == "global":
        return y3.amax(dim=(2, 3), keepdim=True).expand(-1, -1, 4, 4)
    return y3


class Judge(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(1, 32, 5, padding=2); s.c2 = nn.Conv2d(32, 64, 5, padding=2)
        s.f1 = nn.Linear(64 * 8 * 8, 128); s.f2 = nn.Linear(128, 10)

    def forward(s, x):
        h = F.max_pool2d(F.relu(s.c1(x)), 2); h = F.max_pool2d(F.relu(s.c2(h)), 2)
        return s.f2(F.relu(s.f1(h.flatten(1))))


def train_classifier(logits, params, Xtr, ytr, Xte, yte, epochs):
    opt = torch.optim.Adam(params, 1e-3)
    for ep in range(epochs):
        for b in batches(len(Xtr), 128):
            loss = F.cross_entropy(logits(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return sum((logits(Xte[b]).argmax(1) == yte[b]).sum().item() for b in batches(len(Xte), 1000, False)) / len(Xte)


# ---------------------------------------------------------------- where codes

def energy_maps(As):
    out = []
    for a in As:
        e = a.sum(1, keepdim=True)
        out.append(e / e.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))
    return out


def where_codes(kind, As):
    """Three maps, one per level, each (N, 1, 2H_l, 2W_l), the size of the pre-pool map."""
    if kind == "none":
        return [torch.ones(a.shape[0], 1, a.shape[2], a.shape[3], device=dev) for a in As]
    E = energy_maps(As)
    if kind == "energy":
        return E
    if kind == "energy3":
        return [F.interpolate(E[2], size=a.shape[2:], mode="bilinear", align_corners=False) for a in As]
    if kind == "switch1":
        out = []
        for a in As:
            e = a.sum(1, keepdim=True)
            _, idx = F.max_pool2d(e, 2, return_indices=True)
            out.append(F.max_unpool2d(torch.ones_like(idx, dtype=e.dtype), idx, 2))
        return out
    if kind == "box":
        e = E[0][:, 0]                                            # (N, 32, 32)
        yy, xx = torch.meshgrid(torch.arange(32., device=dev), torch.arange(32., device=dev), indexing="ij")
        w = e / e.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        cx = (w * xx).sum((1, 2)); cy = (w * yy).sum((1, 2))
        sx = (w * (xx - cx.view(-1, 1, 1)) ** 2).sum((1, 2)).sqrt().clamp_min(1.0)
        sy = (w * (yy - cy.view(-1, 1, 1)) ** 2).sum((1, 2)).sqrt().clamp_min(1.0)
        out = []
        for a in As:
            H = a.shape[2]; s = 32 / H
            c = torch.arange(H, device=dev) * s + (s - 1) / 2                # centres in level-1 pixel units
            gy, gx = torch.meshgrid(c, c, indexing="ij")
            bump = torch.exp(-((gx - cx.view(-1, 1, 1)) ** 2 / (2 * sx.view(-1, 1, 1) ** 2)
                               + (gy - cy.view(-1, 1, 1)) ** 2 / (2 * sy.view(-1, 1, 1) ** 2)))
            out.append(bump.unsqueeze(1))
        return out
    raise ValueError(kind)


# ---------------------------------------------------------------- backward

class Backward(nn.Module):
    def __init__(s, learned):
        super().__init__()
        s.learned = learned
        cins = (1,) + CH[:-1]
        s.tconvs = nn.ModuleList([nn.ConvTranspose2d(co, ci, 3, padding=1) for ci, co in zip(cins, CH)])
        if learned:
            s.unpools = nn.ModuleList([nn.Conv2d(c + 1, c, 3, padding=1) for c in CH])

    def unpool(s, l, y, w):
        sp = F.interpolate(y, scale_factor=2, mode="nearest")
        if s.learned:
            return F.relu(s.unpools[l - 1](torch.cat([sp, w], 1)))
        return sp * w

    def down(s, l, a_hat):
        out = s.tconvs[l - 1](a_hat)
        return F.relu(out) if l > 1 else out

    def render(s, y3, ws):
        h = y3
        for l in (3, 2, 1):
            h = s.down(l, s.unpool(l, h, ws[l - 1]))
        return h


def train_backward(bwd, fwd, kind, X, epochs):
    opt = torch.optim.Adam(bwd.parameters(), 1e-3)
    hist = []
    for ep in range(epochs):
        tot = torch.zeros(3, device=dev); nb = 0
        for b in batches(len(X), 128):
            with torch.no_grad():
                ys, As = fwd(X[b]); ws = where_codes(kind, As)
            losses = []
            for l in (1, 2, 3):
                a_hat = bwd.unpool(l, top(ys) if l == 3 else ys[l], ws[l - 1])
                if bwd.learned:
                    losses.append(F.mse_loss(a_hat, As[l - 1]))
                losses.append(F.mse_loss(bwd.down(l, a_hat.detach()), ys[l - 1]))
            opt.zero_grad(); sum(losses).backward(); opt.step()
            down_losses = losses[1::2] if bwd.learned else losses
            tot += torch.stack([v.detach() for v in down_losses]); nb += 1
        hist.append((tot / nb).tolist())
    log(f"    final mismatch  pixels {hist[-1][0]:.4f}  L1 {hist[-1][1]:.4f}  L2 {hist[-1][2]:.4f}")
    return hist


# ---------------------------------------------------------------- means and evaluation

@torch.no_grad()
def class_means(fwd, X, y):
    """Per class: mean top map, and mean where map per kind and level.  Also the all-class mean top map."""
    m3 = torch.zeros(10, CH[-1], 4, 4, device=dev); n = torch.zeros(10, device=dev)
    mw = {k: [torch.zeros(10, 1, 32 // 2 ** i, 32 // 2 ** i, device=dev) for i in range(3)] for k in KINDS}
    for b in batches(len(X), 1000, False):
        ys, As = fwd(X[b]); one = torch.ones(len(b), device=dev)
        m3.index_add_(0, y[b], top(ys)); n.index_add_(0, y[b], one)
        for k in KINDS:
            for i, w in enumerate(where_codes(k, As)):
                mw[k][i].index_add_(0, y[b], w)
    n = n.view(10, 1, 1, 1)
    return m3 / n, {k: [w / n for w in ws] for k, ws in mw.items()}, (m3.sum(0) / n.sum()).unsqueeze(0)


def normed(r):
    return r / r.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def evaluate(bwd, fwd, kind, judge, Xte, yte, m3, mw, mall):
    res, pics = {}, {}
    cnt = {"recon": 0, "recon_n": 0, "where_alone": 0, "swap_content": 0, "swap_where": 0, "own": 0}; mse = 0.0
    cos, cos_floor = 0.0, 0.0
    for b in batches(len(Xte), 500, False):
        ys, As = fwd(Xte[b]); ws = where_codes(kind, As); yb = yte[b]
        what_in = top(ys)
        r = bwd.render(what_in, ws).clamp(0, 1)
        ys_r, _ = fwd(r); what_out = top(ys_r)
        cnt["own"] += (fwd.head(ys_r[3].flatten(1)).argmax(1) == yb).sum().item()
        cos += F.cosine_similarity(what_in.flatten(1), what_out.flatten(1), dim=1).sum().item()
        cos_floor += F.cosine_similarity(what_in.flatten(1), what_out.flatten(1).roll(1, 0), dim=1).sum().item()
        cnt["recon"] += (judge(r).argmax(1) == yb).sum().item()
        cnt["recon_n"] += (judge(normed(r)).argmax(1) == yb).sum().item()
        mse += F.mse_loss(r, Xte[b], reduction="sum").item() / 1024
        ra = bwd.render(mall.expand(len(b), -1, -1, -1), ws).clamp(0, 1)
        cnt["where_alone"] += (judge(normed(ra)).argmax(1) == yb).sum().item()
        other = (yb + 1) % 10
        rs = bwd.render(m3[other], ws).clamp(0, 1)
        pred = judge(normed(rs)).argmax(1)
        cnt["swap_content"] += (pred == other).sum().item(); cnt["swap_where"] += (pred == yb).sum().item()
        if b[0] == 0:
            pics["recon"] = r[:8].cpu(); pics["where_alone"] = ra[:8].cpu(); pics["swap"] = rs[:8].cpu()
    N = len(Xte)
    res.update(recon=cnt["recon"] / N, recon_norm=cnt["recon_n"] / N, recon_mse=mse / N, where_alone=cnt["where_alone"] / N,
               swap_content=cnt["swap_content"] / N, swap_where=cnt["swap_where"] / N,
               own_recon=cnt["own"] / N, code_cos=cos / N, code_cos_floor=cos_floor / N)
    rc = bwd.render(m3, mw).clamp(0, 1); ok = torch.arange(10, device=dev)
    res["concept"] = (judge(rc).argmax(1) == ok).float().mean().item()
    res["concept_norm"] = (judge(normed(rc)).argmax(1) == ok).float().mean().item()
    ys_c, _ = fwd(rc)
    res["own_concept"] = (fwd.head(ys_c[3].flatten(1)).argmax(1) == ok).float().mean().item()
    res["concept_code_cos"] = F.cosine_similarity(m3.flatten(1), top(ys_c).flatten(1), dim=1).mean().item()
    pics["concept"] = rc.cpu()
    return res, pics


# ---------------------------------------------------------------- figures, summary

def figures(arms, pics, Xte, fwd):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        _, As = fwd(Xte[:1])
    fig, ax = plt.subplots(len(KINDS), 4, figsize=(4.4, len(KINDS) * 1.1))
    for i, k in enumerate(KINDS):
        ws = where_codes(k, As)
        ax[i, 0].imshow(Xte[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        for j in range(3):
            ax[i, j + 1].imshow(ws[j][0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        ax[i, 0].set_ylabel(k, rotation=0, ha="right", va="center", fontsize=7)
        for j in range(4): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    for j, t in enumerate(["digit", "where, level 1", "level 2", "level 3"]): ax[0, j].set_title(t, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "where.png", dpi=130); plt.close(fig)

    names = list(arms)
    rows = [("original", None)] + [(a, a) for a in names]
    fig, ax = plt.subplots(len(rows), 8, figsize=(8, len(rows) * 1.05))
    for i, (label, a) in enumerate(rows):
        imgs = Xte[:8].cpu() if a is None else pics[a]["recon"]
        for j in range(8):
            ax[i, j].imshow(imgs[j, 0], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("reconstruction: the digit's own what and where", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "recon.png", dpi=130); plt.close(fig)

    rows = [(f"{a}\n{c}", a, c) for a in names if a.endswith("learned") for c in ("where_alone", "swap")]
    fig, ax = plt.subplots(len(rows), 8, figsize=(8, len(rows) * 1.05))
    for i, (label, a, c) in enumerate(rows):
        for j in range(8):
            ax[i, j].imshow(pics[a][c][j, 0], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("controls, learned unpool.  where_alone: mean what + own where.  swap: own where + the what of the next class", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "controls.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(len(names), 10, figsize=(10, len(names) * 1.05))
    for i, a in enumerate(names):
        for j in range(10):
            ax[i, j].imshow(pics[a]["concept"][j, 0], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(a, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("concept renders: class-mean what with class-mean where", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "concepts.png", dpi=130); plt.close(fig)


def summary(arms, judge_acc, fwd_acc):
    L = ["# Results", "", f"what = {ARGS.what}.  Judge on real test digits: {judge_acc:.4f}.  Forward net on real test digits: {fwd_acc:.4f}.", "",
         "Fractions named correctly by the judge over the 10,000 test digits (concept: over 10 class renders).",
         "recon is given as rendered / brightness-normalised; the controls and concept use the normalised render.",
         "swap: the digit's own where with the what of the next class; 'content' = judge said the next class, 'where' = judge said the digit's own class.", "",
         "own = the forward stack's own head on the render (recon / concept).  code cos = cosine between the what rendered and the what",
         "recomputed from the render, recon (floor: cosine to a shuffled digit's what) / concept.", "",
         "| where code | unpool | recon | recon mse | where alone | swap: content / where | concept | own | code cos |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a, r in arms.items():
        k, u = a.rsplit("_", 1)
        L.append(f"| {k} | {u} | {r['recon']:.3f} / {r['recon_norm']:.3f} | {r['recon_mse']:.4f} | {r['where_alone']:.3f} "
                 f"| {r['swap_content']:.3f} / {r['swap_where']:.3f} | {r['concept']:.1f} / {r['concept_norm']:.1f} "
                 f"| {r['own_recon']:.3f} / {r['own_concept']:.1f} | {r['code_cos']:.2f} ({r['code_cos_floor']:.2f}) / {r['concept_code_cos']:.2f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    log("\n".join(L))


# ---------------------------------------------------------------- main

def main():
    a = ARGS
    t0 = time.time()
    Xtr, ytr = mnist("train"); Xte, yte = mnist("test")
    log(f"{len(Xtr)} training digits, {len(Xte)} test digits, {dev}, what = {a.what}")

    torch.manual_seed(a.seed + 100)
    judge = Judge().to(dev)
    judge_acc = train_classifier(judge, judge.parameters(), Xtr, ytr, Xte, yte, 3)
    judge.requires_grad_(False); judge.eval()
    torch.manual_seed(a.seed)
    fwd = Forward().to(dev)
    fwd_acc = train_classifier(fwd.logits, fwd.parameters(), Xtr, ytr, Xte, yte, 3)
    fwd.requires_grad_(False); fwd.eval()
    log(f"judge {judge_acc:.4f}, forward {fwd_acc:.4f} on real test digits  ({time.time() - t0:.0f}s)")
    m3, mw, mall = class_means(fwd, Xtr, ytr)

    arms, pics = {}, {}
    for kind in KINDS:
        for uname in ("product", "learned"):
            arm = f"{kind}_{uname}"
            log(f"{arm}:")
            torch.manual_seed(a.seed + 1)
            bwd = Backward(learned=(uname == "learned")).to(dev)
            hist = train_backward(bwd, fwd, kind, Xtr, a.epochs)
            res, p = evaluate(bwd, fwd, kind, judge, Xte, yte, m3, mw[kind], mall)
            res["hist"] = hist; arms[arm] = res; pics[arm] = p
            log(f"    judge: recon {res['recon_norm']:.3f}  where alone {res['where_alone']:.3f}  swap {res['swap_content']:.3f}/{res['swap_where']:.3f}  concept {res['concept_norm']:.1f}   "
                f"own: recon {res['own_recon']:.3f}  concept {res['own_concept']:.1f}   code cos {res['code_cos']:.2f} (floor {res['code_cos_floor']:.2f}) concept {res['concept_code_cos']:.2f}  ({time.time() - t0:.0f}s)")

    figures(arms, pics, Xte, fwd)
    summary(arms, judge_acc, fwd_acc)
    json.dump({"what": a.what, "judge_acc": judge_acc, "forward_acc": fwd_acc, "epochs": a.epochs, "seed": a.seed, "arms": arms}, open(OUT / "metrics.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
