"""Both directions at once, from mismatch, with no labels.

image -> forward -> code -> backward -> image.  Every level of the forward stack and every
level of the backward stack is trained at the same time, by the mismatch between the two
networks' states at that level, and nothing else.  No labels anywhere in training.  The
graph is cut at every level boundary, so every update is local to its level.

Forward level l:   a_l = ReLU(conv_l(y_{l-1}));  y_l = maxpool(a_l).      y_0 = the image.
Backward level l:  a_hat_l = ReLU(unpool_l(spread(y_l)));  p_{l-1} = tconv_l(a_hat_l).
Top code:          c = k winners of 256 units over y_3 (competitive learning, with a
                   conscience so that every unit wins sometimes); p_3 = ReLU(V c).

The backward levels are always trained the same way: unpool_l toward a_l, tconv_l toward
y_{l-1}.  What differs is how the forward kernels learn:

    labels       backprop on the digit labels, then frozen (the reference from the other
                 experiments; the only arm that ever sees a label)
    random       frozen as initialised
    agree        conv_l moves y_l toward p_l, the backward prediction of y_l from y_{l+1}
                 (from the top code for l = 3).  Lavender's rule: each state copies its twin.
    reconstruct  conv_l moves y_l so that its own backward level reconstructs y_{l-1} better.
                 The error one level down reaches the forward synapses through the feedback.
    both         the two losses added.

Measured afterwards, everything frozen:
    readout      a linear classifier trained on y_3 with labels, test accuracy: does the
                 code the network settled on discriminate?  (a measurement, not training)
    recon        render from y_3 alone, named by the judge; pixel error
    concept      render from the class-mean y_3, ten digits
    code         cosine between y_3 and the y_3 recomputed from the render (floor: shuffled)
    alive        fraction of y_3 channels that vary over the test set; mean activity

Usage: python run.py     # ~4 min
"""
import argparse, json, struct, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data/mnist/digits/raw"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CH = (16, 32, 64)
ARMS = ("labels", "random", "agree", "reconstruct", "both")
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


# ---------------------------------------------------------------- networks

class Forward(nn.Module):
    def __init__(s):
        super().__init__()
        cins = (1,) + CH[:-1]
        s.convs = nn.ModuleList([nn.Conv2d(ci, co, 3, padding=1) for ci, co in zip(cins, CH)])
        s.head = nn.Linear(CH[-1] * 16, 10)

    def forward(s, x, cut=True):
        """cut: the input to every level is detached, so a level's loss reaches only its own kernel."""
        ys, As, h = [x], [], x
        for conv in s.convs:
            a = F.relu(conv(h.detach() if cut else h)); y = F.max_pool2d(a, 2)
            As.append(a); ys.append(y); h = y
        return ys, As

    def logits(s, x):
        return s.head(s(x, cut=False)[0][-1].flatten(1))


class Judge(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(1, 32, 5, padding=2); s.c2 = nn.Conv2d(32, 64, 5, padding=2)
        s.f1 = nn.Linear(64 * 8 * 8, 128); s.f2 = nn.Linear(128, 10)

    def forward(s, x):
        h = F.max_pool2d(F.relu(s.c1(x)), 2); h = F.max_pool2d(F.relu(s.c2(h)), 2)
        return s.f2(F.relu(s.f1(h.flatten(1))))


class Decoder(nn.Module):
    def __init__(s):
        super().__init__()
        cins = (1,) + CH[:-1]
        s.unpools = nn.ModuleList([nn.Conv2d(c, c, 3, padding=1) for c in CH])
        s.tconvs = nn.ModuleList([nn.ConvTranspose2d(co, ci, 3, padding=1) for ci, co in zip(cins, CH)])

    def unpool(s, l, y):
        return F.relu(s.unpools[l - 1](F.interpolate(y, scale_factor=2, mode="nearest")))

    def down(s, l, a_hat):
        out = s.tconvs[l - 1](a_hat)
        return F.relu(out) if l > 1 else out

    def predict(s, l, y):
        """Backward level l: from y_l predict y_{l-1}."""
        return s.down(l, s.unpool(l, y))

    def render(s, y3):
        h = y3
        for l in (3, 2, 1):
            h = s.predict(l, h)
        return h


class TopCode(nn.Module):
    """k winners of n units over the top map.  Competitive learning with a conscience; a linear
    backward map that predicts the top map from the code."""
    def __init__(s, d=CH[-1] * 16, n=256, k=16):
        super().__init__()
        s.W = nn.Parameter(F.normalize(torch.randn(n, d), dim=1)); s.k = k; s.n = n
        s.register_buffer("wins", torch.ones(n))
        s.V = nn.Linear(n, d)

    def code(s, y3):
        x = F.normalize(y3.flatten(1), dim=1)
        freq = s.wins / s.wins.sum()
        score = x @ s.W.t() + 0.3 * (1.0 / s.n - freq)            # conscience: rarely winning units get a lift
        top = score.topk(s.k, dim=1)
        c = torch.zeros_like(score).scatter(1, top.indices, (x @ s.W.t()).gather(1, top.indices).clamp_min(0))
        return c, top.indices, x

    def losses(s, y3):
        c, idx, x = s.code(y3.detach())
        with torch.no_grad():
            s.wins.index_add_(0, idx.flatten(), torch.ones(idx.numel(), device=dev))
        loss_w = ((s.W[idx] - x.unsqueeze(1)) ** 2).sum(-1).mean()      # winners move toward the input
        p3 = F.relu(s.V(c.detach())).view_as(y3)
        loss_v = F.mse_loss(p3, y3.detach())
        return loss_w + loss_v, p3.detach()

    @torch.no_grad()
    def renorm(s):
        s.W.data = F.normalize(s.W.data, dim=1)


# ---------------------------------------------------------------- training

def train_classifier(logits, params, Xtr, ytr, Xte, yte, epochs):
    opt = torch.optim.Adam(params, 1e-3)
    for ep in range(epochs):
        for b in batches(len(Xtr), 128):
            loss = F.cross_entropy(logits(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return sum((logits(Xte[b]).argmax(1) == yte[b]).sum().item() for b in batches(len(Xte), 1000, False)) / len(Xte)


def train_joint(arm, fwd, dec, topc, X, epochs):
    """One optimizer over everything, but every loss is cut so that it reaches one level's weights."""
    params = list(dec.parameters()) + (list(fwd.convs.parameters()) + list(topc.parameters()) if arm in ("agree", "reconstruct", "both") else [])
    opt = torch.optim.Adam(params, 1e-3)
    hist = []
    for ep in range(epochs):
        tot = torch.zeros(3, device=dev); nb = 0
        for b in batches(len(X), 128):
            x = X[b]
            if arm in ("labels", "random"):
                with torch.no_grad():
                    ys, As = fwd(x)
            else:
                ys, As = fwd(x)
            losses = []
            # backward levels, always the same: unpool toward the pre-pool map, tconv toward the map below
            preds = {}
            for l in (1, 2, 3):
                a_hat = dec.unpool(l, ys[l].detach())
                losses.append(F.mse_loss(a_hat, As[l - 1].detach()))
                p = dec.down(l, a_hat.detach())
                losses.append(F.mse_loss(p, ys[l - 1].detach()))
                preds[l - 1] = p.detach()                                  # p_{l-1}: the backward prediction of y_{l-1}
            if arm in ("agree", "both"):
                loss_top, p3 = topc.losses(ys[3]); losses.append(loss_top)
                preds[3] = p3
                for l in (1, 2, 3):                                        # y_l toward p_l
                    losses.append(F.mse_loss(ys[l], preds[l]))
            if arm in ("reconstruct", "both"):
                for l in (1, 2, 3):                                        # y_l so that its own level reconstructs y_{l-1}
                    losses.append(F.mse_loss(dec.predict(l, ys[l]), ys[l - 1].detach()))
            opt.zero_grad(); sum(losses).backward(); opt.step()
            if arm in ("agree", "both"): topc.renorm()
            tot += torch.stack([losses[1].detach(), losses[3].detach(), losses[5].detach()]); nb += 1
        hist.append((tot / nb).tolist())
        log(f"    ep {ep + 1}  mismatch  pixels {hist[-1][0]:.4f}  L1 {hist[-1][1]:.4f}  L2 {hist[-1][2]:.4f}")
    return hist


# ---------------------------------------------------------------- measurement

def normed(r):
    return r / r.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def top_maps(fwd, X):
    return torch.cat([fwd(X[b])[0][3] for b in batches(len(X), 1000, False)])


def evaluate(fwd, dec, judge, Xtr, ytr, Xte, yte):
    res, pics = {}, {}
    Ytr, Yte = top_maps(fwd, Xtr), top_maps(fwd, Xte)
    # discriminability: a linear readout trained afterwards, with labels, as a measurement
    torch.manual_seed(7); lin = nn.Linear(CH[-1] * 16, 10).to(dev)
    opt = torch.optim.Adam(lin.parameters(), 1e-3)
    for ep in range(5):
        for b in batches(len(Ytr), 256):
            loss = F.cross_entropy(lin(Ytr[b].flatten(1)), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        res["readout"] = (lin(Yte.flatten(1)).argmax(1) == yte).float().mean().item()
        # aliveness of the code
        flat = Yte.flatten(2)                                              # (N, 64, 16)
        res["alive"] = (flat.std(0).mean(1) > 1e-4).float().mean().item()
        res["activity"] = Yte.mean().item()
        # rendering from the top map alone
        c, cn, mse, cos, cosf = 0, 0, 0.0, 0.0, 0.0
        for b in batches(len(Xte), 500, False):
            r = dec.render(Yte[b]).clamp(0, 1)
            c += (judge(r).argmax(1) == yte[b]).sum().item(); cn += (judge(normed(r)).argmax(1) == yte[b]).sum().item()
            mse += F.mse_loss(r, Xte[b], reduction="sum").item() / 1024
            y_out = fwd(r)[0][3]
            cos += F.cosine_similarity(Yte[b].flatten(1), y_out.flatten(1), dim=1).sum().item()
            cosf += F.cosine_similarity(Yte[b].flatten(1), y_out.flatten(1).roll(1, 0), dim=1).sum().item()
            if b[0] == 0: pics["recon"] = r[:8].cpu()
        N = len(Xte)
        res.update(recon=c / N, recon_norm=cn / N, recon_mse=mse / N, code_cos=cos / N, code_cos_floor=cosf / N)
        means = torch.stack([Ytr[ytr == k].mean(0) for k in range(10)])
        rc = dec.render(means).clamp(0, 1); ok = torch.arange(10, device=dev)
        res["concept"] = (judge(rc).argmax(1) == ok).float().mean().item()
        res["concept_norm"] = (judge(normed(rc)).argmax(1) == ok).float().mean().item()
        pics["concept"] = rc.cpu()
        pics["kernels"] = fwd.convs[0].weight.detach().cpu()
    return res, pics


def figures(arms, pics, Xte):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(arms)
    fig, ax = plt.subplots(len(names) + 1, 8, figsize=(8, (len(names) + 1) * 1.05))
    for j in range(8):
        ax[0, j].imshow(Xte[j, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    ax[0, 0].set_ylabel("original", rotation=0, ha="right", va="center", fontsize=7)
    for i, a in enumerate(names):
        for j in range(8):
            ax[i + 1, j].imshow(pics[a]["recon"][j, 0], cmap="gray", vmin=0, vmax=1)
        ax[i + 1, 0].set_ylabel(a, rotation=0, ha="right", va="center", fontsize=7)
    for row in ax:
        for a_ in row: a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("render from the top map alone, per way of training the forward stack", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "renders.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(len(names), 10, figsize=(10, len(names) * 1.05))
    for i, a in enumerate(names):
        for j in range(10):
            ax[i, j].imshow(pics[a]["concept"][j, 0], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(a, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("concept renders from the class-mean top map", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "concepts.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(len(names), 16, figsize=(12, len(names) * 0.85))
    for i, a in enumerate(names):
        k = pics[a]["kernels"]; v = k.abs().max()
        for j in range(16):
            ax[i, j].imshow(k[j, 0], cmap="RdBu_r", vmin=-v, vmax=v); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(a, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("the sixteen level-1 kernels each way of training left behind", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "kernels.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for l, t in enumerate(["mismatch at the pixels", "mismatch at level 1", "mismatch at level 2"]):
        for a in names:
            ax[l].plot([h[l] for h in arms[a]["hist"]], marker="o", ms=3, label=a)
        ax[l].set_title(t); ax[l].set_xlabel("epoch"); ax[l].set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "curves.png", dpi=130); plt.close(fig)


def summary(arms, judge_acc):
    L = ["# Results", "", f"Judge on real test digits: {judge_acc:.4f}.  One seed, 8 epochs.", "",
         "readout = linear classifier trained on the frozen top map afterwards (the only place labels enter, as a measurement).",
         "recon = judge on the render from the top map alone, as rendered / brightness-normalised.  concept = judge on the ten class-mean renders.",
         "code = cosine between the top map and the top map recomputed from its render (floor: shuffled).  alive = fraction of top channels that vary.", "",
         "| forward trained by | readout | recon | recon mse | concept | code (floor) | alive | activity |",
         "|---|---|---|---|---|---|---|---|"]
    for a, r in arms.items():
        L.append(f"| {a} | {r['readout']:.4f} | {r['recon']:.3f} / {r['recon_norm']:.3f} | {r['recon_mse']:.4f} | {r['concept']:.1f} / {r['concept_norm']:.1f} "
                 f"| {r['code_cos']:.2f} ({r['code_cos_floor']:.2f}) | {r['alive']:.2f} | {r['activity']:.3f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n"); log("\n".join(L))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    Xtr, ytr = mnist("train"); Xte, yte = mnist("test")
    torch.manual_seed(a.seed + 100); judge = Judge().to(dev)
    judge_acc = train_classifier(judge, judge.parameters(), Xtr, ytr, Xte, yte, 3); judge.requires_grad_(False); judge.eval()
    log(f"judge {judge_acc:.4f} on real test digits  ({time.time() - t0:.0f}s)")
    arms, pics = {}, {}
    for arm in ARMS:
        log(f"{arm}:")
        torch.manual_seed(a.seed); fwd = Forward().to(dev)
        if arm == "labels":
            acc = train_classifier(fwd.logits, fwd.parameters(), Xtr, ytr, Xte, yte, 3); log(f"    forward by labels: {acc:.4f} on real test digits")
        if arm in ("labels", "random"):
            fwd.requires_grad_(False)
        torch.manual_seed(a.seed + 1); dec = Decoder().to(dev); topc = TopCode().to(dev)
        hist = train_joint(arm, fwd, dec, topc, Xtr, a.epochs)
        fwd.requires_grad_(False); dec.requires_grad_(False)
        res, p = evaluate(fwd, dec, judge, Xtr, ytr, Xte, yte)
        res["hist"] = hist; arms[arm] = res; pics[arm] = p
        log(f"    readout {res['readout']:.4f}  recon {res['recon_norm']:.3f}  concept {res['concept_norm']:.1f}  code {res['code_cos']:.2f} (floor {res['code_cos_floor']:.2f})  "
            f"alive {res['alive']:.2f}  activity {res['activity']:.3f}  ({time.time() - t0:.0f}s)")
    figures(arms, pics, Xte)
    summary(arms, judge_acc)
    json.dump({"judge_acc": judge_acc, "epochs": a.epochs, "seed": a.seed, "arms": arms}, open(OUT / "metrics.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
