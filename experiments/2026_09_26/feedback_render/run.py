"""A backward path that learns only from mismatch: does it render?

Forward network: three levels of conv3x3 + ReLU + 2x2 max pool, switches (the winner's
position in each pool) kept.  32x32 image -> 4x4x64 top map.

Backward network: three levels of unpool + transposed conv3x3.  Level l predicts the pooled
map of level l-1 (or the image) from the pooled map of level l.  Each level is trained only
by the mismatch between its prediction and the true activity below it, from the true
activity above it.  Nothing crosses a level.

Arms (two by two):
    forward   trained  backprop on the labels, then frozen
              random   frozen as initialised
    backward  tied     the forward kernels used in transpose, no training except one
                       scalar gain per level (three numbers)
              untied   its own kernels, learned with switches
              mixed    its own kernels, learned with switches and with the mass-preserving
                       spread (trained forward only): the no-position case is in training

Checks, all scored by a separate judge network trained on real digits.  Each render is judged
twice: as it is (clamped to [0, 1]) and after dividing by its own maximum, so that a faint but
correct render is not counted as wrong for being faint.
    recon         switches from the real forward pass; prediction feeds prediction downward
    imag_spread   no switches: each pooled value copied to all four positions of its pool
    imag_spread4  no switches: a quarter of each pooled value in all four positions (same mass)
    imag_corner   no switches: each pooled value put in the top-left position only
    concept_*     the class-mean top map rendered without switches, ten digits

Usage:
    python run.py                 # everything, ~5 min on a laptop GPU
    python run.py --epochs 3
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
    """conv + ReLU + max pool, three times.  Returns every pooled map and every switch map."""
    def __init__(s):
        super().__init__()
        cins = (1,) + CH[:-1]
        s.convs = nn.ModuleList([nn.Conv2d(ci, co, 3, padding=1) for ci, co in zip(cins, CH)])
        s.head = nn.Linear(CH[-1] * 16, 10)

    def forward(s, x):
        ys, ss, h = [x], [], x
        for conv in s.convs:
            h, idx = F.max_pool2d(F.relu(conv(h)), 2, return_indices=True)
            ys.append(h); ss.append(idx)
        return ys, ss                      # ys[0] = image, ys[1..3] = y1..y3; ss[0..2] = s1..s3

    def logits(s, x):
        return s.head(s(x)[0][-1].flatten(1))


class Judge(nn.Module):
    """A separate classifier, trained on real digits, that names the renders."""
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(1, 32, 5, padding=2); s.c2 = nn.Conv2d(32, 64, 5, padding=2)
        s.f1 = nn.Linear(64 * 8 * 8, 128); s.f2 = nn.Linear(128, 10)

    def forward(s, x):
        h = F.max_pool2d(F.relu(s.c1(x)), 2); h = F.max_pool2d(F.relu(s.c2(h)), 2)
        return s.f2(F.relu(s.f1(h.flatten(1))))


def unpool(y, idx, mode):
    if mode == "switch":
        return F.max_unpool2d(y, idx, 2)
    if mode == "spread":
        return F.interpolate(y, scale_factor=2, mode="nearest")
    if mode == "spread4":
        return F.interpolate(y, scale_factor=2, mode="nearest") / 4
    if mode == "corner":
        out = torch.zeros(y.shape[0], y.shape[1], y.shape[2] * 2, y.shape[3] * 2, device=y.device)
        out[:, :, ::2, ::2] = y
        return out
    raise ValueError(mode)


class Backward(nn.Module):
    """Level l: unpool the pooled map of level l, transposed conv, predict level l-1."""
    def __init__(s, fwd, tied):
        super().__init__()
        s.tied = tied; s.fwd = [fwd]       # in a list so the forward net is not registered here
        if tied:
            s.gain = nn.Parameter(torch.ones(3))
        else:
            cins = (1,) + CH[:-1]
            s.tconvs = nn.ModuleList([nn.ConvTranspose2d(co, ci, 3, padding=1) for ci, co in zip(cins, CH)])

    def level(s, l, y, idx=None, mode="switch"):
        a = unpool(y, idx, mode)
        if s.tied:
            out = F.conv_transpose2d(a, s.fwd[0].convs[l - 1].weight, padding=1) * s.gain[l - 1]
        else:
            out = s.tconvs[l - 1](a)
        return F.relu(out) if l > 1 else out            # hidden levels are ReLU maps; pixels are linear

    def render(s, y3, ss=None, mode="switch"):
        h = y3
        for l in (3, 2, 1):
            h = s.level(l, h, ss[l - 1] if ss is not None else None, mode)
        return h


# ---------------------------------------------------------------- training

def train_classifier(logits, params, Xtr, ytr, Xte, yte, epochs):
    opt = torch.optim.Adam(params, 1e-3)
    for ep in range(epochs):
        for b in batches(len(Xtr), 128):
            loss = F.cross_entropy(logits(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return sum((logits(Xte[b]).argmax(1) == yte[b]).sum().item() for b in batches(len(Xte), 1000, False)) / len(Xte)


def train_backward(bwd, fwd, X, epochs, modes=("switch",)):
    """Each level: its own loss, from the true map above to the true map below.  Local by construction.
    With several unpool modes, each level's loss is the mean over the modes."""
    opt = torch.optim.Adam(bwd.parameters(), 1e-3)
    hist = []
    for ep in range(epochs):
        tot = torch.zeros(3, device=dev); nb = 0
        for b in batches(len(X), 128):
            with torch.no_grad():
                ys, ss = fwd(X[b])
            losses = [sum(F.mse_loss(bwd.level(l, ys[l], ss[l - 1], m), ys[l - 1]) for m in modes) / len(modes) for l in (1, 2, 3)]
            opt.zero_grad(); sum(losses).backward(); opt.step()
            tot += torch.stack([v.detach() for v in losses]); nb += 1
        hist.append((tot / nb).tolist())
        log(f"    ep {ep + 1}  mismatch  L1 {hist[-1][0]:.4f}  L2 {hist[-1][1]:.4f}  L3 {hist[-1][2]:.4f}")
    return hist


@torch.no_grad()
def class_means(fwd, X, y):
    m = torch.zeros(10, CH[-1], 4, 4, device=dev); n = torch.zeros(10, device=dev)
    for b in batches(len(X), 1000, False):
        m.index_add_(0, y[b], fwd(X[b])[0][3]); n.index_add_(0, y[b], torch.ones(len(b), device=dev))
    return m / n.view(10, 1, 1, 1)


def normed(r):
    return r / r.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def evaluate(bwd, fwd, judge, Xte, yte, means):
    res, pics = {}, {}
    for name, mode, sw in [("recon", "switch", True), ("imag_spread", "spread", False), ("imag_spread4", "spread4", False), ("imag_corner", "corner", False)]:
        correct, correct_n, mse = 0, 0, 0.0
        for b in batches(len(Xte), 500, False):
            ys, ss = fwd(Xte[b])
            r = bwd.render(ys[3], ss if sw else None, mode).clamp(0, 1)
            correct += (judge(r).argmax(1) == yte[b]).sum().item()
            correct_n += (judge(normed(r)).argmax(1) == yte[b]).sum().item()
            mse += F.mse_loss(r, Xte[b], reduction="sum").item() / (32 * 32)
            if b[0] == 0: pics[name] = r[:8].cpu()
        res[name] = {"judge": correct / len(Xte), "judge_norm": correct_n / len(Xte), "mse": mse / len(Xte)}
    for mode in ("spread", "spread4", "corner"):
        r = bwd.render(means, None, mode).clamp(0, 1)
        ok = torch.arange(10, device=dev)
        res[f"concept_{mode}"] = {"judge": (judge(r).argmax(1) == ok).float().mean().item(),
                                  "judge_norm": (judge(normed(r)).argmax(1) == ok).float().mean().item()}
        pics[f"concept_{mode}"] = r.cpu()
    # per-level mismatch on the test set, each level from the true map above (as in training)
    tot = torch.zeros(3, device=dev); nb = 0
    for b in batches(len(Xte), 500, False):
        ys, ss = fwd(Xte[b])
        tot += torch.stack([F.mse_loss(bwd.level(l, ys[l], ss[l - 1]), ys[l - 1]) for l in (1, 2, 3)]); nb += 1
    res["mismatch_test"] = (tot / nb).tolist()
    return res, pics


# ---------------------------------------------------------------- figures and summary

def figures(arms, pics, originals):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(arms)
    rows = [("original", None, None)] + [(f"{a}\n{k}", a, k) for a in names for k in ("recon", "imag_spread", "imag_spread4")]
    fig, ax = plt.subplots(len(rows), 8, figsize=(8, len(rows) * 1.05))
    for i, (label, a, k) in enumerate(rows):
        imgs = originals if a is None else pics[a][k]
        for j in range(8):
            ax[i, j].imshow(imgs[j, 0], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("top row: the test digits.  then, per arm: reconstruction with switches, imagination (spread), imagination (spread, same mass)", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "renders.png", dpi=130); plt.close(fig)

    rows = [(f"{a}\n{m}", a, f"concept_{m}") for a in names for m in ("spread", "spread4")]
    fig, ax = plt.subplots(len(rows), 10, figsize=(10, len(rows) * 1.05))
    for i, (label, a, k) in enumerate(rows):
        for j in range(10):
            ax[i, j].imshow(pics[a][k][j, 0], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("concept renders: the class-mean top map, rendered without switches, digits 0 to 9", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "concepts.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for l in range(3):
        for a in names:
            ax[l].plot([h[l] for h in arms[a]["hist"]], marker="o", ms=3, label=a)
        ax[l].set_title(f"mismatch at level {l + 1}" if l else "mismatch at the pixels"); ax[l].set_xlabel("epoch"); ax[l].set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "curves.png", dpi=130); plt.close(fig)


def summary(arms, judge_acc, fwd_acc):
    L = ["# Results", "", f"Judge on real test digits: {judge_acc:.4f}.  Forward net (trained arm) on real test digits: {fwd_acc:.4f}.", "",
         "Each cell is the fraction of renders the judge names correctly, as rendered / after dividing each render by its own maximum.",
         "10,000 test digits for recon and imag; 10 class-mean renders for concept.  mse = pixel error of the reconstruction to the original.",
         "The corner mode (value in one fixed position of the pool) is in metrics.json; it was below 0.25 everywhere.", "",
         "| arm | recon | recon mse | imag spread | imag spread4 | concept spread | concept spread4 | test mismatch L1 / L2 / L3 |",
         "|---|---|---|---|---|---|---|---|"]
    def c(r): return f"{r['judge']:.3f} / {r['judge_norm']:.3f}"
    def k(r): return f"{r['judge']:.1f} / {r['judge_norm']:.1f}"
    for a, r in arms.items():
        m = r["mismatch_test"]
        L.append(f"| {a} | {c(r['recon'])} | {r['recon']['mse']:.4f} | {c(r['imag_spread'])} | {c(r['imag_spread4'])} "
                 f"| {k(r['concept_spread'])} | {k(r['concept_spread4'])} | {m[0]:.4f} / {m[1]:.4f} / {m[2]:.4f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    log("\n".join(L))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5, help="epochs for each backward network")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    Xtr, ytr = mnist("train"); Xte, yte = mnist("test")
    log(f"{len(Xtr)} training digits, {len(Xte)} test digits, {dev}")

    torch.manual_seed(a.seed + 100)
    judge = Judge().to(dev)
    judge_acc = train_classifier(judge, judge.parameters(), Xtr, ytr, Xte, yte, 3)
    judge.requires_grad_(False); judge.eval()
    log(f"judge: {judge_acc:.4f} on real test digits  ({time.time() - t0:.0f}s)")

    arms, pics, fwd_acc = {}, {}, None
    for fname in ("trained", "random"):
        torch.manual_seed(a.seed)
        fwd = Forward().to(dev)
        if fname == "trained":
            fwd_acc = train_classifier(fwd.logits, fwd.parameters(), Xtr, ytr, Xte, yte, 3)
            log(f"forward (trained): {fwd_acc:.4f} on real test digits  ({time.time() - t0:.0f}s)")
        fwd.requires_grad_(False); fwd.eval()
        means = class_means(fwd, Xtr, ytr)
        for bname in ("tied", "untied", "mixed"):
            if bname == "mixed" and fname != "trained": continue
            arm = f"{fname}_{bname}"
            log(f"{arm}:")
            torch.manual_seed(a.seed + 1)
            bwd = Backward(fwd, tied=(bname == "tied")).to(dev)
            hist = train_backward(bwd, fwd, Xtr, a.epochs, ("switch", "spread4") if bname == "mixed" else ("switch",))
            res, p = evaluate(bwd, fwd, judge, Xte, yte, means)
            res["hist"] = hist; arms[arm] = res; pics[arm] = p
            log(f"    raw/normed  recon {res['recon']['judge']:.3f}/{res['recon']['judge_norm']:.3f}  spread {res['imag_spread']['judge']:.3f}/{res['imag_spread']['judge_norm']:.3f}  "
                f"spread4 {res['imag_spread4']['judge']:.3f}/{res['imag_spread4']['judge_norm']:.3f}  concept spread {res['concept_spread']['judge']:.1f}/{res['concept_spread']['judge_norm']:.1f}  "
                f"spread4 {res['concept_spread4']['judge']:.1f}/{res['concept_spread4']['judge_norm']:.1f}  ({time.time() - t0:.0f}s)")

    figures(arms, pics, Xte[:8].cpu())
    summary(arms, judge_acc, fwd_acc)
    json.dump({"judge_acc": judge_acc, "forward_acc": fwd_acc, "epochs": a.epochs, "seed": a.seed, "arms": arms}, open(OUT / "metrics.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
