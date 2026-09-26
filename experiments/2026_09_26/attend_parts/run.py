"""Attend to a part, render the part, sweep the parts: does attention buy vividness?

The one-shot render of what_where is the coarse image attended as a whole.  Here a window is
attended instead.  The what of a window is the top map of the forward stack run on the
window zoomed to the canonical 32x32.  The where is the window's position and size, the
pointer.  The render is the window's content drawn at 32x32, shrunk back to the window's
size and pasted at its place.  Sweep windows over the field and composite.

Window sets:
    whole       one window of 32: the one-shot render
    tiles16     four windows of 16, non-overlapping
    overlap16   nine windows of 16 at stride 8, overlaps averaged
    tiles8      sixteen windows of 8

Forward stack as before (conv3x3 + ReLU + max pool, three levels, trained on the labels,
frozen).  One decoder per window size, content alone, learned unpool (the none/learned arm
of what_where), trained by local mismatch on zoomed random windows of that size.  Scored by
pixel error to the original and by the judge on the composite.

Usage: python run.py     # ~1 min
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
SETS = {"whole": [(0, 0, 32)],
        "tiles16": [(t, l, 16) for t in (0, 16) for l in (0, 16)],
        "overlap16": [(t, l, 16) for t in (0, 8, 16) for l in (0, 8, 16)],
        "tiles8": [(t, l, 8) for t in range(0, 32, 8) for l in range(0, 32, 8)]}
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
        return ys, As

    def logits(s, x):
        return s.head(s(x)[0][-1].flatten(1))


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


class Decoder(nn.Module):
    """Content alone.  Per level: spread, a learned 3x3 unpool trained against the pre-pool map,
    then a transposed conv trained against the pooled map below.  All losses local."""
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

    def render(s, y3):
        h = y3
        for l in (3, 2, 1):
            h = s.down(l, s.unpool(l, h))
        return h


def zoom(x, t, l, w):
    """Crop a w by w window at (t, l) and resample it to 32x32: the attended part at canonical size."""
    c = x[:, :, t:t + w, l:l + w]
    return c if w == 32 else F.interpolate(c, size=(32, 32), mode="bilinear", align_corners=False)


def unzoom(r, w):
    return r if w == 32 else F.interpolate(r, size=(w, w), mode="area")


def train_decoder(dec, fwd, X, w, epochs):
    opt = torch.optim.Adam(dec.parameters(), 1e-3)
    for ep in range(epochs):
        tot = torch.zeros(3, device=dev); nb = 0
        for b in batches(len(X), 128):
            t, l = np.random.randint(0, 33 - w, size=2)
            with torch.no_grad():
                ys, As = fwd(zoom(X[b], t, l, w))
            losses = []
            for lv in (1, 2, 3):
                a_hat = dec.unpool(lv, ys[lv])
                losses.append(F.mse_loss(a_hat, As[lv - 1]))
                losses.append(F.mse_loss(dec.down(lv, a_hat.detach()), ys[lv - 1]))
            opt.zero_grad(); sum(losses).backward(); opt.step()
            tot += torch.stack([v.detach() for v in losses[1::2]]); nb += 1
    log(f"    decoder for windows of {w}: final mismatch pixels {tot[0] / nb:.4f}  L1 {tot[1] / nb:.4f}  L2 {tot[2] / nb:.4f}")


@torch.no_grad()
def composite(decs, fwd, x, windows):
    canvas = torch.zeros_like(x); count = torch.zeros_like(x)
    for t, l, w in windows:
        ys, _ = fwd(zoom(x, t, l, w))
        r = unzoom(decs[w].render(ys[3]).clamp(0, 1), w)
        canvas[:, :, t:t + w, l:l + w] += r; count[:, :, t:t + w, l:l + w] += 1
    return canvas / count


def normed(r):
    return r / r.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def evaluate(decs, fwd, judge, Xte, yte):
    res, pics = {}, {}
    for name, windows in SETS.items():
        mse, c, cn = 0.0, 0, 0
        for b in batches(len(Xte), 500, False):
            r = composite(decs, fwd, Xte[b], windows)
            mse += F.mse_loss(r, Xte[b], reduction="sum").item() / 1024
            c += (judge(r).argmax(1) == yte[b]).sum().item(); cn += (judge(normed(r)).argmax(1) == yte[b]).sum().item()
            if b[0] == 0: pics[name] = r[:8].cpu()
        res[name] = {"windows": len(windows), "mse": mse / len(Xte), "judge": c / len(Xte), "judge_norm": cn / len(Xte)}
        log(f"{name:10s} {len(windows):2d} windows   mse {res[name]['mse']:.4f}   judge {res[name]['judge']:.3f} / {res[name]['judge_norm']:.3f}")
    return res, pics


def figures(decs, fwd, pics, Xte):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("original", None)] + [(k, k) for k in SETS]
    fig, ax = plt.subplots(len(rows), 8, figsize=(8, len(rows) * 1.05))
    for i, (label, k) in enumerate(rows):
        imgs = Xte[:8].cpu() if k is None else pics[k]
        for j in range(8):
            ax[i, j].imshow(imgs[j, 0], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("the one-shot render (whole) against composites of attended windows", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "composites.png", dpi=130); plt.close(fig)

    # one digit, its four 16-windows: the zoomed part, its render at 32x32, the render shrunk back
    x = Xte[:1]
    fig, ax = plt.subplots(4, 3, figsize=(3.4, 4.6))
    with torch.no_grad():
        for i, (t, l, w) in enumerate(SETS["tiles16"]):
            z = zoom(x, t, l, w); ys, _ = fwd(z); r = decs[w].render(ys[3]).clamp(0, 1)
            for j, img in enumerate((z, r, unzoom(r, w))):
                ax[i, j].imshow(img[0, 0].cpu(), cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, 0].set_ylabel(f"window ({t},{l})", rotation=0, ha="right", va="center", fontsize=7)
    for j, t in enumerate(["attended, zoomed", "rendered", "shrunk back"]): ax[0, j].set_title(t, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "one_digit.png", dpi=130); plt.close(fig)

    # attend one part: the whole-image render everywhere, the attended window's render pasted over it
    from matplotlib.patches import Rectangle
    with torch.no_grad():
        x = Xte[:4]; base = composite(decs, fwd, x, SETS["whole"])
        fig, ax = plt.subplots(4, 6, figsize=(6.6, 4.6))
        for i in range(4):
            ax[i, 0].imshow(x[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            ax[i, 1].imshow(base[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
            for j, (t, l, w) in enumerate(SETS["tiles16"]):
                comp = base.clone()
                ys, _ = fwd(zoom(x, t, l, w))
                comp[:, :, t:t + w, l:l + w] = unzoom(decs[w].render(ys[3]).clamp(0, 1), w)
                ax[i, j + 2].imshow(comp[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
                ax[i, j + 2].add_patch(Rectangle((l - 0.5, t - 0.5), w, w, fill=False, edgecolor="tab:orange", lw=0.8))
            for j in range(6): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        for j, t in enumerate(["original", "whole, one look", "attend 1", "attend 2", "attend 3", "attend 4"]): ax[0, j].set_title(t, fontsize=7)
        fig.suptitle("one window attended (orange): sharp inside, the one-look render outside", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "attend_one.png", dpi=130); plt.close(fig)

        # the sweep: windows of 8 added one by one in raster order over the whole-image render
        x = Xte[:2]; base = composite(decs, fwd, x, SETS["whole"])
        steps = list(range(0, 17, 2))
        fig, ax = plt.subplots(2, len(steps), figsize=(len(steps) * 1.1, 2.5))
        for i in range(2):
            for j, n in enumerate(steps):
                comp = base[i:i + 1].clone()
                for (t, l, w) in SETS["tiles8"][:n]:
                    ys, _ = fwd(zoom(x[i:i + 1], t, l, w))
                    comp[:, :, t:t + w, l:l + w] = unzoom(decs[w].render(ys[3]).clamp(0, 1), w)
                ax[i, j].imshow(comp[0, 0].cpu(), cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
                if i == 0: ax[i, j].set_title(f"{n} of 16", fontsize=7)
        fig.suptitle("sweeping windows of 8 over the one-look render, raster order", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "sweep.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    np.random.seed(a.seed)
    Xtr, ytr = mnist("train"); Xte, yte = mnist("test")
    torch.manual_seed(a.seed + 100); judge = Judge().to(dev)
    judge_acc = train_classifier(judge, judge.parameters(), Xtr, ytr, Xte, yte, 3); judge.requires_grad_(False); judge.eval()
    torch.manual_seed(a.seed); fwd = Forward().to(dev)
    fwd_acc = train_classifier(fwd.logits, fwd.parameters(), Xtr, ytr, Xte, yte, 3); fwd.requires_grad_(False); fwd.eval()
    log(f"judge {judge_acc:.4f}, forward {fwd_acc:.4f} on real test digits  ({time.time() - t0:.0f}s)")
    decs = {}
    for w in (32, 16, 8):
        torch.manual_seed(a.seed + 1); decs[w] = Decoder().to(dev)
        train_decoder(decs[w], fwd, Xtr, w, a.epochs)
    log(f"decoders trained  ({time.time() - t0:.0f}s)")
    res, pics = evaluate(decs, fwd, judge, Xte, yte)
    figures(decs, fwd, pics, Xte)
    L = ["# Results", "", f"Judge {judge_acc:.4f}, forward {fwd_acc:.4f} on real test digits.  mse = pixel error of the composite to the original;",
         "judge = fraction of composites named correctly, as rendered / brightness-normalised.", "",
         "| window set | windows | window size | mse | judge |", "|---|---|---|---|---|"]
    for k, r in res.items():
        L.append(f"| {k} | {r['windows']} | {SETS[k][0][2]} | {r['mse']:.4f} | {r['judge']:.3f} / {r['judge_norm']:.3f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n"); log("\n".join(L))
    json.dump({"judge_acc": judge_acc, "forward_acc": fwd_acc, "epochs": a.epochs, "seed": a.seed, "sets": res}, open(OUT / "metrics.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
