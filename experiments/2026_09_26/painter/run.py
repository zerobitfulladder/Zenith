"""The painter: draw parts from memory, look back at the whole, keep what fits, go round again.

Lavender's description of imagining a face: a goal in the head, a coarse image from it, then
parts drawn sharp one at a time from what one knows a part looks like, each checked by looking
back at the whole (does it still read as this face?) and at the part (does it read as that
part?), redrawn where off, in more than one pass.

Pieces (all from today): the forward stack and whole-face decoder from ../faces, the 40x32
window decoder from ../sample_windows.  Windows of 40x32 on a 7x7 grid (stride 20x16),
feathered.

Library.  For 1,000 training faces and every grid position: the real window's code (the
part) and the code of the same window of the face's coarse render (the key).  A part is
looked up by the key: the blurry region of what is on the canvas now.

Painting one face, given a goal code:
    canvas = the coarse render of the goal
    for each sweep, for each window in raster order:
        key        = code of the window of the current canvas, zoomed
        candidates = the k parts at this position whose keys are nearest
        for each candidate: paste its render, read the whole canvas back, score
        keep the candidate with the best score
Scores:
    coarse   agreement (cosine) between the goal and the read-back of the whole canvas
    both     the same plus agreement between the candidate and the read-back of its window

Arms:
    blur        the coarse render, nothing painted
    first       the nearest part, no look-back, one sweep         (retrieval)
    random      a random one of the k candidates, one sweep       (control for the choice)
    coarse1     chosen by the coarse look-back, one sweep
    coarse2     the same, two sweeps
    both2       chosen by both look-backs, two sweeps
    from_real   the real face's own parts (the ceiling; not available for an imagined face)

Goals: the codes of 300 test faces (pixel error to the real face can be measured) and eight
imagined faces (pca64 samples).  Scored by agreement of the final canvas with the goal, pixel
error, sharpness (mean absolute Laplacian), and the fine agreement of the chosen parts.

Usage: python run.py     # ~6 min
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data/celeba"
NETS = HERE.parent / "faces/results/nets.pt"
WIN40 = HERE.parent / "sample_windows/results/win40.pt"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
dev = "cuda"
torch.backends.cudnn.benchmark = True
H, W = 160, 128
CH = (16, 32, 64, 128, 256)
D = CH[-1] * 20
MEAN, STD = 0.45, 0.25
WH, WW = 40, 32
POS = [(t, l) for t in range(0, H - WH + 1, 20) for l in range(0, W - WW + 1, 16)]     # 7 x 7 = 49
LOG = open(OUT / "run.log", "w")


def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


raw = np.load(DATA / "gray_218x178.npy", mmap_mode="r")
imgs = torch.from_numpy(np.ascontiguousarray(raw[:, 29:189, 25:153])).to(dev)
N = len(imgs); idx_test = torch.arange(N - 5000, N, device=dev); idx_train = torch.arange(0, N - 7000, device=dev)


def images(idx):
    return imgs[idx].float().div(255).sub(MEAN).div(STD).unsqueeze(1)


def show(x):
    return (x * STD + MEAN).clamp(0, 1)


def batches(idx, bs, shuffle=True):
    order = idx[torch.randperm(len(idx), device=dev)] if shuffle else idx
    for i in range(0, len(order), bs):
        yield order[i:i + bs]


def zoom(x, t, l):
    return F.interpolate(x[:, :, t:t + WH, l:l + WW], size=(H, W), mode="bilinear", align_corners=False)


def unzoom(r):
    return F.interpolate(r, size=(WH, WW), mode="area")


class Forward(nn.Module):
    def __init__(s):
        super().__init__()
        cins = (1,) + CH[:-1]
        s.convs = nn.ModuleList([nn.Conv2d(ci, co, 3, padding=1) for ci, co in zip(cins, CH)])
        s.head = nn.Linear(D, 40)

    def top(s, x):
        h = x
        for conv in s.convs:
            h = F.max_pool2d(F.relu(conv(h)), 2)
        return h


class Decoder(nn.Module):
    def __init__(s):
        super().__init__()
        cins = (1,) + CH[:-1]
        s.unpools = nn.ModuleList([nn.Conv2d(c, c, 3, padding=1) for c in CH])
        s.tconvs = nn.ModuleList([nn.ConvTranspose2d(co, ci, 3, padding=1) for ci, co in zip(cins, CH)])

    def render(s, y):
        for l in range(len(CH), 0, -1):
            a = F.relu(s.unpools[l - 1](F.interpolate(y, scale_factor=2, mode="nearest")))
            y = s.tconvs[l - 1](a)
            if l > 1: y = F.relu(y)
        return y


def ramp(n):
    i = torch.arange(n, device=dev, dtype=torch.float32)
    return 0.5 - 0.5 * torch.cos(2 * np.pi * (i + 0.5) / n)


FEATHER = (ramp(WH)[:, None] * ramp(WW)[None, :] + 1e-3).view(1, 1, WH, WW)
LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=dev).view(1, 1, 3, 3)
W_BG = 0.05


def sharpness(x):
    return F.conv2d(x, LAP).abs().mean((1, 2, 3))


def cos(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


class Canvas:
    """The coarse render underneath, painted windows on top, feathered.  Windows can be replaced."""
    def __init__(s, blur):
        s.blur = blur; s.parts = {}; s.num = W_BG * blur.clone(); s.den = torch.full_like(blur, W_BG)

    def _add(s, p, r, sign):
        t, l = POS[p]
        s.num[:, :, t:t + WH, l:l + WW] += sign * FEATHER * r; s.den[:, :, t:t + WH, l:l + WW] += sign * FEATHER

    def set(s, p, r):
        if p in s.parts: s._add(p, s.parts[p], -1)
        s.parts[p] = r; s._add(p, r, +1)

    def image(s):
        return s.num / s.den

    def image_with(s, p, r):
        """The canvas as it would be with r at window p, without committing."""
        t, l = POS[p]
        num = s.num.clone(); den = s.den.clone()
        if p in s.parts:
            num[:, :, t:t + WH, l:l + WW] -= FEATHER * s.parts[p]; den[:, :, t:t + WH, l:l + WW] -= FEATHER
        num[:, :, t:t + WH, l:l + WW] += FEATHER * r; den[:, :, t:t + WH, l:l + WW] += FEATHER
        return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_lib", type=int, default=1000); ap.add_argument("--k", type=int, default=8); ap.add_argument("--n_eval", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    st = torch.load(NETS, map_location=dev)
    fwd, whole, win = Forward().to(dev), Decoder().to(dev), Decoder().to(dev)
    fwd.load_state_dict(st["fwd"]); whole.load_state_dict(st["whole"]); win.load_state_dict(torch.load(WIN40, map_location=dev))
    for m in (fwd, whole, win): m.requires_grad_(False); m.eval()

    # ------------------------------------------------ the library: parts and keys at every position
    with torch.no_grad():
        lib = idx_train[:a.n_lib]
        T = torch.zeros(len(POS), a.n_lib, D, device=dev, dtype=torch.float16); K = torch.zeros_like(T)
        for i, b in enumerate(batches(lib, 250, False)):
            x = images(b); blur = whole.render(fwd.top(x)); sl = slice(i * 250, i * 250 + len(b))
            for p, (t_, l_) in enumerate(POS):
                T[p, sl] = fwd.top(zoom(x, t_, l_)).flatten(1).half()
                K[p, sl] = F.normalize(fwd.top(zoom(blur, t_, l_)).flatten(1), dim=1).half()
        log(f"library: {a.n_lib} faces x {len(POS)} positions  ({time.time() - t0:.0f}s)")

        def candidates(key, p, k):
            sims = F.normalize(key.flatten(1), dim=1).half() @ K[p].T                # (B, n_lib)
            idx = sims.topk(k, dim=1).indices                                         # (B, k)
            return T[p][idx].float().view(key.shape[0], k, CH[-1], 5, 4)

        def paint(goal, arm, sweeps, stages=None):
            B = goal.shape[0]; blur = whole.render(goal); cv = Canvas(blur); fine_sum = 0.0; n_fine = 0; snaps = {}
            for sw in range(sweeps):
                for p, (t_, l_) in enumerate(POS):
                    key = fwd.top(zoom(cv.image(), t_, l_))
                    cands = candidates(key, p, a.k)                                    # (B, k, 256, 5, 4)
                    if arm == "first": j = torch.zeros(B, dtype=torch.long, device=dev)
                    elif arm == "random": j = torch.randint(0, a.k, (B,), device=dev)
                    else:
                        scores = []
                        for jj in range(a.k):
                            r = unzoom(win.render(cands[:, jj])); img = cv.image_with(p, r)
                            sc = cos(fwd.top(img), goal)
                            if arm == "both": sc = sc + cos(fwd.top(zoom(img, t_, l_)), cands[:, jj])
                            scores.append(sc)
                        j = torch.stack(scores, 1).argmax(1)
                    chosen = cands[torch.arange(B, device=dev), j]
                    r = unzoom(win.render(chosen)); cv.set(p, r)
                    fine_sum += cos(fwd.top(zoom(cv.image(), t_, l_)), chosen).sum().item(); n_fine += B
                    if stages is not None and (sw, p) in stages: snaps[(sw, p)] = show(cv.image()[:2]).cpu()
            return cv.image(), fine_sum / n_fine, snaps

        def paint_real(x):
            cv = Canvas(whole.render(fwd.top(x)))
            for p, (t_, l_) in enumerate(POS): cv.set(p, unzoom(win.render(fwd.top(zoom(x, t_, l_)))))
            return cv.image()

        arms = {"first": ("first", 1), "random": ("random", 1), "coarse1": ("coarse", 1), "coarse2": ("coarse", 2), "both2": ("both", 2)}
        res, pics = {}, {}
        ev = idx_test[:a.n_eval]
        acc = {k: {"agree": 0.0, "err": 0.0, "sharp": 0.0, "fine": 0.0} for k in list(arms) + ["blur", "from_real"]}
        stages = {(0, 6), (0, 20), (0, 34), (0, 48), (1, 48)}
        for b in batches(ev, 100, False):
            x = images(b); goal = fwd.top(x); first = b[0] == ev[0]
            outs = {"blur": (whole.render(goal), None), "from_real": (paint_real(x), None)}
            for name, (mode, sweeps) in arms.items():
                img, fine, snaps = paint(goal, mode, sweeps, stages if (first and name == "coarse2") else None)
                outs[name] = (img, fine)
                if first and name == "coarse2": pics["stages"] = snaps
            for name, (img, fine) in outs.items():
                acc[name]["agree"] += cos(fwd.top(img), goal).sum().item(); acc[name]["err"] += F.mse_loss(show(img), show(x), reduction="none").mean((1, 2, 3)).sum().item()
                acc[name]["sharp"] += sharpness(show(img)).sum().item(); acc[name]["fine"] += (fine if fine is not None else 0.0) * len(b)
                if first: pics[name] = show(img[:6])[:, 0].cpu()
            if first: pics["real"] = show(x[:6])[:, 0].cpu()
            log(f"  {int(b[-1] - ev[0]) + 1} faces  ({time.time() - t0:.0f}s)")
        res["faces"] = {k: {m: v / len(ev) for m, v in d.items()} for k, d in acc.items()}
        for k, d in res["faces"].items():
            log(f"{k:10s} agreement {d['agree']:.3f}   pixel error {d['err']:.4f}   sharpness {d['sharp']:.4f}   fine agreement {d['fine']:.3f}")

        # imagined goals
        Y = torch.cat([fwd.top(images(b)).flatten(1) for b in batches(idx_train, 250, False)])
        mu = Y.mean(0); U, Sv, V = torch.pca_lowrank(Y - mu, q=64, center=False); scale = Sv / (len(Y) - 1) ** 0.5
        g = torch.Generator(device=dev).manual_seed(a.seed)
        goal = F.relu(mu + (torch.randn(8, 64, device=dev, generator=g) * scale) @ V.T).view(-1, CH[-1], 5, 4)
        del Y, U
        res["imagined"] = {}; pics["imagined"] = {"coarse": show(whole.render(goal))[:, 0].cpu()}
        res["imagined"]["coarse"] = {"agree": cos(fwd.top(whole.render(goal)), goal).mean().item(), "sharp": sharpness(show(whole.render(goal))).mean().item()}
        for name in ("first", "coarse2", "both2"):
            mode, sweeps = arms[name]; img, fine, _ = paint(goal, mode, sweeps)
            res["imagined"][name] = {"agree": cos(fwd.top(img), goal).mean().item(), "sharp": sharpness(show(img)).mean().item(), "fine": fine}
            pics["imagined"][name] = show(img)[:, 0].cpu()
        log("imagined: " + "   ".join(f"{k} agree {v['agree']:.3f} sharp {v['sharp']:.4f}" for k, v in res["imagined"].items()))

    figures(pics)
    L = ["# Results", "", f"{a.n_eval} test faces as goals; library of {a.n_lib} faces x 49 positions; {a.k} candidates per window.", "",
         "agreement = cosine between the goal and the read-back of the finished canvas.  fine = mean cosine between each chosen part and the read-back of its window.", "",
         "| arm | agreement | pixel error | sharpness | fine agreement |", "|---|---|---|---|---|"]
    for k in ["blur", "first", "random", "coarse1", "coarse2", "both2", "from_real"]:
        d = res["faces"][k]; L.append(f"| {k} | {d['agree']:.3f} | {d['err']:.4f} | {d['sharp']:.4f} | {d['fine']:.3f} |")
    L += ["", "Imagined goals (8 pca64 samples):", "", "| arm | agreement | sharpness |", "|---|---|---|"]
    for k, v in res["imagined"].items(): L.append(f"| {k} | {v['agree']:.3f} | {v['sharp']:.4f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n"); log("\n".join(L))
    json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


def figures(pics):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["real", "blur", "first", "random", "coarse2", "both2", "from_real"]
    fig, ax = plt.subplots(6, len(cols), figsize=(len(cols) * 1.3, 9.5))
    for i in range(6):
        for j, k in enumerate(cols):
            ax[i, j].imshow(pics[k][i], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    for j, k in enumerate(cols): ax[0, j].set_title(k, fontsize=8)
    fig.suptitle("painting real-face goals: parts from the library, chosen by the look-back (coarse2, both2) or not (first, random)", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "painter_faces.png", dpi=130); plt.close(fig)

    keys = sorted(pics["stages"]); labels = [f"sweep {sw + 1}, {p + 1} of 49" for sw, p in keys]
    fig, ax = plt.subplots(2, len(keys), figsize=(len(keys) * 1.6, 4.2))
    for i in range(2):
        for j, k in enumerate(keys):
            ax[i, j].imshow(pics["stages"][k][i, 0], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    for j, lab in enumerate(labels): ax[0, j].set_title(lab, fontsize=8)
    fig.suptitle("the painting in progress (coarse look-back): windows in raster order, then the second sweep", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "painting.png", dpi=130); plt.close(fig)

    keys = list(pics["imagined"])
    fig, ax = plt.subplots(len(keys), 8, figsize=(10, len(keys) * 1.6))
    for i, k in enumerate(keys):
        for j in range(8):
            ax[i, j].imshow(pics["imagined"][k][j], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(k, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("imagined goals painted from the library", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "painter_imagined.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
