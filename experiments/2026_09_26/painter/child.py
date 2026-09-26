"""Would it draw a face like a child, given only lines, ovals and dots?

The painter of run.py with a different library and a different canvas.  The library is a few
hundred drawn primitives on a 40x32 window: lines at eight angles, three offsets and two
thicknesses; oval outlines and filled ovals of four sizes at five positions; arcs (half ovals,
up and down) of three sizes at three positions; dots; and blanks.  The canvas is paper: light,
with nothing of the face on it.  Ink accumulates: where windows overlap the darker stroke
shows.  For every window, every primitive is tried, the whole canvas is read back through
the stack, and the primitive that makes it read most like the goal is kept.  Two sweeps.

Scored by agreement of the drawing's read-back with the goal, against the coarse render's
and blank paper's; by the stack's own head on the drawing for a few attributes of the goal
face; and by which primitives were used where.

Usage: python child.py     # ~4 min
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data/celeba"
NETS = HERE.parent / "faces/results/nets.pt"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
dev = "cuda"
torch.backends.cudnn.benchmark = True
H, W = 160, 128
CH = (16, 32, 64, 128, 256)
D = CH[-1] * 20
MEAN, STD = 0.45, 0.25
WH, WW = 40, 32
POS = [(t, l) for t in range(0, H - WH + 1, 20) for l in range(0, W - WW + 1, 16)]
PAPER = 0.85
LOG = open(OUT / "child.log", "w")


def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


raw = np.load(DATA / "gray_218x178.npy", mmap_mode="r")
imgs = torch.from_numpy(np.ascontiguousarray(raw[:, 29:189, 25:153])).to(dev)
attrs = torch.from_numpy(np.load(DATA / "attrs_40.npy")).float().to(dev)
names = json.loads((DATA / "attr_names_40.json").read_text()); A = {n: i for i, n in enumerate(names)}
N = len(imgs); idx_test = torch.arange(N - 5000, N, device=dev); idx_train = torch.arange(0, N - 7000, device=dev)


def images(idx):
    return imgs[idx].float().div(255).sub(MEAN).div(STD).unsqueeze(1)


def norm(x01):
    return (x01 - MEAN) / STD


def show(x):
    return (x * STD + MEAN).clamp(0, 1)


def batches(idx, bs):
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


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

    def probs(s, x):
        return torch.sigmoid(s.head(s.top(x).flatten(1)))


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


# ---------------------------------------------------------------- the primitives, drawn on a 40x32 window, values in [0, 1]
def primitives():
    yy, xx = np.mgrid[0:WH, 0:WW].astype(np.float32); cy, cx = (WH - 1) / 2, (WW - 1) / 2
    shapes, kinds = [], []
    def add(mask, kind, ink=0.1):
        img = np.full((WH, WW), PAPER, np.float32); img[mask] = ink; shapes.append(img); kinds.append(kind)
    add(np.zeros((WH, WW), bool), "blank")
    for ink in (0.5, 0.2): shapes.append(np.full((WH, WW), ink, np.float32)); kinds.append("fill")
    for ang in np.arange(0, np.pi, np.pi / 8):
        n = np.array([-np.sin(ang), np.cos(ang)])
        for off in (-8, 0, 8):
            for th in (1.0, 2.0):
                d = (yy - cy) * n[0] + (xx - cx) * n[1] - off
                add(np.abs(d) <= th, "line")
    for (ry, rx) in ((4, 6), (6, 10), (8, 14), (12, 12)):
        for (dy, dx) in ((0, 0), (0, -6), (0, 6), (-6, 0), (6, 0)):
            r = np.sqrt(((yy - cy - dy) / ry) ** 2 + ((xx - cx - dx) / rx) ** 2)
            add(np.abs(r - 1) <= 1.5 / min(rx, ry), "oval")
            add(r <= 1, "filled oval")
    for (ry, rx) in ((5, 8), (8, 12), (10, 14)):
        for dx in (-6, 0, 6):
            r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx - dx) / rx) ** 2)
            ring = np.abs(r - 1) <= 1.5 / min(rx, ry)
            add(ring & (yy < cy), "arc up"); add(ring & (yy > cy), "arc down")
    for dy in (-10, 0, 10):
        for dx in (-8, 0, 8):
            add(np.sqrt((yy - cy - dy) ** 2 + (xx - cx - dx) ** 2) <= 2.5, "dot")
    return torch.tensor(np.stack(shapes), device=dev).unsqueeze(1), kinds


def cos(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n_eval", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed)
    st = torch.load(NETS, map_location=dev)
    fwd, whole = Forward().to(dev), Decoder().to(dev)
    fwd.load_state_dict(st["fwd"]); whole.load_state_dict(st["whole"])
    for m in (fwd, whole): m.requires_grad_(False); m.eval()
    P, kinds = primitives(); nP = len(P)
    kinds_arr = np.array(kinds); log(f"{nP} primitives: " + ", ".join(f"{k} {int((kinds_arr == k).sum())}" for k in dict.fromkeys(kinds)))

    @torch.no_grad()
    def compose(parts, B):
        canvas = torch.full((B, 1, H, W), PAPER, device=dev)
        for p, img in parts.items():
            t, l = POS[p]; canvas[:, :, t:t + WH, l:l + WW] = torch.minimum(canvas[:, :, t:t + WH, l:l + WW], img)
        return canvas

    @torch.no_grad()
    def draw(goal, sweeps=2, snaps_at=()):
        B = goal.shape[0]; parts = {}; choice = torch.zeros(sweeps, len(POS), B, dtype=torch.long); snaps = {}
        for sw in range(sweeps):
            for p, (t, l) in enumerate(POS):
                others = {q: v for q, v in parts.items() if q != p}
                base = compose(others, B)
                scores = []
                for j in range(nP):
                    cv = base.clone(); cv[:, :, t:t + WH, l:l + WW] = torch.minimum(cv[:, :, t:t + WH, l:l + WW], P[j].expand(B, -1, -1, -1))
                    scores.append(cos(fwd.top(norm(cv)), goal))
                j = torch.stack(scores, 1).argmax(1); choice[sw, p] = j.cpu()
                parts[p] = P[j]
                if (sw, p) in snaps_at: snaps[(sw, p)] = compose(parts, B)[:2].cpu()
        return compose(parts, B), choice, snaps

    with torch.no_grad():
        ev = idx_test[:a.n_eval]; res = {"agree_paper": 0.0, "agree_blur": 0.0, "agree_drawing": 0.0}; pics = {}; choices = []
        head_hits = {k: [0, 0] for k in ("Male", "Smiling", "Eyeglasses", "Blond_Hair")}
        for b in batches(ev, 8):
            x = images(b); goal = fwd.top(x); first = b[0] == ev[0]
            blur = whole.render(goal)
            drawing, choice, snaps = draw(goal, 2, snaps_at={(0, 10), (0, 24), (0, 48), (1, 48)} if first else ())
            choices.append(choice)
            res["agree_paper"] += cos(fwd.top(norm(torch.full_like(x, PAPER))), goal).sum().item()
            res["agree_blur"] += cos(fwd.top(blur), goal).sum().item()
            res["agree_drawing"] += cos(fwd.top(norm(drawing)), goal).sum().item()
            pr = fwd.probs(norm(drawing)); pr_blur = fwd.probs(blur)
            for k in head_hits:
                head_hits[k][0] += ((pr[:, A[k]] > 0.5).float() == attrs[b, A[k]]).sum().item()
                head_hits[k][1] += ((pr_blur[:, A[k]] > 0.5).float() == attrs[b, A[k]]).sum().item()
            if first: pics["real"] = show(x[:8])[:, 0].cpu(); pics["blur"] = show(blur[:8])[:, 0].cpu(); pics["drawing"] = drawing[:8, 0].cpu(); pics["snaps"] = snaps
            log(f"  {len(choices) * 8} faces drawn  ({time.time() - t0:.0f}s)")
        n = len(ev); res = {k: v / n for k, v in res.items()}
        res["head_on_drawing"] = {k: v[0] / n for k, v in head_hits.items()}; res["head_on_blur"] = {k: v[1] / n for k, v in head_hits.items()}
        choice = torch.cat(choices, 2)                                                   # (sweeps, 49, n)
        final = kinds_arr[choice[-1].numpy()]                                              # (49, n)
        rows = {"brow/eyes (rows 60-120)": [p for p, (t, l) in enumerate(POS) if t in (60, 80)], "mouth (rows 100-140)": [p for p, (t, l) in enumerate(POS) if t == 100],
                "hair/top (rows 0-40)": [p for p, (t, l) in enumerate(POS) if t in (0, 20)], "chin/neck (row 120)": [p for p, (t, l) in enumerate(POS) if t == 120]}
        res["kinds_by_region"] = {}
        for name, ps in rows.items():
            k_, c_ = np.unique(final[ps].ravel(), return_counts=True); order = np.argsort(-c_)
            res["kinds_by_region"][name] = {str(k_[i]): int(c_[i]) for i in order}
        res["changed_in_sweep_2"] = float((choice[0] != choice[1]).float().mean())
        log(f"agreement with the goal: paper {res['agree_paper']:.3f}   coarse render {res['agree_blur']:.3f}   drawing {res['agree_drawing']:.3f}")
        log("head right on the drawing / on the coarse render: " + "  ".join(f"{k} {res['head_on_drawing'][k]:.2f}/{res['head_on_blur'][k]:.2f}" for k in head_hits))
        for name, d in res["kinds_by_region"].items(): log(f"  {name:26s} " + ", ".join(f"{k} {v}" for k, v in d.items()))
        log(f"windows redrawn in the second sweep: {res['changed_in_sweep_2']:.2f}")

        # imagined
        Y = torch.cat([fwd.top(images(b)).flatten(1) for b in batches(idx_train, 250)])
        mu = Y.mean(0); U, Sv, V = torch.pca_lowrank(Y - mu, q=64, center=False); scale = Sv / (len(Y) - 1) ** 0.5; del Y, U
        g = torch.Generator(device=dev).manual_seed(a.seed)
        goal = F.relu(mu + (torch.randn(8, 64, device=dev, generator=g) * scale) @ V.T).view(-1, CH[-1], 5, 4)
        drawing, _, _ = draw(goal, 2)
        res["imagined_agree"] = {"coarse": cos(fwd.top(whole.render(goal)), goal).mean().item(), "drawing": cos(fwd.top(norm(drawing)), goal).mean().item()}
        pics["imagined coarse"] = show(whole.render(goal))[:, 0].cpu(); pics["imagined drawing"] = drawing[:, 0].cpu()
        log(f"imagined: agreement coarse {res['imagined_agree']['coarse']:.3f}   drawing {res['imagined_agree']['drawing']:.3f}  ({time.time() - t0:.0f}s)")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows_ = ["real", "blur", "drawing", "imagined coarse", "imagined drawing"]
    fig, ax = plt.subplots(len(rows_), 8, figsize=(10, len(rows_) * 1.7))
    for i, k in enumerate(rows_):
        for j in range(8):
            ax[i, j].imshow(pics[k][j], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(k, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("drawing faces with lines, ovals, arcs and dots, each stroke chosen by reading the whole drawing back", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "child.png", dpi=130); plt.close(fig)
    keys = sorted(pics["snaps"]); labels = [f"sweep {sw + 1}, {p + 1} of 49" for sw, p in keys]
    fig, ax = plt.subplots(2, len(keys) + 1, figsize=((len(keys) + 1) * 1.7, 4.2))
    for i in range(2):
        ax[i, 0].imshow(pics["real"][i], cmap="gray", vmin=0, vmax=1)
        for j, k in enumerate(keys): ax[i, j + 1].imshow(pics["snaps"][k][i, 0], cmap="gray", vmin=0, vmax=1)
        for j in range(len(keys) + 1): ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    ax[0, 0].set_title("goal face", fontsize=8)
    for j, lab in enumerate(labels): ax[0, j + 1].set_title(lab, fontsize=8)
    fig.suptitle("the drawing in progress", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "child_progress.png", dpi=130); plt.close(fig)
    json.dump(res, open(OUT / "child.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
