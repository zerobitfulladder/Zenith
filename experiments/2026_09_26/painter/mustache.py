"""Can the painter put a mustache on a woman?

Same painter as run.py.  Goals are the codes of test women.  Three ways of painting:
    plain        as in run.py: all parts allowed, chosen by the coarse look-back
    parts        at the six windows over the upper lip, the candidates are drawn only from
                 the mustached faces in the library; the look-back still chooses the one
                 that keeps the whole reading as her.  Everywhere else as plain.
    parts+goal   the same, and the goal itself has the mustache difference (men with minus
                 men without, three times) added at the two top-map cells over the lip
Read by the stack's own head: P(Male), P(Mustache), P(No_Beard), on the finished canvas.
The same on eight imagined faces.

Usage: python mustache.py     # ~6 min
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
POS = [(t, l) for t in range(0, H - WH + 1, 20) for l in range(0, W - WW + 1, 16)]
LIP = [p for p, (t, l) in enumerate(POS) if t in (80, 100) and l in (32, 48, 64)]      # six windows over the upper lip
LIP_CELLS = [(3, 1), (3, 2)]
LOG = open(OUT / "mustache.log", "w")


def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


raw = np.load(DATA / "gray_218x178.npy", mmap_mode="r")
imgs = torch.from_numpy(np.ascontiguousarray(raw[:, 29:189, 25:153])).to(dev)
attrs = torch.from_numpy(np.load(DATA / "attrs_40.npy")).float().to(dev)
names = json.loads((DATA / "attr_names_40.json").read_text()); A = {n: i for i, n in enumerate(names)}
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


def ramp(n):
    i = torch.arange(n, device=dev, dtype=torch.float32)
    return 0.5 - 0.5 * torch.cos(2 * np.pi * (i + 0.5) / n)


FEATHER = (ramp(WH)[:, None] * ramp(WW)[None, :] + 1e-3).view(1, 1, WH, WW)
W_BG = 0.05


def cos(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)


class Canvas:
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
        t, l = POS[p]
        num = s.num.clone(); den = s.den.clone()
        if p in s.parts:
            num[:, :, t:t + WH, l:l + WW] -= FEATHER * s.parts[p]; den[:, :, t:t + WH, l:l + WW] -= FEATHER
        num[:, :, t:t + WH, l:l + WW] += FEATHER * r; den[:, :, t:t + WH, l:l + WW] += FEATHER
        return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_lib", type=int, default=1000); ap.add_argument("--k", type=int, default=8); ap.add_argument("--n_eval", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    st = torch.load(NETS, map_location=dev)
    fwd, whole, win = Forward().to(dev), Decoder().to(dev), Decoder().to(dev)
    fwd.load_state_dict(st["fwd"]); whole.load_state_dict(st["whole"]); win.load_state_dict(torch.load(WIN40, map_location=dev))
    for m in (fwd, whole, win): m.requires_grad_(False); m.eval()

    with torch.no_grad():
        lib = idx_train[:a.n_lib]
        T = torch.zeros(len(POS), a.n_lib, D, device=dev, dtype=torch.float16); K = torch.zeros_like(T)
        for i, b in enumerate(batches(lib, 250, False)):
            x = images(b); blur = whole.render(fwd.top(x)); sl = slice(i * 250, i * 250 + len(b))
            for p, (t_, l_) in enumerate(POS):
                T[p, sl] = fwd.top(zoom(x, t_, l_)).flatten(1).half()
                K[p, sl] = F.normalize(fwd.top(zoom(blur, t_, l_)).flatten(1), dim=1).half()
        must_lib = (attrs[lib, A["Mustache"]] == 1)
        log(f"library: {a.n_lib} faces, {int(must_lib.sum())} of them with a mustache  ({time.time() - t0:.0f}s)")

        # the mustache difference in the top map, from all training faces
        Y = torch.cat([fwd.top(images(b)).flatten(1) for b in batches(idx_train, 250, False)]); At = attrs[idx_train]
        male, must = At[:, A["Male"]] == 1, At[:, A["Mustache"]] == 1
        delta = (Y[male & must].mean(0) - Y[male & ~must].mean(0)).view(1, CH[-1], 5, 4); del Y

        def candidates(key, p, k, only_must):
            sims = F.normalize(key.flatten(1), dim=1).half() @ K[p].T
            if only_must: sims = sims.masked_fill(~must_lib.view(1, -1), -2.0)
            idx = sims.topk(k, dim=1).indices
            return T[p][idx].float().view(key.shape[0], k, CH[-1], 5, 4)

        def paint(goal, lip_mustache, sweeps=2):
            B = goal.shape[0]; cv = Canvas(whole.render(goal))
            for sw in range(sweeps):
                for p, (t_, l_) in enumerate(POS):
                    key = fwd.top(zoom(cv.image(), t_, l_))
                    cands = candidates(key, p, a.k, lip_mustache and p in LIP)
                    scores = []
                    for jj in range(a.k):
                        r = unzoom(win.render(cands[:, jj])); scores.append(cos(fwd.top(cv.image_with(p, r)), goal))
                    j = torch.stack(scores, 1).argmax(1)
                    cv.set(p, unzoom(win.render(cands[torch.arange(B, device=dev), j])))
            return cv.image()

        def readings(x):
            pr = fwd.probs(x); return {"p_male": pr[:, A["Male"]].mean().item(), "p_mustache": pr[:, A["Mustache"]].mean().item(), "p_no_beard": pr[:, A["No_Beard"]].mean().item()}

        women = idx_test[attrs[idx_test, A["Male"]] == 0][:a.n_eval]
        res, pics = {}, {}
        acc = {}
        for b in batches(women, 50, False):
            x = images(b); goal = fwd.top(x); first = b[0] == women[0]
            goal_m = goal.clone()
            for (r0, c0) in LIP_CELLS: goal_m[:, :, r0, c0] += 3 * delta[:, :, r0, c0]
            outs = {"real women": x, "plain": paint(goal, False), "parts": paint(goal, True), "parts+goal": paint(goal_m, True)}
            for k, img in outs.items():
                rd = readings(img); acc.setdefault(k, {m: 0.0 for m in rd})
                for m, v in rd.items(): acc[k][m] += v * len(b)
                if first: pics[k] = show(img[:8])[:, 0].cpu()
            log(f"  {int((b[-1] - women[0]).item()) + 1 if False else len(acc)} arms, batch done  ({time.time() - t0:.0f}s)")
        res["women"] = {k: {m: v / len(women) for m, v in d.items()} for k, d in acc.items()}
        # reference: real men with a mustache
        men_m = idx_test[(attrs[idx_test, A["Male"]] == 1) & (attrs[idx_test, A["Mustache"]] == 1)][:100]
        res["women"]["real men with mustache"] = readings(images(men_m))
        for k, d in res["women"].items():
            log(f"{k:24s} P(Male) {d['p_male']:.2f}   P(Mustache) {d['p_mustache']:.2f}   P(No_Beard) {d['p_no_beard']:.2f}")

        # imagined faces
        Yt = torch.cat([fwd.top(images(b)).flatten(1) for b in batches(idx_train, 250, False)])
        mu = Yt.mean(0); U, Sv, V = torch.pca_lowrank(Yt - mu, q=64, center=False); scale = Sv / (len(Yt) - 1) ** 0.5; del Yt, U
        g = torch.Generator(device=dev).manual_seed(a.seed)
        goal = F.relu(mu + (torch.randn(8, 64, device=dev, generator=g) * scale) @ V.T).view(-1, CH[-1], 5, 4)
        pics["imagined plain"] = show(paint(goal, False))[:, 0].cpu(); pics["imagined parts"] = show(paint(goal, True))[:, 0].cpu()
        res["imagined"] = {"plain": readings(whole.render(goal)), "parts": None}
        log(f"imagined done  ({time.time() - t0:.0f}s)")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    rows = ["real women", "plain", "parts", "parts+goal", "imagined plain", "imagined parts"]
    fig, ax = plt.subplots(len(rows), 8, figsize=(10, len(rows) * 1.65))
    for i, k in enumerate(rows):
        for j in range(8):
            ax[i, j].imshow(pics[k][j], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if "parts" in k: ax[i, j].add_patch(Rectangle((32 - 0.5, 80 - 0.5), 64, 60, fill=False, edgecolor="tab:orange", lw=0.6))
        ax[i, 0].set_ylabel(k, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("a mustache for a woman: the painter with mustached parts offered at the lip windows (orange), chosen by the look-back", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "mustache.png", dpi=130); plt.close(fig)

    L = ["# A mustache for a woman", "", f"{len(women)} test women as goals.  The stack's own head on the finished canvas.", "",
         "| painting | P(Male) | P(Mustache) | P(No_Beard) |", "|---|---|---|---|"]
    for k, d in res["women"].items(): L.append(f"| {k} | {d['p_male']:.2f} | {d['p_mustache']:.2f} | {d['p_no_beard']:.2f} |")
    (OUT / "mustache.md").write_text("\n".join(L) + "\n"); log("\n".join(L))
    json.dump(res, open(OUT / "mustache.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
