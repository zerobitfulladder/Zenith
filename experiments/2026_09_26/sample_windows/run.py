"""Unique and crisp: sample the top code for a new face, render it coarse, then slide a window
over the whole face and sample each window's missing detail conditioned on the coarse face.

Uses the three networks the faces experiment trained (../faces/results/nets.pt): the forward
stack, the whole-face decoder, and the half-size window decoder (trained on random 80x64
windows anywhere on the face, zoomed twice).

Unique.  A Gaussian fitted to the real top maps of the training faces, two ways: one spread
per unit (diag), and a Gaussian in the space of the top 64 principal components (pca64).
Sample, render with the whole decoder.

Crisp.  For a face, the coarse render is what its top code gives.  Take a window of it, zoom
it to 160x128, run the stack: c, the code of the blurry region.  The code of the same window
of the real face, t, is what the window decoder renders sharply.  A conditional sampler
learns p(t | c): encoder q(z | t, c), prior p(z | c), decoder g(z, c), small MLPs on the
standardised 5120-dim codes, squared error on t plus the KL between q and p.  Trained on one
random window per training face.  The sampler never sees where the window is; c carries what
is there.

Rebuild.  Nine windows of 80x64 at stride 40x32 cover the face.  Each window's code comes
from one of:
    from_blur   the blurry region zoomed through the stack (deterministic, no sampler)
    mean        the sampler at the prior mean
    sample      the sampler with z drawn from p(z | c)
    nearest     the real window code of the training window whose blurry code is nearest to
                this one (a crisp candidate picked, not an average)
    from_real   the real face's window code (the ceiling; not available for an imagined face)
rendered by the window decoder, shrunk, pasted, overlaps averaged.  Scored on test faces by
pixel error to the real face and by sharpness (mean absolute Laplacian), against the coarse
render (blur) and the real face.

Then both halves: a pca64 sample, rendered coarse, rebuilt with sampled windows.

Two fixes for the seams, run against the hard-edged 80x64 grid: feathering (each window's
render fades to its edge under a raised-cosine weight, overlaps summed by weight), and a
40x32 window decoder with its own sampler on a 7x7 grid at stride 20x16.  The 40x32 decoder
is trained here by mismatch, four epochs, and saved in results/win40.pt.

Usage: python run.py     # ~5 min the first time, ~2 min after
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
WH, WW = 80, 64
WINDOWS = [(t, l) for t in (0, 40, 80) for l in (0, 32, 64)]
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
        return s.maps(x)[0][-1]

    def maps(s, x):
        ys, As, h = [x], [], x
        for conv in s.convs:
            a = F.relu(conv(h)); h = F.max_pool2d(a, 2); As.append(a); ys.append(h)
        return ys, As


class Decoder(nn.Module):
    def __init__(s):
        super().__init__()
        cins = (1,) + CH[:-1]
        s.unpools = nn.ModuleList([nn.Conv2d(c, c, 3, padding=1) for c in CH])
        s.tconvs = nn.ModuleList([nn.ConvTranspose2d(co, ci, 3, padding=1) for ci, co in zip(cins, CH)])

    def unpool(s, l, y):
        return F.relu(s.unpools[l - 1](F.interpolate(y, scale_factor=2, mode="nearest")))

    def down(s, l, a):
        out = s.tconvs[l - 1](a)
        return F.relu(out) if l > 1 else out

    def render(s, y):
        for l in range(len(CH), 0, -1):
            y = s.down(l, s.unpool(l, y))
        return y


def zoomw(x, t, l, wh, ww):
    return F.interpolate(x[:, :, t:t + wh, l:l + ww], size=(H, W), mode="bilinear", align_corners=False)


def unzoomw(r, wh, ww):
    return F.interpolate(r, size=(wh, ww), mode="area")


def feather(wh, ww):
    def ramp(n):
        i = torch.arange(n, device=dev, dtype=torch.float32)
        return 0.5 - 0.5 * torch.cos(2 * np.pi * (i + 0.5) / n)
    return (ramp(wh)[:, None] * ramp(ww)[None, :] + 1e-3).view(1, 1, wh, ww)


def train_window_decoder(dec, fwd, wh, ww, epochs):
    """Local mismatch, as in faces/: unpool toward the pre-pool map, transposed conv toward the map below."""
    opt = torch.optim.Adam(dec.parameters(), 1e-3)
    for ep in range(epochs):
        tot = 0.0; nb = 0
        for b in batches(idx_train, 64):
            t_, l_ = np.random.randint(0, H - wh + 1), np.random.randint(0, W - ww + 1)
            with torch.no_grad():
                ys, As = fwd.maps(zoomw(images(b), t_, l_, wh, ww))
            losses = []
            for lv in range(1, len(CH) + 1):
                a_hat = dec.unpool(lv, ys[lv]); losses.append(F.mse_loss(a_hat, As[lv - 1]))
                losses.append(F.mse_loss(dec.down(lv, a_hat.detach()), ys[lv - 1]))
            opt.zero_grad(set_to_none=True); sum(losses).backward(); opt.step()
            tot += losses[1].item(); nb += 1
        log(f"    ep {ep + 1}  pixel mismatch {tot / nb:.4f}")


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class Sampler(nn.Module):
    """p(t | c) through a latent z: encoder q(z | t, c), prior p(z | c), decoder g(z, c)."""
    def __init__(s, d=D, h=512, z=64):
        super().__init__()
        s.enc = mlp(2 * d, h, 2 * z); s.prior = mlp(d, h, 2 * z); s.dec = mlp(z + d, 1024, d)

    def gauss(s, out):
        mu, lv = out.chunk(2, 1)
        return mu, lv.clamp(-6, 4)

    def forward(s, t, c):
        mu, lv = s.gauss(s.enc(torch.cat([t, c], 1))); pm, plv = s.gauss(s.prior(c))
        zs = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        t_hat = s.dec(torch.cat([zs, c], 1))
        kl = 0.5 * (plv - lv + (torch.exp(lv) + (mu - pm) ** 2) / torch.exp(plv) - 1).sum(1)
        return t_hat, kl

    def draw(s, c, sample=True):
        pm, plv = s.gauss(s.prior(c))
        zs = pm + (torch.exp(0.5 * plv) * torch.randn_like(pm) if sample else 0)
        return s.dec(torch.cat([zs, c], 1))


LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=dev).view(1, 1, 3, 3)


def sharpness(x):
    return F.conv2d(x, LAP).abs().mean((1, 2, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n_eval", type=int, default=1000)
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    st = torch.load(NETS, map_location=dev)
    fwd, whole, win = Forward().to(dev), Decoder().to(dev), Decoder().to(dev)
    fwd.load_state_dict(st["fwd"]); whole.load_state_dict(st["whole"]); win.load_state_dict(st["mouth"])
    for m in (fwd, whole, win): m.requires_grad_(False); m.eval()
    res, pics = {}, {}

    # ------------------------------------------------ unique: a Gaussian over the real top codes
    with torch.no_grad():
        Y = torch.cat([fwd.top(images(b)).flatten(1) for b in batches(idx_train, 250, False)])
        mu, sd = Y.mean(0), Y.std(0)
        U, S, V = torch.pca_lowrank(Y - mu, q=64, center=False)
        scale = S / (len(Y) - 1) ** 0.5
        res["pca64_variance_kept"] = float((S ** 2).sum() / ((Y - mu) ** 2).sum())
        g = torch.Generator(device=dev).manual_seed(a.seed)
        samp = {"diag": mu + sd * torch.randn(64, D, device=dev, generator=g),
                "pca64": mu + (torch.randn(64, 64, device=dev, generator=g) * scale) @ V.T}
        real_r = show(whole.render(Y[:64].view(-1, CH[-1], 5, 4)))
        def spread(r): return float(torch.pdist(r.flatten(1)).mean() / (H * W) ** 0.5)
        res["diversity"] = {"real renders": spread(real_r)}
        for k, s_ in samp.items():
            r = show(whole.render(F.relu(s_).view(-1, CH[-1], 5, 4)))
            res["diversity"][k] = spread(r); pics[k] = r[:8, 0].cpu()
        pics["real"] = real_r[:8, 0].cpu()
        del Y, U
        log(f"unique: pca64 keeps {res['pca64_variance_kept']:.2f} of the variance; mean pixel distance between faces: " +
            "  ".join(f"{k} {v:.3f}" for k, v in res["diversity"].items()) + f"  ({time.time() - t0:.0f}s)")

    # ------------------------------------------------ training pairs and sampler, per window size
    @torch.no_grad()
    def pairs(wh, ww):
        T, C = [], []
        for b in batches(idx_train, 250, False):
            x = images(b); coarse = whole.render(fwd.top(x))
            t_, l_ = np.random.randint(0, H - wh + 1), np.random.randint(0, W - ww + 1)
            T.append(fwd.top(zoomw(x, t_, l_, wh, ww)).flatten(1).half()); C.append(fwd.top(zoomw(coarse, t_, l_, wh, ww)).flatten(1).half())
        T, C = torch.cat(T), torch.cat(C)
        tm, ts = T.float().mean(0), T.float().std(0); cm, cs = C.float().mean(0), C.float().std(0)
        return T, C, (tm, ts.clamp_min(0.05 * ts.mean()), cm, cs.clamp_min(0.05 * cs.mean()))

    def train_sampler(T, C, stats):
        tm, ts, cm, cs = stats
        net = Sampler().to(dev); opt = torch.optim.Adam(net.parameters(), 3e-4)
        for ep in range(a.epochs):
            tot = torch.zeros(2, device=dev); nb = 0
            for b in batches(torch.arange(len(T), device=dev), 256):
                t = (T[b].float() - tm) / ts; c = (C[b].float() - cm) / cs
                t_hat, kl = net(t, c)
                rec = F.mse_loss(t_hat, t); loss = rec + a.beta * kl.mean() / D
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
                tot += torch.stack([rec.detach(), kl.mean().detach()]); nb += 1
            log(f"    ep {ep + 1}  code error {tot[0] / nb:.4f}  KL {tot[1] / nb:.1f} nats")
        return net.requires_grad_(False).eval()

    T, C, stats80 = pairs(WH, WW); tm, ts, cm, cs = stats80
    log(f"{len(T)} window pairs of 80x64; code values up to {T.float().max():.1f}  ({time.time() - t0:.0f}s)")
    net = train_sampler(T, C, stats80)

    # ------------------------------------------------ rebuild
    Cn = F.normalize((C.float() - cm) / cs, dim=1).half()                                     # for nearest-neighbour lookup

    @torch.no_grad()
    def nearest(c):
        sims = F.normalize(c, dim=1).half() @ Cn.T
        return T[sims.argmax(1)].float().view(-1, CH[-1], 5, 4)

    @torch.no_grad()
    def rebuild(coarse, how, x=None):
        canvas = torch.zeros_like(coarse); count = torch.zeros_like(coarse)
        for (t_, l_) in WINDOWS:
            c_raw = fwd.top(zoom(coarse, t_, l_))
            if how == "from_blur": code = c_raw
            elif how == "from_real": code = fwd.top(zoom(x, t_, l_))
            elif how == "nearest": code = nearest((c_raw.flatten(1) - cm) / cs)
            else:
                c = (c_raw.flatten(1) - cm) / cs
                code = F.relu(net.draw(c, sample=(how == "sample")) * ts + tm).view(-1, CH[-1], 5, 4)
            r = unzoom(win.render(code))
            canvas[:, :, t_:t_ + WH, l_:l_ + WW] += r; count[:, :, t_:t_ + WH, l_:l_ + WW] += 1
        return canvas / count

    with torch.no_grad():
        ways = ["real", "blur", "from_blur", "mean", "sample", "nearest", "from_real"]
        err = {k: 0.0 for k in ways}; shp = {k: 0.0 for k in ways}
        ev = idx_test[:a.n_eval]
        for b in batches(ev, 100, False):
            x = images(b); coarse = whole.render(fwd.top(x))
            out = {"real": x, "blur": coarse, "from_blur": rebuild(coarse, "from_blur"), "mean": rebuild(coarse, "mean"),
                   "sample": rebuild(coarse, "sample"), "nearest": rebuild(coarse, "nearest"), "from_real": rebuild(coarse, "from_real", x)}
            for k in ways:
                err[k] += F.mse_loss(show(out[k]), show(x), reduction="sum").item() / (H * W)
                shp[k] += sharpness(show(out[k])).sum().item()
            if b[0] == ev[0]:
                pics["rebuild"] = {k: show(out[k][:6])[:, 0].cpu() for k in ways}
                pics["rebuild"]["sample 2"] = show(rebuild(coarse[:6], "sample"))[:, 0].cpu()
        res["rebuild"] = {k: {"pixel_error": err[k] / len(ev), "sharpness": shp[k] / len(ev)} for k in ways}
        for k in ways:
            log(f"rebuild {k:10s} pixel error {res['rebuild'][k]['pixel_error']:.4f}   sharpness {res['rebuild'][k]['sharpness']:.4f}")
        log(f"({time.time() - t0:.0f}s)")

        # ------------------------------------------------ both halves: imagined faces, rebuilt
        y_new = F.relu(samp["pca64"][:8]).view(-1, CH[-1], 5, 4)
        coarse = whole.render(y_new)
        crisp = rebuild(coarse, "sample"); crisp2 = rebuild(coarse, "sample"); det = rebuild(coarse, "from_blur"); near = rebuild(coarse, "nearest")
        pics["imagined"] = {"coarse": show(coarse)[:, 0].cpu(), "from_blur": show(det)[:, 0].cpu(), "sample": show(crisp)[:, 0].cpu(), "nearest": show(near)[:, 0].cpu()}
        res["imagined"] = {"coarse": float(sharpness(show(coarse)).mean()), "from_blur": float(sharpness(show(det)).mean()), "sample": float(sharpness(show(crisp)).mean()),
                           "nearest": float(sharpness(show(near)).mean()), "two samples differ by": float((show(crisp) - show(crisp2)).abs().mean())}
        log("imagined faces: sharpness " + "  ".join(f"{k} {v:.4f}" for k, v in res["imagined"].items()))

    # ------------------------------------------------ two fixes: feathering, and 40x32 windows
    win40 = Decoder().to(dev); ck = OUT / "win40.pt"
    if ck.exists():
        win40.load_state_dict(torch.load(ck, map_location=dev)); log("loaded the 40x32 window decoder")
    else:
        log("40x32 window decoder, by mismatch:"); torch.manual_seed(a.seed + 3); train_window_decoder(win40, fwd, 40, 32, 4); torch.save(win40.state_dict(), ck)
    win40.requires_grad_(False); win40.eval()
    with torch.no_grad():
        T40, C40, stats40 = pairs(40, 32)
    log(f"{len(T40)} window pairs of 40x32  ({time.time() - t0:.0f}s)")
    net40 = train_sampler(T40, C40, stats40)

    @torch.no_grad()
    def rebuild2(coarse, how, wh, ww, stride, dec, sampler, stats, soft, x=None):
        tm_, ts_, cm_, cs_ = stats
        wmap = feather(wh, ww) if soft else torch.ones(1, 1, wh, ww, device=dev)
        canvas = torch.zeros_like(coarse); count = torch.zeros_like(coarse)
        for t_ in range(0, H - wh + 1, stride[0]):
            for l_ in range(0, W - ww + 1, stride[1]):
                c_raw = fwd.top(zoomw(coarse, t_, l_, wh, ww))
                if how == "from_blur": code = c_raw
                elif how == "from_real": code = fwd.top(zoomw(x, t_, l_, wh, ww))
                else: code = F.relu(sampler.draw((c_raw.flatten(1) - cm_) / cs_, sample=True) * ts_ + tm_).view(-1, CH[-1], 5, 4)
                r = unzoomw(dec.render(code), wh, ww)
                canvas[:, :, t_:t_ + wh, l_:l_ + ww] += r * wmap; count[:, :, t_:t_ + wh, l_:l_ + ww] += wmap
        return canvas / count

    configs = {"80x64 hard": (WH, WW, (40, 32), win, net, stats80, False), "80x64 feathered": (WH, WW, (40, 32), win, net, stats80, True),
               "40x32 hard": (40, 32, (20, 16), win40, net40, stats40, False), "40x32 feathered": (40, 32, (20, 16), win40, net40, stats40, True)}
    sources = ["from_blur", "sample", "from_real"]
    with torch.no_grad():
        ev = idx_test[:a.n_eval]; fx = {k: {s_: [0.0, 0.0] for s_ in sources} for k in configs}; pics["fix"] = {}
        for b in batches(ev, 100, False):
            x = images(b); coarse = whole.render(fwd.top(x))
            for k, cfg in configs.items():
                for s_ in sources:
                    r = rebuild2(coarse, s_, *cfg, x=x)
                    fx[k][s_][0] += F.mse_loss(show(r), show(x), reduction="sum").item() / (H * W); fx[k][s_][1] += sharpness(show(r)).sum().item()
                    if b[0] == ev[0]: pics["fix"][(k, s_)] = show(r[:6])[:, 0].cpu()
            if b[0] == ev[0]: pics["fix"]["real"] = show(x[:6])[:, 0].cpu(); pics["fix"]["blur"] = show(coarse[:6])[:, 0].cpu()
        res["fixes"] = {k: {s_: {"pixel_error": v[0] / len(ev), "sharpness": v[1] / len(ev)} for s_, v in d.items()} for k, d in fx.items()}
        for k in configs:
            log(f"fix {k:16s} " + "   ".join(f"{s_} err {res['fixes'][k][s_]['pixel_error']:.4f} sharp {res['fixes'][k][s_]['sharpness']:.4f}" for s_ in sources))
        y_new = F.relu(samp["pca64"][:8]).view(-1, CH[-1], 5, 4); coarse = whole.render(y_new)
        pics["imagined2"] = {"coarse": show(coarse)[:, 0].cpu()}
        res["imagined2"] = {"coarse": float(sharpness(show(coarse)).mean())}
        for k in ("80x64 feathered", "40x32 feathered"):
            for s_ in ("from_blur", "sample"):
                r = rebuild2(coarse, s_, *configs[k]); pics["imagined2"][f"{k}, {s_}"] = show(r)[:, 0].cpu(); res["imagined2"][f"{k}, {s_}"] = float(sharpness(show(r)).mean())
        log("imagined, fixed: sharpness " + "  ".join(f"{k} {v:.4f}" for k, v in res["imagined2"].items()) + f"  ({time.time() - t0:.0f}s)")

    figures(pics)
    L = ["# Results", "", f"Unique: a Gaussian over the real top codes.  The 64-component version keeps {res['pca64_variance_kept']:.2f} of the variance.",
         "Mean pixel distance between rendered faces (diversity): " + ", ".join(f"{k} {v:.3f}" for k, v in res["diversity"].items()), "",
         f"Rebuild of {a.n_eval} test faces from nine windows.  Pixel error to the real face; sharpness = mean absolute Laplacian.", "",
         "| way | pixel error | sharpness |", "|---|---|---|"]
    for k in ways: L.append(f"| {k} | {res['rebuild'][k]['pixel_error']:.4f} | {res['rebuild'][k]['sharpness']:.4f} |")
    L += ["", "Imagined faces (pca64 samples): sharpness " + ", ".join(f"{k} {v:.4f}" for k, v in res["imagined"].items()) + ".", "",
          "## Two fixes: feathered blending, and 40x32 windows on a 7x7 grid", "",
          "| windows | source | pixel error | sharpness |", "|---|---|---|---|"]
    for k in configs:
        for s_ in sources:
            L.append(f"| {k} | {s_} | {res['fixes'][k][s_]['pixel_error']:.4f} | {res['fixes'][k][s_]['sharpness']:.4f} |")
    L += ["", "Imagined faces, fixed: sharpness " + ", ".join(f"{k} {v:.4f}" for k, v in res["imagined2"].items()) + "."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n"); log("\n".join(L))
    json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


def figures(pics):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("real faces, rendered", "real"), ("sampled, one spread per unit", "diag"), ("sampled, 64 components", "pca64")]
    fig, ax = plt.subplots(3, 8, figsize=(10, 5))
    for i, (lab, k) in enumerate(rows):
        for j in range(8):
            ax[i, j].imshow(pics[k][j], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("unique: faces rendered from top codes drawn from a Gaussian fitted to the real ones", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "samples.png", dpi=130); plt.close(fig)

    keys = ["real", "blur", "from_blur", "mean", "sample", "nearest", "from_real"]
    fig, ax = plt.subplots(6, len(keys), figsize=(len(keys) * 1.3, 9.5))
    for i in range(6):
        for j, k in enumerate(keys):
            ax[i, j].imshow(pics["rebuild"][k][i], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    for j, k in enumerate(keys): ax[0, j].set_title(k, fontsize=7)
    fig.suptitle("rebuild from nine windows.  blur = coarse render.  from_blur = each window zoomed through the stack.  mean/sample = the conditional sampler.  nearest = a real window picked by its blurry code.  from_real = real window codes", fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "rebuild.png", dpi=130); plt.close(fig)

    keys = ["coarse", "from_blur", "sample", "nearest"]
    fig, ax = plt.subplots(len(keys), 8, figsize=(10, 6.8))
    for i, k in enumerate(keys):
        for j in range(8):
            ax[i, j].imshow(pics["imagined"][k][j], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(k, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("faces that never existed: a sampled top code rendered coarse, then rebuilt window by window, deterministic or sampled", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "imagined.png", dpi=130); plt.close(fig)

    cols = [("real", "real"), ("blur", "blur"), ("80x64 hard\nsample", ("80x64 hard", "sample")), ("80x64 feathered\nsample", ("80x64 feathered", "sample")),
            ("40x32 hard\nsample", ("40x32 hard", "sample")), ("40x32 feathered\nsample", ("40x32 feathered", "sample")), ("40x32 feathered\nfrom_real", ("40x32 feathered", "from_real"))]
    fig, ax = plt.subplots(6, len(cols), figsize=(len(cols) * 1.3, 9.5))
    for i in range(6):
        for j, (lab, k) in enumerate(cols):
            ax[i, j].imshow(pics["fix"][k][i], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    for j, (lab, k) in enumerate(cols): ax[0, j].set_title(lab, fontsize=7)
    fig.suptitle("the two fixes on test faces: feathered blending, and 40x32 windows on a 7x7 grid", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fixes.png", dpi=130); plt.close(fig)

    keys = list(pics["imagined2"])
    fig, ax = plt.subplots(len(keys), 8, figsize=(10, len(keys) * 1.7))
    for i, k in enumerate(keys):
        for j in range(8):
            ax[i, j].imshow(pics["imagined2"][k][j], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(k.replace(", ", "\n"), rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("faces that never existed, rebuilt with the two fixes", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "imagined_fixed.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
