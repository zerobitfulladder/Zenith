"""The loop: render, read back, correct toward the request, render again.  With and without
denoising-trained decoders, correction, and blending of the up-pass states.

Uses the forward stack and the whole-face decoder from ../faces/results/nets.pt.  Trains one
more decoder, the same architecture, by the same local mismatch rule but with noise added to
every level's input code during training (a denoising autoencoder per level), saved in
results/dae.pt.

One cycle, for a request c_req (a top code):
    c_in   = c + s_k * std(c) * noise                 s_k falls from S to 0 over the K cycles
    image  = render(c_in)                             each level cleans as it renders (dae) or not (plain)
    c_read = forward(image)                           the stack reads its own render
    c     <- relu(c + alpha * (c_req - c_read))       correction at the top
With blending, each level's rendered map is mixed with the map the up pass read at that
level in the previous cycle: y <- (1 - beta) * y_rendered + beta * y_read.

Arms:
    plain            today's decoder, correction, no noise, no blend
    plain_blend      today's decoder, correction, no noise, blend
    dae              denoising decoder, correction, noise schedule
    dae_nocorr       denoising decoder, no correction, noise schedule (does cleaning alone help?)
    dae_blend        denoising decoder, correction, noise schedule, blend

Requests: the top codes of 500 test faces (so pixel error to the real face can be measured)
and eight faces that never existed (pca64 samples from the real codes' Gaussian).  Per cycle:
cosine between the request and the read-back code, pixel error to the real face, sharpness
(mean absolute Laplacian), and at the end the head's balanced accuracy on the renders against
the true attributes.

Usage: python run.py     # ~4 min the first time (trains the denoising decoder), ~1 min after
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
LOG = open(OUT / "run.log", "w")


def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


raw = np.load(DATA / "gray_218x178.npy", mmap_mode="r")
imgs = torch.from_numpy(np.ascontiguousarray(raw[:, 29:189, 25:153])).to(dev)
attrs = torch.from_numpy(np.load(DATA / "attrs_40.npy")).float().to(dev)
N = len(imgs); idx_test = torch.arange(N - 5000, N, device=dev); idx_train = torch.arange(0, N - 7000, device=dev)


def images(idx):
    return imgs[idx].float().div(255).sub(MEAN).div(STD).unsqueeze(1)


def show(x):
    return (x * STD + MEAN).clamp(0, 1)


def batches(idx, bs, shuffle=True):
    order = idx[torch.randperm(len(idx), device=dev)] if shuffle else idx
    for i in range(0, len(order), bs):
        yield order[i:i + bs]


class Forward(nn.Module):
    def __init__(s):
        super().__init__()
        cins = (1,) + CH[:-1]
        s.convs = nn.ModuleList([nn.Conv2d(ci, co, 3, padding=1) for ci, co in zip(cins, CH)])
        s.head = nn.Linear(D, 40)

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

    def render(s, y, read=None, beta=0.0):
        """Top down.  With read (the maps of the previous up pass) and beta, blend at every level."""
        for l in range(len(CH), 0, -1):
            y = s.down(l, s.unpool(l, y))
            if read is not None and l > 1:
                y = (1 - beta) * y + beta * read[l - 1]
        return y


def train_dae(dec, fwd, epochs):
    """Local mismatch with noise on every level's input code: unpool from a noisy y_l toward the clean
    pre-pool map, transposed conv toward the clean map below."""
    opt = torch.optim.Adam(dec.parameters(), 1e-3)
    for ep in range(epochs):
        tot = 0.0; nb = 0
        for b in batches(idx_train, 64):
            with torch.no_grad():
                ys, As = fwd.maps(images(b))
            sigma = float(np.random.uniform(0, 1))
            losses = []
            for lv in range(1, len(CH) + 1):
                y_noisy = F.relu(ys[lv] + sigma * ys[lv].std() * torch.randn_like(ys[lv]))
                a_hat = dec.unpool(lv, y_noisy); losses.append(F.mse_loss(a_hat, As[lv - 1]))
                losses.append(F.mse_loss(dec.down(lv, a_hat.detach()), ys[lv - 1]))
            opt.zero_grad(set_to_none=True); sum(losses).backward(); opt.step()
            tot += losses[1].item(); nb += 1
        log(f"    ep {ep + 1}  pixel mismatch {tot / nb:.4f}")


LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=dev).view(1, 1, 3, 3)


def sharpness(x):
    return F.conv2d(x, LAP).abs().mean((1, 2, 3))


@torch.no_grad()
def balanced_accuracy(fwd, x, y):
    out = fwd.head(fwd.maps(x)[0][-1].flatten(1)); pred = (out > 0).float()
    tp = (pred * y).sum(0); fp = (pred * (1 - y)).sum(0); tn = ((1 - pred) * (1 - y)).sum(0); fn = ((1 - pred) * y).sum(0)
    return ((tp / (tp + fn).clamp_min(1) + tn / (tn + fp).clamp_min(1)) / 2).mean().item()


@torch.no_grad()
def run_loop(fwd, dec, c_req, K, alpha, S, beta, x_real=None, keep=()):
    """Returns per-cycle metrics (lists of length K+1) and the kept renders."""
    c = c_req.clone(); read = None; kept = {}
    cos, err, shp = [], [], []
    for k in range(K + 1):
        s_k = S * (1 - k / K) if k > 0 else 0.0
        c_in = F.relu(c + s_k * c.std() * torch.randn_like(c)) if s_k > 0 else c
        x = dec.render(c_in, read if beta > 0 else None, beta)
        ys, _ = fwd.maps(x); c_read = ys[-1]; read = ys
        cos.append(F.cosine_similarity(c_req.flatten(1), c_read.flatten(1), dim=1).mean().item())
        shp.append(sharpness(show(x)).mean().item())
        if x_real is not None: err.append(F.mse_loss(show(x), show(x_real), reduction="none").mean((1, 2, 3)).mean().item())
        if k in keep: kept[k] = x.clone()
        c = F.relu(c + alpha * (c_req - c_read))
    return {"cos": cos, "err": err, "sharp": shp}, kept, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10); ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--S", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=0.5); ap.add_argument("--n_eval", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    st = torch.load(NETS, map_location=dev)
    fwd, plain = Forward().to(dev), Decoder().to(dev)
    fwd.load_state_dict(st["fwd"]); plain.load_state_dict(st["whole"])
    fwd.requires_grad_(False); fwd.eval(); plain.requires_grad_(False); plain.eval()
    dae = Decoder().to(dev); ck = OUT / "dae.pt"
    if ck.exists():
        dae.load_state_dict(torch.load(ck, map_location=dev)); log("loaded the denoising decoder")
    else:
        log("denoising decoder, by mismatch with noise on every level's input:"); torch.manual_seed(a.seed + 5); train_dae(dae, fwd, 4); torch.save(dae.state_dict(), ck)
        log(f"trained and saved  ({time.time() - t0:.0f}s)")
    dae.requires_grad_(False); dae.eval()

    arms = {"plain": (plain, a.alpha, 0.0, 0.0), "plain_blend": (plain, a.alpha, 0.0, a.beta),
            "dae": (dae, a.alpha, a.S, 0.0), "dae_nocorr": (dae, 0.0, a.S, 0.0), "dae_blend": (dae, a.alpha, a.S, a.beta)}
    keep = (0, 2, 5, a.K)
    res, pics = {}, {}
    ev = idx_test[:a.n_eval]
    with torch.no_grad():
        # the stack's own ceiling: its head on the real faces
        res["head_on_real"] = np.mean([balanced_accuracy(fwd, images(b), attrs[b]) for b in batches(ev, 100, False)])
        # pca64 samples for imagined requests
        Y = torch.cat([fwd.maps(images(b))[0][-1].flatten(1) for b in batches(idx_train, 250, False)])
        mu = Y.mean(0); U, Sv, V = torch.pca_lowrank(Y - mu, q=64, center=False); scale = Sv / (len(Y) - 1) ** 0.5
        g = torch.Generator(device=dev).manual_seed(a.seed)
        c_imag = F.relu(mu + (torch.randn(8, 64, device=dev, generator=g) * scale) @ V.T).view(-1, CH[-1], 5, 4)
        del Y, U
        for name, (dec, alpha, S, beta) in arms.items():
            acc = {"cos": np.zeros(a.K + 1), "err": np.zeros(a.K + 1), "sharp": np.zeros(a.K + 1)}; nb = 0; head = 0.0
            for b in batches(ev, 100, False):
                x = images(b); c_req = fwd.maps(x)[0][-1]
                m, kept, x_last = run_loop(fwd, dec, c_req, a.K, alpha, S, beta, x_real=x, keep=keep if b[0] == ev[0] else ())
                for k_ in acc: acc[k_] += np.array(m[k_])
                nb += 1; head += balanced_accuracy(fwd, x_last, attrs[b])
                if b[0] == ev[0]: pics[name] = {k: show(v[:6])[:, 0].cpu() for k, v in kept.items()}; pics["real"] = show(x[:6])[:, 0].cpu()
            res[name] = {k_: (v / nb).tolist() for k_, v in acc.items()}; res[name]["head_on_final"] = head / nb
            mi, kept_i, _ = run_loop(fwd, dec, c_imag, a.K, alpha, S, beta, keep=(0, a.K))
            res[name]["imagined"] = mi; pics[name]["imag0"] = show(kept_i[0])[:, 0].cpu(); pics[name]["imagK"] = show(kept_i[a.K])[:, 0].cpu()
            r = res[name]
            log(f"{name:12s} cos {r['cos'][0]:.3f} -> {r['cos'][-1]:.3f}   pixel error {r['err'][0]:.4f} -> {r['err'][-1]:.4f}   sharpness {r['sharp'][0]:.4f} -> {r['sharp'][-1]:.4f}   "
                f"head on final {r['head_on_final']:.3f}   imagined: cos {mi['cos'][0]:.3f} -> {mi['cos'][-1]:.3f} sharp {mi['sharp'][0]:.4f} -> {mi['sharp'][-1]:.4f}  ({time.time() - t0:.0f}s)")
    figures(res, pics, a.K, keep)
    L = ["# Results", "", f"{a.n_eval} test faces as requests, {a.K} cycles, alpha {a.alpha}, noise start {a.S}, blend {a.beta}.  The head reads real faces at {res['head_on_real']:.3f}.", "",
         "| arm | code agreement 0 -> K | pixel error 0 -> K | sharpness 0 -> K | head on final render | imagined: agreement 0 -> K | imagined: sharpness 0 -> K |",
         "|---|---|---|---|---|---|---|"]
    for name in arms:
        r = res[name]; mi = r["imagined"]
        L.append(f"| {name} | {r['cos'][0]:.3f} -> {r['cos'][-1]:.3f} | {r['err'][0]:.4f} -> {r['err'][-1]:.4f} | {r['sharp'][0]:.4f} -> {r['sharp'][-1]:.4f} | {r['head_on_final']:.3f} "
                 f"| {mi['cos'][0]:.3f} -> {mi['cos'][-1]:.3f} | {mi['sharp'][0]:.4f} -> {mi['sharp'][-1]:.4f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n"); log("\n".join(L))
    json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


def figures(res, pics, K, keep):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = [k for k in res if isinstance(res[k], dict) and "cos" in res[k]]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for i, (key, title) in enumerate([("cos", "code agreement with the request"), ("err", "pixel error to the real face"), ("sharp", "sharpness of the render")]):
        for name in arms: ax[i].plot(res[name][key], marker="o", ms=3, label=name)
        ax[i].set_title(title, fontsize=9); ax[i].set_xlabel("cycle")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "cycles.png", dpi=130); plt.close(fig)

    cols = [("real", None)] + [(f"{name}\ncycle {k}", (name, k)) for name in ("plain", "dae") for k in keep]
    fig, ax = plt.subplots(6, len(cols), figsize=(len(cols) * 1.25, 9.5))
    for i in range(6):
        for j, (lab, key) in enumerate(cols):
            img = pics["real"][i] if key is None else pics[key[0]][key[1]][i]
            ax[i, j].imshow(img, cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    for j, (lab, key) in enumerate(cols): ax[0, j].set_title(lab, fontsize=7)
    fig.suptitle("real-face requests through the loop: today's decoder (plain) and the denoising one (dae)", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "loop_faces.png", dpi=130); plt.close(fig)

    rows = [(f"{name}\ncycle {k}", name, kk) for name in ("plain", "dae", "dae_blend") for k, kk in ((0, "imag0"), (K, "imagK"))]
    fig, ax = plt.subplots(len(rows), 8, figsize=(10, len(rows) * 1.5))
    for i, (lab, name, kk) in enumerate(rows):
        for j in range(8):
            ax[i, j].imshow(pics[name][kk][j], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("imagined requests (pca64 samples) at the first and the last cycle", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "loop_imagined.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
