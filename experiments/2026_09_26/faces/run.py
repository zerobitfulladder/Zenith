"""Faces: a deeper stack trained on the 40 attributes, a backward path from mismatch, and
composition by attending a part.

Grayscale CelebA, the 160x128 centre crop of the original 218x178 (not scaled), from the
cache the 2026-09-23 experiment built.  Five levels of conv3x3 + ReLU + 2x2 max pool with 16,
32, 64, 128, 256 channels, so the top map is 5x4x256: 256 features at 20 places, each place
a 32x32 patch.  The forward stack is trained on the attributes (balanced BCE) and frozen.

Two decoders, both trained only by local mismatch from the frozen stack (learned unpool
toward the pre-pool map, transposed conv toward the map below, nothing crossing a level):
    whole    on the full crop
    mouth    on the mouth window (rows 88-168, cols 32-96, zoomed twice to 160x128)

Checks:
    recon      render from a face's own top map: pixel error, and the stack's own attribute
               head read on the render against the true attributes
    concept    render the mean top map of faces with and without an attribute; the head's
               probability of that attribute on each render
    part       pixel error inside the mouth region: whole render against the mouth-window
               render pasted back (does attending a part sharpen it, on faces?)
    compose    woman + mustache, three ways, each read by the head for P(Male) and
               P(Mustache):
                 arith    woman's mean top map + (men with mustache - men without), whole
                 grid     the same difference added only at the two top-map cells over the
                          upper lip
                 window   render the woman whole, then attend the mouth window and render
                          (woman's mouth code + the mustache difference in mouth-window
                          codes) with the mouth decoder, paste it back
                 replace  as window, but a mustached man's mouth code outright
    locate     the same edit on real women shifted by up to 24 rows and 20 columns, so that
               the mouth is not where it is in aligned faces.  A mouth signature (the mean
               feature vector at the mouth cells over training faces) is matched against every
               cell of the shifted face's own top map; the best cell is the mouth; the
               mustache difference is added there.  Against adding it at the fixed cells.

Usage: python run.py     # ~6 min on a laptop GPU; trained nets are saved in results/*.pt and reused
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data/celeba"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
dev = "cuda"
torch.backends.cudnn.benchmark = True
H, W = 160, 128
CH = (16, 32, 64, 128, 256)
MEAN, STD = 0.45, 0.25
MOUTH = (80, 32, 80, 64)                 # top, left, height, width: the lower half of the face; zooms exactly 2x to 160x128
JITTER = (24, 20)                        # locate test: faces shifted by up to this many rows and columns
MOUTH_CELLS = [(3, 1), (3, 2)]           # top-map cells (32x32 pixels each) over the upper lip
SHOW = ["Male", "Smiling", "Eyeglasses", "Mustache", "Blond_Hair", "Wearing_Hat", "Bald", "Young", "Wearing_Lipstick"]
LOG = open(OUT / "run.log", "w")


def log(s):
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


# ---------------------------------------------------------------- data
raw = np.load(DATA / "gray_218x178.npy", mmap_mode="r")
imgs = torch.from_numpy(np.ascontiguousarray(raw[:, 29:189, 25:153])).to(dev)      # uint8 N,160,128
attrs = torch.from_numpy(np.load(DATA / "attrs_40.npy")).float().to(dev)
names = json.loads((DATA / "attr_names_40.json").read_text()); A = {n: i for i, n in enumerate(names)}
N = len(imgs); n_test, n_val = 5000, 2000
idx_test = torch.arange(N - n_test, N, device=dev)
idx_train = torch.arange(0, N - n_test - n_val, device=dev)
pos = attrs[idx_train].mean(0); pos_weight = (1 - pos) / pos.clamp_min(1e-3)


def images(idx):
    return imgs[idx].float().div(255).sub(MEAN).div(STD).unsqueeze(1)


def show(x):
    return (x * STD + MEAN).clamp(0, 1)


def batches(idx, bs, shuffle=True):
    order = idx[torch.randperm(len(idx), device=dev)] if shuffle else idx
    for i in range(0, len(order), bs):
        yield order[i:i + bs]


def zoom(x, t, l, h, w):
    return F.interpolate(x[:, :, t:t + h, l:l + w], size=(H, W), mode="bilinear", align_corners=False)


def unzoom(r, h, w):
    return F.interpolate(r, size=(h, w), mode="area")


# ---------------------------------------------------------------- networks
class Forward(nn.Module):
    def __init__(s):
        super().__init__()
        cins = (1,) + CH[:-1]
        s.convs = nn.ModuleList([nn.Conv2d(ci, co, 3, padding=1) for ci, co in zip(cins, CH)])
        s.head = nn.Linear(CH[-1] * 20, 40)

    def forward(s, x):
        ys, As, h = [x], [], x
        for conv in s.convs:
            a = F.relu(conv(h)); h = F.max_pool2d(a, 2); As.append(a); ys.append(h)
        return ys, As

    def logits(s, x):
        return s.head(s(x)[0][-1].flatten(1))


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


# ---------------------------------------------------------------- training
@torch.no_grad()
def balanced_accuracy(logits_fn, idx, source=images):
    tp = fp = tn = fn = torch.zeros(40, device=dev)
    for b in batches(idx, 250, False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = logits_fn(source(b))
        pred = (out.float() > 0).float(); y = attrs[b]
        tp = tp + (pred * y).sum(0); fp = fp + (pred * (1 - y)).sum(0)
        tn = tn + ((1 - pred) * (1 - y)).sum(0); fn = fn + ((1 - pred) * y).sum(0)
    return ((tp / (tp + fn).clamp_min(1) + tn / (tn + fp).clamp_min(1)) / 2)


def train_forward(fwd, epochs):
    opt = torch.optim.AdamW(fwd.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * (len(idx_train) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.15)
    for ep in range(epochs):
        for b in batches(idx_train, 128):
            if len(b) < 128: continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fwd.logits(images(b))
            loss = F.binary_cross_entropy_with_logits(out.float(), attrs[b], pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        log(f"    ep {ep + 1}  test balanced accuracy {balanced_accuracy(fwd.logits, idx_test).mean():.4f}")


def train_decoder(dec, fwd, epochs, window=None):
    opt = torch.optim.Adam(dec.parameters(), 1e-3)
    for ep in range(epochs):
        tot = torch.zeros(len(CH), device=dev); nb = 0
        for b in batches(idx_train, 64):
            x = images(b)
            if window is not None:
                h, w = window
                t, l = np.random.randint(0, H - h + 1), np.random.randint(0, W - w + 1)
                x = zoom(x, t, l, h, w)
            with torch.no_grad():
                ys, As = fwd(x)
            losses = []
            for lv in range(1, len(CH) + 1):
                a_hat = dec.unpool(lv, ys[lv])
                losses.append(F.mse_loss(a_hat, As[lv - 1]))
                losses.append(F.mse_loss(dec.down(lv, a_hat.detach()), ys[lv - 1]))
            opt.zero_grad(set_to_none=True); sum(losses).backward(); opt.step()
            tot += torch.stack([v.detach() for v in losses[1::2]]); nb += 1
        log(f"    ep {ep + 1}  mismatch pixels {tot[0] / nb:.4f}  L1 {tot[1] / nb:.4f}  L2 {tot[2] / nb:.4f}  L3 {tot[3] / nb:.4f}  L4 {tot[4] / nb:.4f}")


@torch.no_grad()
def top_maps(fwd, idx, window=None):
    out = []
    for b in batches(idx, 250, False):
        x = images(b)
        if window is not None:
            t, l, h, w = window; x = zoom(x, t, l, h, w)
        out.append(fwd(x)[0][-1].half())
    return torch.cat(out)


def mean_where(Y, mask):
    return Y[mask].float().mean(0, keepdim=True)


@torch.no_grad()
def probs(fwd, x):
    return torch.sigmoid(fwd.logits(x))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_fwd", type=int, default=6); ap.add_argument("--epochs_dec", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    log(f"{len(idx_train)} training faces, {len(idx_test)} test faces, crop {H}x{W}, {dev}")

    ck = OUT / "nets.pt"
    fwd = Forward().to(dev); torch.manual_seed(a.seed + 1); whole = Decoder().to(dev); torch.manual_seed(a.seed + 2); mouth = Decoder().to(dev)
    if ck.exists():
        st = torch.load(ck, map_location=dev); fwd.load_state_dict(st["fwd"]); whole.load_state_dict(st["whole"]); mouth.load_state_dict(st["mouth"])
        log(f"loaded trained nets from {ck.name}")
    else:
        log("forward, on the 40 attributes:")
        train_forward(fwd, a.epochs_fwd)
        fwd.requires_grad_(False); fwd.eval()
        log("whole decoder, by mismatch:"); train_decoder(whole, fwd, a.epochs_dec)
        log("mouth-window decoder, by mismatch:"); train_decoder(mouth, fwd, a.epochs_dec, window=MOUTH[2:])
        torch.save({"fwd": fwd.state_dict(), "whole": whole.state_dict(), "mouth": mouth.state_dict()}, ck)
        log(f"trained and saved  ({time.time() - t0:.0f}s)")
    fwd.requires_grad_(False); fwd.eval(); whole.requires_grad_(False); mouth.requires_grad_(False)
    bal_real = balanced_accuracy(fwd.logits, idx_test)
    log(f"forward: {bal_real.mean():.4f} balanced accuracy on real test faces")

    res, pics = {}, {}
    mt, ml, mh, mw = MOUTH
    with torch.no_grad():
        # recon: pixel error, own-head reading, part sharpness
        mse = mse_part_whole = mse_part_mouth = 0.0
        for b in batches(idx_test, 250, False):
            x = images(b); r = whole.render(fwd(x)[0][-1])
            mse += F.mse_loss(show(r), show(x), reduction="sum").item() / (H * W)
            rm = unzoom(mouth.render(fwd(zoom(x, mt, ml, mh, mw))[0][-1]), mh, mw)
            region = lambda z: show(z[:, :, mt:mt + mh, ml:ml + mw])
            mse_part_whole += F.mse_loss(region(r), region(x), reduction="sum").item() / (mh * mw)
            mse_part_mouth += F.mse_loss(show(rm), region(x), reduction="sum").item() / (mh * mw)
            if b[0] == idx_test[0]:
                pics["orig"] = show(x[:8]).cpu(); pics["recon"] = show(r[:8]).cpu()
                comp = r[:8].clone(); comp[:, :, mt:mt + mh, ml:ml + mw] = rm[:8]; pics["recon_mouth"] = show(comp).cpu()
        n = len(idx_test)
        res["recon_mse"] = mse / n; res["part_mse_whole"] = mse_part_whole / n; res["part_mse_mouth"] = mse_part_mouth / n
        bal_render = balanced_accuracy(fwd.logits, idx_test, source=lambda b: whole.render(fwd(images(b))[0][-1]))
        res["bal_real"] = bal_real.mean().item(); res["bal_render"] = bal_render.mean().item()
        res["bal_per_attr"] = {nm: [float(bal_real[i]), float(bal_render[i])] for i, nm in enumerate(names)}
        log(f"recon: pixel error {res['recon_mse']:.4f}; own head on renders {res['bal_render']:.4f} (real {res['bal_real']:.4f}); "
            f"mouth region error whole {res['part_mse_whole']:.4f} vs attended {res['part_mse_mouth']:.4f}  ({time.time() - t0:.0f}s)")

        # concept renders
        Y = top_maps(fwd, idx_train); At = attrs[idx_train]
        res["concept"] = {}; pics["concept"] = {}
        for nm in SHOW:
            k = A[nm]; row = {}
            for v in (1, 0):
                r = whole.render(mean_where(Y, At[:, k] == v)); p = probs(fwd, r)[0]
                row[v] = {"p_attr": float(p[k]), "p_male": float(p[A["Male"]])}; pics["concept"][(nm, v)] = show(r)[0, 0].cpu()
            res["concept"][nm] = row
            log(f"concept {nm:17s} present: P({nm})={row[1]['p_attr']:.2f}   absent: P({nm})={row[0]['p_attr']:.2f}")

        # composition: woman + mustache
        male, must = At[:, A["Male"]] == 1, At[:, A["Mustache"]] == 1
        woman = mean_where(Y, ~male); man_m = mean_where(Y, male & must); man_n = mean_where(Y, male & ~must)
        delta = man_m - man_n
        grid = woman.clone()
        for (rr, cc) in MOUTH_CELLS: grid[:, :, rr, cc] += delta[:, :, rr, cc]
        Ym = top_maps(fwd, idx_train, window=MOUTH)
        wwoman = mean_where(Ym, ~male); wdelta = mean_where(Ym, male & must) - mean_where(Ym, male & ~must); wman_m = mean_where(Ym, male & must)
        base = whole.render(woman)
        def paste(code):
            c = base.clone(); c[:, :, mt:mt + mh, ml:ml + mw] = unzoom(mouth.render(code), mh, mw); return c
        cases = {"woman": base, "man + mustache": whole.render(man_m), "arith": whole.render(woman + delta),
                 "arith x3": whole.render(woman + 3 * delta), "grid": whole.render(grid),
                 "window": paste(wwoman + wdelta), "window x3": paste(wwoman + 3 * wdelta), "replace": paste(wman_m)}
        res["compose"] = {}; pics["compose"] = {}
        for nm, r in cases.items():
            p = probs(fwd, r)[0]
            res["compose"][nm] = {"p_male": float(p[A["Male"]]), "p_mustache": float(p[A["Mustache"]]), "p_no_beard": float(p[A["No_Beard"]])}
            pics["compose"][nm] = show(r)[0, 0].cpu()
            log(f"compose {nm:16s} P(Male)={p[A['Male']]:.2f}  P(Mustache)={p[A['Mustache']]:.2f}  P(No_Beard)={p[A['No_Beard']]:.2f}")
        # reference: the head on real faces
        pr = torch.cat([probs(fwd, images(b)) for b in batches(idx_test, 250, False)]); Ate = attrs[idx_test]
        wm, mm = Ate[:, A["Male"]] == 0, (Ate[:, A["Male"]] == 1) & (Ate[:, A["Mustache"]] == 1)
        res["compose"]["real women"] = {"p_male": float(pr[wm, A["Male"]].mean()), "p_mustache": float(pr[wm, A["Mustache"]].mean())}
        res["compose"]["real men w/ mustache"] = {"p_male": float(pr[mm, A["Male"]].mean()), "p_mustache": float(pr[mm, A["Mustache"]].mean())}
        log(f"real women: P(Male)={res['compose']['real women']['p_male']:.2f} P(Mustache)={res['compose']['real women']['p_mustache']:.2f};  "
            f"real men with mustache: P(Male)={res['compose']['real men w/ mustache']['p_male']:.2f} P(Mustache)={res['compose']['real men w/ mustache']['p_mustache']:.2f}")

        # locate: real women shifted in one direction at a time; the mouth found from the face's own map by
        # matching a mouth signature (the mean feature vector at the mouth cells) against every cell.
        # The head is not read on shifted faces: it was trained on aligned ones and reads absolute position.
        sig = torch.stack([Y[:, :, r, c].float() for (r, c) in MOUTH_CELLS], 0).mean((0, 1))          # (256,)
        d_left, d_right = delta[0, :, MOUTH_CELLS[0][0], MOUTH_CELLS[0][1]], delta[0, :, MOUTH_CELLS[1][0], MOUTH_CELLS[1][1]]
        women_te = idx_test[attrs[idx_test, A["Male"]] == 0][:300]
        full = torch.from_numpy(np.stack([raw[i] for i in women_te.tolist()])).to(dev)              # uint8 n,218,178
        res["locate"] = {}; pics["locate"] = {}
        for dy, dx in [(0, 0), (0, -JITTER[1]), (0, JITTER[1]), (-JITTER[0], 0), (JITTER[0], 0)]:
            xs = full[:, 29 + dy:29 + dy + H, 25 + dx:25 + dx + W].float().div(255).sub(MEAN).div(STD).unsqueeze(1)
            y = fwd(xs)[0][-1]
            cos = F.cosine_similarity(y, sig.view(1, -1, 1, 1).expand_as(y), dim=1)                 # (n, 5, 4)
            best = cos.flatten(1).argmax(1); rr, cc = best // 4, best % 4
            exp_r, exp_c = (125 - dy) // 32, (64 - dx) // 32                                        # the cell holding the mouth line
            row_hit = (rr == exp_r).float().mean().item(); cell_hit = ((rr == exp_r) & ((cc - exp_c).abs() <= 1)).float().mean().item()
            res["locate"][f"dy={dy:+d} dx={dx:+d}"] = {"expected_row": exp_r, "row_hit": row_hit, "cell_hit": cell_hit,
                                                       "found_rows": torch.bincount(rr, minlength=5).tolist()}
            log(f"locate dy={dy:+3d} dx={dx:+3d}: mouth row {exp_r}, found row in {row_hit:.2f}, cell within one of {cell_hit:.2f}; rows found {torch.bincount(rr, minlength=5).tolist()}")
            if (dy, dx) == (0, JITTER[1]):
                n_ = torch.arange(6, device=dev); y6 = y[:6]
                c_l = torch.where(cc[:6] < 3, cc[:6], cc[:6] - 1)
                y_fixed = y6.clone(); y_found = y6.clone()
                for (r0, c0), d in zip(MOUTH_CELLS, (d_left, d_right)): y_fixed[:, :, r0, c0] += 3 * d
                y_found[n_, :, rr[:6], c_l] += 3 * d_left; y_found[n_, :, rr[:6], (c_l + 1).clamp(0, 3)] += 3 * d_right
                pics["locate"]["shifted original"] = show(xs[:6])[:, 0].cpu()
                pics["locate"]["plain render"] = show(whole.render(y6))[:, 0].cpu()
                pics["locate"]["mustache x3 at fixed cells"] = show(whole.render(y_fixed))[:, 0].cpu()
                pics["locate"]["mustache x3 at located cells"] = show(whole.render(y_found))[:, 0].cpu()
                pics["locate_cells"] = [(int(rr[i]), int(c_l[i])) for i in range(6)]

    figures(pics)
    summary(res)
    json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s")


def figures(pics):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    mt, ml, mh, mw = MOUTH
    fig, ax = plt.subplots(3, 8, figsize=(10, 5))
    for j in range(8):
        for i, k in enumerate(("orig", "recon", "recon_mouth")):
            ax[i, j].imshow(pics[k][j, 0], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[2, j].add_patch(Rectangle((ml - 0.5, mt - 0.5), mw, mh, fill=False, edgecolor="tab:orange", lw=0.8))
    for i, t in enumerate(["original", "render from the top map", "mouth window attended and pasted"]):
        ax[i, 0].set_ylabel(t, rotation=0, ha="right", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "recon.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(2, len(SHOW), figsize=(len(SHOW) * 1.3, 3.6))
    for j, nm in enumerate(SHOW):
        for i, v in enumerate((1, 0)):
            ax[i, j].imshow(pics["concept"][(nm, v)], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[0, j].set_title(nm.replace("_", " "), fontsize=7)
    ax[0, 0].set_ylabel("present", rotation=0, ha="right", va="center", fontsize=7); ax[1, 0].set_ylabel("absent", rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("the mean top map of faces with and without an attribute, rendered", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "concepts.png", dpi=130); plt.close(fig)

    keys = list(pics["compose"])
    fig, ax = plt.subplots(2, 4, figsize=(7, 4.8))
    for j, nm in enumerate(keys):
        a_ = ax[j // 4, j % 4]
        a_.imshow(pics["compose"][nm], cmap="gray", vmin=0, vmax=1); a_.set_xticks([]); a_.set_yticks([]); a_.set_title(nm, fontsize=8)
        if nm.startswith(("window", "replace")): a_.add_patch(Rectangle((ml - 0.5, mt - 0.5), mw, mh, fill=False, edgecolor="tab:orange", lw=0.8))
    fig.suptitle("woman + mustache: whole-code arithmetic, the difference at two grid cells, the mouth window attended", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "compose.png", dpi=130); plt.close(fig)

    keys = ["shifted original", "plain render", "mustache x3 at fixed cells", "mustache x3 at located cells"]
    fig, ax = plt.subplots(6, 4, figsize=(6, 10))
    for i in range(6):
        for j, k in enumerate(keys):
            ax[i, j].imshow(pics["locate"][k][i], cmap="gray", vmin=0, vmax=1); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if k == "mustache x3 at fixed cells":
                ax[i, j].add_patch(Rectangle((MOUTH_CELLS[0][1] * 32 - 0.5, MOUTH_CELLS[0][0] * 32 - 0.5), 64, 32, fill=False, edgecolor="tab:red", lw=0.8))
            if k == "mustache x3 at located cells":
                r0, c0 = pics["locate_cells"][i]
                ax[i, j].add_patch(Rectangle((c0 * 32 - 0.5, r0 * 32 - 0.5), 64, 32, fill=False, edgecolor="tab:orange", lw=0.8))
    for j, k in enumerate(keys): ax[0, j].set_title(k, fontsize=7)
    fig.suptitle("women shifted 20 columns: the mustache difference added at the fixed cells (red) or at the cells the face's own map says are the mouth (orange)", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "locate.png", dpi=130); plt.close(fig)


def summary(res):
    L = ["# Results", "", f"Forward stack: {res['bal_real']:.4f} balanced accuracy on real test faces (40 attributes, mean).", "",
         "## Reconstruction from the top map alone", "",
         f"- pixel error {res['recon_mse']:.4f}",
         f"- the stack's own head on the renders: {res['bal_render']:.4f} balanced accuracy against the true attributes (real faces {res['bal_real']:.4f})",
         f"- mouth region pixel error: whole render {res['part_mse_whole']:.4f}, mouth window attended and rendered {res['part_mse_mouth']:.4f}", "",
         "## Concept renders: the head's probability of the attribute on the render", "",
         "| attribute | render of faces with it | render of faces without it |", "|---|---|---|"]
    for nm, row in res["concept"].items():
        L.append(f"| {nm} | {row[1]['p_attr']:.2f} | {row[0]['p_attr']:.2f} |")
    L += ["", "## Woman + mustache", "", "| render | P(Male) | P(Mustache) |", "|---|---|---|"]
    for nm, row in res["compose"].items():
        L.append(f"| {nm} | {row['p_male']:.2f} | {row['p_mustache']:.2f} |")
    L += ["", "## Finding the mouth from the face's own map (300 test women, shifted)", "",
          "Cosine of a mouth signature (mean feature vector at the mouth cells of aligned training faces) against every top-map cell; the best cell is the mouth.",
          "row hit = the found cell is in the row holding the mouth line; cell hit = and within one column.  found rows = how many of the 300 landed in each of the five rows.", "",
          "| shift | mouth row | row hit | cell hit | found rows |", "|---|---|---|---|---|"]
    for nm, row in res["locate"].items():
        L.append(f"| {nm} | {row['expected_row']} | {row['row_hit']:.2f} | {row['cell_hit']:.2f} | {row['found_rows']} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n"); log("\n".join(L))


if __name__ == "__main__":
    main()
