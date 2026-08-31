"""Every figure in this folder. `python run_place_code.py`, ~20 s."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from place_code import PlaceCode, relu_split, unit_rows, patches_of

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
CMAP = "magma"
plt.rcParams.update({"figure.dpi": 130, "font.size": 8})


def digits(n=64):
    X = np.load(ROOT / "data/mnist/digits/train_images.npy").astype(np.float64)
    if X.ndim == 2:
        X = X.reshape(-1, 28, 28)
    if X.max() > 1.5:
        X = X / 255.0
    return X[:n]


# ------------------------------------------------------------------ fig 1
def fig_one_channel():
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.6))

    # (a) one value -> one bump, and how it slides
    pc = PlaceCode(1, nb=8, lo=0.0, hi=1.0, halfw=1.5)
    for v, c in zip([0.12, 0.37, 0.62], ["#1b6ca8", "#c1462d", "#2f8f4e"]):
        w = pc.encode([[v]])[0]
        ax[0, 0].plot(pc.centers[0], w, "o-", color=c, label=f"value {v}")
        ax[0, 0].axvline(v, color=c, ls=":", lw=0.8)
    ax[0, 0].set_title("one channel, nb=8: a value lights the cells near it")
    ax[0, 0].set_xlabel("what each cell stands for"); ax[0, 0].set_ylabel("activation")
    ax[0, 0].legend(fontsize=7); ax[0, 0].set_ylim(-0.05, 1.05)

    # (b,c,d) the nb dial: code over the whole value range
    grid = np.linspace(0, 1, 400)[:, None]
    for k, nb in enumerate([2, 4, 8, 16]):
        a = ax.flat[1 + k] if k < 2 else ax[1, k - 2]
        p = PlaceCode(1, nb=nb, lo=0.0, hi=1.0, halfw=1.5)
        C = p.encode(grid)
        a.imshow(C.T, aspect="auto", origin="lower", cmap=CMAP,
                 extent=[0, 1, -0.5, nb - 0.5], vmin=0, vmax=1)
        a.set_title(f"nb={nb}" + ("   (= the ON/OFF split)" if nb == 2 else ""))
        a.set_xlabel("value in"); a.set_ylabel("cell")
        a.set_yticks(range(nb) if nb <= 8 else range(0, nb, 4))

    # (e) similarity kernel: how fast does the code stop overlapping
    a = ax[1, 1]
    a.clear()
    d = np.linspace(0, 0.6, 300)
    for hw, c in zip([0.75, 1.5, 3.0], ["#1b6ca8", "#c1462d", "#2f8f4e"]):
        p = PlaceCode(1, nb=16, lo=0.0, hi=1.0, halfw=hw)
        A, _ = unit_rows(p.encode(np.full((len(d), 1), 0.5)))
        B, _ = unit_rows(p.encode((0.5 + d)[:, None]))
        a.plot(d, (A * B).sum(1), color=c, label=f"halfw={hw}")
    a.set_title("similarity vs difference in value  (nb=16)")
    a.set_xlabel("|a - b|"); a.set_ylabel("cosine of the two codes")
    a.legend(fontsize=7); a.grid(alpha=0.25)

    # (f) is the norm really constant? measure the ripple
    a = ax[1, 2]
    a.clear()
    g = np.linspace(0, 1, 500)[:, None]
    rip = {}
    for hw, c in zip([1.0, 1.5, 3.0, 6.0], ["#999", "#1b6ca8", "#c1462d", "#2f8f4e"]):
        p = PlaceCode(1, nb=16, lo=0.0, hi=1.0, halfw=hw)
        n = np.linalg.norm(p.encode(g), axis=1)
        a.plot(g[:, 0], n / n.mean(), color=c, label=f"halfw={hw}")
        rip[hw] = float(n.max() / n.min())
    a.set_title("code length as the value slides (nb=16)")
    a.set_xlabel("value in"); a.set_ylabel("L2 norm / mean")
    a.legend(fontsize=7); a.grid(alpha=0.25)

    fig.tight_layout(); fig.savefig(OUT / "01_one_channel.png"); plt.close(fig)
    return rip


# ------------------------------------------------------------------ fig 2
def fig_baseline_vs_unknown():
    """The relu split and the 2-cell bump differ on one important case."""
    v = np.linspace(-1, 1, 400)[:, None]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))

    S = relu_split(v)
    ax[0].plot(v, S[:, 0], label="ON  = relu(v)", color="#c1462d")
    ax[0].plot(v, S[:, 1], label="OFF = relu(-v)", color="#1b6ca8")
    ax[0].set_title("your proposal: rectify into two images")

    p = PlaceCode(1, nb=2, lo=-1.0, hi=1.0, halfw=1.5)
    C = p.encode(v)
    ax[1].plot(v, C[:, 1], label="bright cell", color="#c1462d")
    ax[1].plot(v, C[:, 0], label="dark cell", color="#1b6ca8")
    ax[1].set_title("nb=2 bump: the same two cells, crossfaded")

    for a in ax[:2]:
        a.axvline(0, color="k", lw=0.8, ls=":")
        a.annotate("at baseline", (0, 1.02), ha="center", fontsize=7)
        a.set_xlabel("mean-centred pixel value"); a.set_ylabel("activation")
        a.legend(fontsize=7); a.set_ylim(-0.05, 1.15); a.grid(alpha=0.25)

    ax[2].bar([0, 1], [np.linalg.norm(relu_split([[0.0]])),
                       np.linalg.norm(p.encode([[0.0]]))],
              color=["#c1462d", "#1b6ca8"], width=0.55)
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["relu split", "nb=2 bump"])
    ax[2].set_ylabel("code length at baseline")
    ax[2].set_title("baseline vs 'no information'")
    ax[2].annotate("silent — same as\nan unwritten cell", (0, 0.04),
                   ha="center", fontsize=7)
    ax[2].annotate("still speaks —\ndistinguishable", (1, 0.75),
                   ha="center", fontsize=7, color="w")
    fig.tight_layout(); fig.savefig(OUT / "02_baseline_vs_unknown.png"); plt.close(fig)


# ------------------------------------------------------------------ fig 3
def fig_image_bands(X):
    img = X[0]
    fig, axes = plt.subplots(3, 9, figsize=(11, 4.2))
    for row, nb in enumerate([2, 4, 8]):
        pc = PlaceCode(784, nb=nb, lo=0.0, hi=1.0, halfw=1.5)
        P = pc.profiles(pc.encode(img.reshape(1, 784)))[0]     # (784, nb)
        axes[row, 0].imshow(img, cmap="gray_r"); axes[row, 0].set_ylabel(f"nb={nb}")
        axes[row, 0].set_title("the digit" if row == 0 else "", fontsize=8)
        for j in range(8):
            a = axes[row, j + 1]
            if j < nb:
                a.imshow(P[:, j].reshape(28, 28), cmap=CMAP, vmin=0, vmax=1)
                a.set_title(f"band {j}\n(={pc.centers[0, j]:.2f})", fontsize=6.5)
            else:
                a.axis("off")
        for a in axes[row]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("one image -> nb nonnegative planes, each 'how much is this "
                 "pixel in that intensity band'", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "03_image_bands.png"); plt.close(fig)


# ------------------------------------------------------------------ fig 4
def fig_reconstruction(X):
    NBS = [2, 3, 4, 6, 8, 12, 16, 24]
    err_pk, err_ct, spars = [], [], []
    recon = {}
    V = X[:32].reshape(32, 784)
    for nb in NBS:
        pc = PlaceCode(784, nb=nb, lo=0.0, hi=1.0, halfw=1.5)
        C = pc.encode(V)
        pk, ct = pc.decode(C, "peak"), pc.decode(C, "centroid")
        err_pk.append(float(np.sqrt(((pk - V) ** 2).mean())))
        err_ct.append(float(np.sqrt(((ct - V) ** 2).mean())))
        spars.append(float((C > 1e-6).mean()))
        recon[nb] = ct[0].reshape(28, 28)

    fig = plt.figure(figsize=(11, 3.2))
    gs = fig.add_gridspec(1, 11)
    a = fig.add_subplot(gs[0, :4])
    a.plot(NBS, err_pk, "o-", label="peak (top-1 cell)", color="#c1462d")
    a.plot(NBS, err_ct, "o-", label="centroid (sub-cell)", color="#1b6ca8")
    a.set_xlabel("nb — cells per pixel"); a.set_ylabel("RMSE, image -> code -> image")
    a.set_title("what intensity resolution costs"); a.legend(fontsize=7); a.grid(alpha=0.25)
    a.set_xscale("log"); a.set_xticks(NBS); a.set_xticklabels(NBS)
    b = fig.add_subplot(gs[0, 4:6])
    b.plot(NBS, spars, "o-", color="#2f8f4e"); b.set_xlabel("nb")
    b.set_ylabel("fraction of cells lit"); b.set_title("sparsity"); b.grid(alpha=0.25)
    b.set_xscale("log"); b.set_xticks(NBS); b.set_xticklabels(NBS)
    for k, nb in enumerate([2, 4, 8, 16, 24]):
        c = fig.add_subplot(gs[0, 6 + k])
        c.imshow(recon[nb], cmap="gray_r", vmin=0, vmax=1)
        c.set_title(f"nb={nb}", fontsize=7); c.set_xticks([]); c.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "04_reconstruction.png"); plt.close(fig)
    return dict(zip(map(str, NBS), zip(err_pk, err_ct, spars)))


# ------------------------------------------------------------------ fig 5
def fig_positive_template():
    """Centre dark, surround bright — with no negative weight anywhere."""
    nb = 8
    pc = PlaceCode(9, nb=nb, lo=0.0, hi=1.0, halfw=1.5)
    ideal = np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1]], float)      # dark centre
    tmpl, _ = unit_rows(pc.encode(ideal))

    tests = {"centre dark\n(what it wants)": [1, 1, 1, 1, 0, 1, 1, 1, 1],
             "centre bright\n(inverted)":    [0, 0, 0, 0, 1, 0, 0, 0, 0],
             "flat mid grey":                [.5] * 9,
             "all bright":                   [1] * 9,
             "all dark":                     [0] * 9}
    P, _ = unit_rows(pc.encode(np.array(list(tests.values()), float)))
    s = (P @ tmpl[0])

    fig = plt.figure(figsize=(11, 3.4))
    gs = fig.add_gridspec(2, 8, height_ratios=[1, 1.25])
    a = fig.add_subplot(gs[:, :3])
    im = a.imshow(pc.profiles(tmpl)[0], cmap=CMAP, aspect="auto", vmin=0)
    a.set_xlabel("cell  (left = dark, right = bright)"); a.set_ylabel("pixel of the 3x3")
    a.set_yticks(range(9)); a.set_yticklabels(
        [f"{i}{'  <- CENTRE' if i == 4 else ''}" for i in range(9)], fontsize=6.5)
    a.set_title("the template's weights — every one positive")
    fig.colorbar(im, ax=a, fraction=0.04)
    for k, (name, patch) in enumerate(tests.items()):
        c = fig.add_subplot(gs[0, 3 + k])
        c.imshow(np.array(patch).reshape(3, 3), cmap="gray_r", vmin=0, vmax=1)
        c.set_title(name, fontsize=6.5); c.set_xticks([]); c.set_yticks([])
    d = fig.add_subplot(gs[1, 3:])
    d.bar(range(5), s, color=["#2f8f4e"] + ["#8a8a8a"] * 4)
    d.set_xticks(range(5)); d.set_xticklabels([f"{v:.3f}" for v in s], fontsize=7)
    d.set_ylabel("score"); d.set_ylim(0, 1.05)
    d.set_title("it wins on the patch it wants, with no negative surround weight")
    fig.tight_layout(); fig.savefig(OUT / "05_positive_template.png"); plt.close(fig)
    return {k: float(v) for k, v in zip(tests, s)}


# ------------------------------------------------------------------ fig 6
def fig_motor_channel():
    t = np.linspace(0, 1, 300)
    thrust = 4.0 + 3.0 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-1.5 * t) + 0.6 * t
    pc = PlaceCode(1, nb=24, lo=0.0, hi=8.0, halfw=1.5)
    C = pc.encode(thrust[:, None])
    back = pc.decode(C, "centroid")[:, 0]
    peak = pc.decode(C, "peak")[:, 0]

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.0),
                           gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    ax[0].plot(t, thrust, color="#1b6ca8"); ax[0].set_title("a motor command over time")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("thrust"); ax[0].grid(alpha=0.25)
    ax[1].imshow(C.T, aspect="auto", origin="lower", cmap=CMAP,
                 extent=[0, 1, 0, 8], vmin=0, vmax=1)
    ax[1].set_title("its place code — one bump, sliding")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("what the cell stands for")
    ax[2].plot(t, thrust, color="k", lw=2.2, label="truth")
    ax[2].plot(t, peak, color="#c1462d", lw=0.9, label="read back: peak")
    ax[2].plot(t, back, color="#2f8f4e", lw=1.1, ls="--", label="read back: centroid")
    ax[2].set_title("and back out again"); ax[2].set_xlabel("t")
    ax[2].legend(fontsize=6.5); ax[2].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "06_motor_channel.png"); plt.close(fig)
    return (float(np.sqrt(((peak - thrust) ** 2).mean())),
            float(np.sqrt(((back - thrust) ** 2).mean())),
            float(pc.radius[0]))


def selectivity():
    """How far apart must two values be before their codes stop overlapping?"""
    out = {}
    for nb in [2, 3, 4, 6, 8, 12, 16, 24]:
        p = PlaceCode(1, nb=nb, lo=0.0, hi=1.0, halfw=1.5)
        ends, _ = unit_rows(p.encode([[0.0], [1.0]]))
        d = np.linspace(0, 1, 2001)
        A, _ = unit_rows(p.encode(np.full((len(d), 1), 0.0)))
        B, _ = unit_rows(p.encode(d[:, None]))
        cos = (A * B).sum(1)
        below = np.nonzero(cos < 0.05)[0]
        out[str(nb)] = {"cos_of_the_two_extremes": float(ends[0] @ ends[1]),
                        "value_gap_until_codes_are_disjoint":
                            float(d[below[0]]) if len(below) else None}
    return out


if __name__ == "__main__":
    X = digits()
    rip = fig_one_channel()
    fig_baseline_vs_unknown()
    fig_image_bands(X)
    rec = fig_reconstruction(X)
    sc = fig_positive_template()
    mpk, mct, mrad = fig_motor_channel()

    pc = PlaceCode(784, nb=8, lo=0.0, hi=1.0, halfw=1.5)
    C = pc.encode(X[:32].reshape(32, 784))
    summary = {
        "norm_ripple_max_over_min_by_halfw": rip,
        "image_recon_by_nb__(peak, centroid, fraction_lit)": rec,
        "positive_template_scores": sc,
        "motor_rmse_peak": mpk, "motor_rmse_centroid": mct,
        "motor_cell_spacing": 8.0 / 23, "motor_bump_radius": mrad,
        "selectivity_by_nb": selectivity(),
        "any_negative_value_anywhere": bool((C < 0).any()),
        "fraction_lit_at_nb8": float((C > 1e-6).mean()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
