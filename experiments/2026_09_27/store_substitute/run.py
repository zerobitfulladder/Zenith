"""Store and substitute over a counted table, MNIST, nothing fitted by gradient.

The idea under test (2026-09-27 conversation): replace the deep stack by a search
over a fixed palette, with the actions compose, store and substitute, and tables
of counts instead of weights. This is the flat version, no frontier yet.

    palette    400 templates over centred, unit-length 5x5 patches (k-means)
    tokens     the winning template at each patch position, stride 2, 12x12
               positions; flat patches dropped
    table      N[token, coarse cell (6x6), class], counted once; read as
               log P(token in cell | class) - log P(token in cell), summed
               over the image's tokens, largest of ten                 (arm 1)
    store      the most frequent (or most class-telling) neighbouring pairs
               get a name; pairs are rewritten as the name, the table is
               counted again with names as tokens; three rounds        (arm 2)
    substitute a menu of moves on the image (deskew, thin, thicken, shift,
               scale), each goal digit allowed the moves that pay for it,
               a fixed cost per move, the answer the goal with the best
               score after its own moves                               (arm 3)

Controls: every move applied to every image always (the static merge), the
oracle move, and a CNN with about the same number of parameters (cnn.py).

Split: the 60k training file, first 50k to count and learn, last 10k to choose
the cost and the trigger; the 10k test file for every reported number.
Writes results/: summary.md, metrics.json, store_rounds.png, goal_by_move.png.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
K, PS, FLOOR = 400, 5, 0.05          # palette size, patch side, flatness gate
STRIDE, GRID, NL, ALPHA = 2, 6, 10, 1.0
N_LEARN = 50000                      # the rest of the 60k chooses cost and trigger
N_NAMES, MIN_PAIR, ROUNDS = 300, 20, 3
COSTS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]          # in evidence per token
TRIGGERS = [np.inf, 0.4, 0.2, 0.1, 0.05]          # margin per token below which a goal may move
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
LOG = open(OUT / "run.log", "w")
T0 = time.time()


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()


# ----------------------------------------------------------------- data
def read_idx(path):
    b = Path(path).read_bytes()
    nd = b[3]
    dims = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(b, np.uint8, offset=4 + 4 * nd).reshape(dims)


RAW = ROOT / "data/mnist/digits/raw"
Xall = read_idx(RAW / "train-images-idx3-ubyte").astype(np.float32) / 255.0
yall = read_idx(RAW / "train-labels-idx1-ubyte").astype(np.int64)
Xte = read_idx(RAW / "t10k-images-idx3-ubyte").astype(np.float32) / 255.0
yte = read_idx(RAW / "t10k-labels-idx1-ubyte").astype(np.int64)
Xtr, ytr = Xall[:N_LEARN], yall[:N_LEARN]
Xva, yva = Xall[N_LEARN:], yall[N_LEARN:]
say(f"data: learn {len(Xtr)}, choose {len(Xva)}, test {len(Xte)}; device {DEV}")


# ----------------------------------------------------------------- palette
def unfold(X, stride):
    """(n,28,28) numpy -> centred unit patches (n, L, 25) and gate (n, L) on DEV."""
    x = torch.as_tensor(X, device=DEV)[:, None]
    P = F.unfold(x, PS, stride=stride).transpose(1, 2)          # (n, L, 25)
    P = P - P.mean(-1, keepdim=True)
    nrm = P.norm(dim=-1)
    return P / nrm.clamp_min(1e-12)[..., None], nrm > FLOOR


def make_palette():
    idx = rng.choice(len(Xtr), 6000, replace=False)
    P, g = unfold(Xtr[idx], 1)
    P = P[g]
    P = P[torch.randperm(len(P), device=DEV)[:500000]]
    W = P[torch.randperm(len(P), device=DEV)[:K]].clone()
    for it in range(25):                                        # spherical k-means
        a = (P @ W.T).argmax(1)
        Wn = torch.zeros_like(W).index_add_(0, a, P)
        cnt = torch.bincount(a, minlength=K).float()
        dead = cnt == 0
        Wn[dead] = P[torch.randperm(len(P), device=DEV)[:int(dead.sum())]]
        W = Wn / Wn.norm(dim=1, keepdim=True).clamp_min(1e-12)
    say(f"palette: {K} templates from {len(P)} patches, {int(dead.sum())} reseeded at the end")
    return W


def tokens(X, W, stride=STRIDE, batch=None):
    """Winning template per position, -1 where the patch is flat. (n, L) int32."""
    batch = batch or 4000 * stride * stride // 4
    out = []
    for s in range(0, len(X), batch):
        P, g = unfold(X[s:s + batch], stride)
        a = (P @ W.T).argmax(-1)
        a[~g] = -1
        out.append(a.to(torch.int32).cpu().numpy())
    return np.concatenate(out)


SIDE = (28 - PS) // STRIDE + 1                                   # 12 positions a side
POS_R, POS_C = np.divmod(np.arange(SIDE * SIDE), SIDE)
CELL = (POS_R * GRID // SIDE) * GRID + (POS_C * GRID // SIDE)   # coarse cell per position
NC = GRID * GRID


# ----------------------------------------------------------------- table
def count(tok, y, V):
    img, pos = np.nonzero(tok >= 0)
    N = np.zeros((V, NC, NL))
    np.add.at(N, (tok[img, pos], CELL[pos], y[img]), 1.0)
    return N


def table_from(N):
    pc = (N + ALPHA) / (N.sum(0, keepdims=True) + ALPHA * len(N))
    pm = ((N.sum(2, keepdims=True) + ALPHA * NL)
          / (N.sum((0, 2), keepdims=True) + ALPHA * len(N) * NL))
    return (np.log(pc) - np.log(pm)).astype(np.float32)


def scores(tok, T, per_token=False):
    """Sum of the table's ten-way evidence over the image's tokens. (n, 10).
    per_token divides by the number of tokens, so images with different token
    counts (after a move) can be compared."""
    img, pos = np.nonzero(tok >= 0)
    ev = torch.as_tensor(T, device=DEV)[torch.as_tensor(tok[img, pos], device=DEV),
                                        torch.as_tensor(CELL[pos], device=DEV)]
    S = torch.zeros((len(tok), NL), device=DEV).index_add_(0, torch.as_tensor(img, device=DEV), ev)
    S = S.cpu().numpy()
    if per_token:
        S = S / np.maximum((tok >= 0).sum(1), 1)[:, None]
    return S


def acc(S, y):
    return float((S.argmax(1) == y).mean())


# ----------------------------------------------------------------- store
def pair_matches(tok, V):
    """Every neighbouring pair: (img, pos, dir, key) with key = (a*4+d)*V + b."""
    rows = []
    for d, (dr, dc) in enumerate(DIRS):
        r2, c2 = POS_R + dr, POS_C + dc
        ok = (r2 >= 0) & (r2 < SIDE) & (c2 >= 0) & (c2 < SIDE)
        p1 = np.nonzero(ok)[0]; p2 = r2[p1] * SIDE + c2[p1]
        a, b = tok[:, p1], tok[:, p2]
        both = (a >= 0) & (b >= 0)
        img, j = np.nonzero(both)
        key = (a[img, j].astype(np.int64) * 4 + d) * V + b[img, j]
        rows.append(np.stack([img, p1[j], np.full(len(img), d), key], 1))
    return np.concatenate(rows)


def choose_names(M, y, V, how):
    """The pairs worth a name: by count, or by count times how class-telling."""
    key = M[:, 3]; lab = y[M[:, 0]]
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    hist = np.zeros((len(u), NL)); np.add.at(hist, (inv, lab), 1.0)
    keep = cnt >= MIN_PAIR
    if how == "freq":
        score = cnt.astype(np.float64)
    else:                                                        # count x KL(p(c|pair) || p(c))
        p = (hist + 0.5) / (cnt[:, None] + 0.5 * NL)
        q = np.bincount(y, minlength=NL) / len(y)
        score = cnt * (p * np.log(p / q)).sum(1)
    score[~keep] = -1
    order = np.argsort(-score)[:N_NAMES]
    order = order[score[order] > 0]
    return u[order]                                              # rank order


def rewrite(tok, V, names, mode):
    """Replace each named pair by its name at the first position (replace), or
    add the name as an extra token beside the parts (add). Greedy by rank."""
    M = pair_matches(tok, V)
    order = np.argsort(names)
    j = np.searchsorted(names[order], M[:, 3])
    j = np.minimum(j, len(names) - 1)
    hit = names[order][j] == M[:, 3]
    M = M[hit]; rank = order[j[hit]]
    M = M[np.argsort(rank, kind="stable")]; rank = np.sort(rank, kind="stable")
    used = np.zeros(tok.shape, bool)
    new = tok.copy() if mode == "replace" else np.full(tok.shape, -1, np.int32)
    n_acc = 0
    for (img, p1, d, _), r in zip(M, rank):
        dr, dc = DIRS[d]
        p2 = (POS_R[p1] + dr) * SIDE + POS_C[p1] + dc
        if used[img, p1] or used[img, p2]:
            continue
        used[img, p1] = used[img, p2] = True
        new[img, p1] = V + r
        if mode == "replace":
            new[img, p2] = -1
        n_acc += 1
    return new, n_acc


def merge(tok, extra):
    """For add mode: the parts and the names side by side, as one token array."""
    return np.concatenate([tok, extra], 1)


# ----------------------------------------------------------------- moves
def move(X, name):
    """One move on a batch of images (torch, on DEV). X: (n,28,28)."""
    x = X[:, None]
    n = len(x)
    if name == "none":
        return X
    if name in ("thin", "thicken"):
        pad = F.pad(x, (0, 1, 0, 1), value=0.0 if name == "thicken" else 1.0)
        return (F.max_pool2d(pad, 2, 1) if name == "thicken" else -F.max_pool2d(-pad, 2, 1))[:, 0]
    theta = torch.zeros(n, 2, 3, device=X.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    if name == "deskew":                                         # shear so the stroke stands up
        rows = torch.arange(28., device=X.device)
        tot = X.sum((1, 2)).clamp_min(1e-6)
        cr = (X.sum(2) * rows).sum(1) / tot; cc = (X.sum(1) * rows).sum(1) / tot
        dr = rows[None, :, None] - cr[:, None, None]; dc = rows[None, None, :] - cc[:, None, None]
        mu_rr = (X * dr * dr).sum((1, 2)) / tot; mu_rc = (X * dr * dc).sum((1, 2)) / tot
        alpha = mu_rc / mu_rr.clamp_min(1e-6)
        theta[:, 0, 1] = alpha                                   # x_in = x_out + alpha * y_out
        theta[:, 0, 2] = -alpha * (cr - 13.5) / 13.5             # keep the centre of mass put
    elif name.startswith("shift"):
        dy, dx = {"shift_up": (-2, 0), "shift_down": (2, 0), "shift_left": (0, -2), "shift_right": (0, 2)}[name]
        theta[:, 0, 2] = -dx / 13.5; theta[:, 1, 2] = -dy / 13.5
    elif name.startswith("scale"):
        s = 1.1 if name == "scale_up" else 0.9
        theta[:, 0, 0] = theta[:, 1, 1] = 1.0 / s
    else:
        raise ValueError(name)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]


MOVES = ["deskew", "thin", "thicken", "shift_up", "shift_down", "shift_left", "shift_right", "scale_up", "scale_down"]


def moved(X, seq, batch=10000):
    out = []
    for s in range(0, len(X), batch):
        x = torch.as_tensor(X[s:s + batch], device=DEV)
        for m in seq:
            x = move(x, m)
        out.append(x.clamp(0, 1).cpu().numpy())
    return np.concatenate(out)


def margins(S):
    top2 = np.sort(S, 1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def learn_U(S, y, menu_idx):
    """U[g, m] = mean lift of g's margin under move m for true g's minus for non-g's."""
    n, M, _ = S.shape
    U = np.zeros((NL, M))
    for g in range(NL):
        others = np.delete(np.arange(NL), g)
        mg = S[:, :, g] - S[:, :, others].max(2)                 # (n, M) margin of g
        lift = mg - mg[:, :1]
        is_g = y == g
        U[g] = lift[is_g].mean(0) - lift[~is_g].mean(0)
    U[:, 0] = 0.0
    return U


def decide(S, U, cost, trig, menu_cost):
    """Each goal takes its allowed moves, pays for them, keeps its best score."""
    base = S[:, 0, :]
    allowed = U > 0; allowed[:, 0] = True
    cand = S - (cost * menu_cost)[None, :, None]
    cand = np.where(allowed.T[None], cand, -np.inf)
    best = cand.max(1)
    use = margins(base) < trig
    return np.where(use[:, None], best, base)


# ================================================================= arm 1: baseline
W = make_palette()
np.savez(OUT / "palette.npz", W=W.cpu().numpy())
res = {"setup": dict(K=K, stride=STRIDE, grid=GRID, n_learn=N_LEARN, n_names=N_NAMES, rounds=ROUNDS)}

tok_tr, tok_va, tok_te = tokens(Xtr, W), tokens(Xva, W), tokens(Xte, W)
N = count(tok_tr, ytr, K); T = table_from(N)
S_tr, S_va, S_te = scores(tok_tr, T), scores(tok_va, T), scores(tok_te, T)
per_img = float((tok_te >= 0).sum(1).mean())
res["baseline"] = dict(test=acc(S_te, yte), choose=acc(S_va, yva), learn=acc(S_tr, ytr),
                       tokens_per_image=per_img, table_rows=int(K * NC), median_margin=float(np.median(margins(S_te))),
                       median_margin_per_token=float(np.median(margins(scores(tok_te, T, per_token=True)))))
say(f"arm 1 baseline, stride {STRIDE}: test {res['baseline']['test']:.4f}  tokens/image {per_img:.1f}  "
    f"median margin {res['baseline']['median_margin']:.1f}   [{time.time() - T0:.0f}s]")

# stride-1 reference, the 2026_09_01 shape, for the record
_SIDE1 = 28 - PS + 1
_r1, _c1 = np.divmod(np.arange(_SIDE1 * _SIDE1), _SIDE1)
_CELL1 = (_r1 * GRID // _SIDE1) * GRID + (_c1 * GRID // _SIDE1)
def _count1(tok, y):
    img, pos = np.nonzero(tok >= 0); N = np.zeros((K, NC, NL))
    np.add.at(N, (tok[img, pos], _CELL1[pos], y[img]), 1.0); return N
def _scores1(tok, T):
    img, pos = np.nonzero(tok >= 0)
    ev = torch.as_tensor(T, device=DEV)[torch.as_tensor(tok[img, pos], device=DEV), torch.as_tensor(_CELL1[pos], device=DEV)]
    return torch.zeros((len(tok), NL), device=DEV).index_add_(0, torch.as_tensor(img, device=DEV), ev).cpu().numpy()
t1_tr, t1_te = tokens(Xtr, W, 1), tokens(Xte, W, 1)
res["baseline_stride1"] = dict(test=acc(_scores1(t1_te, table_from(_count1(t1_tr, ytr))), yte),
                               tokens_per_image=float((t1_te >= 0).sum(1).mean()))
say(f"       stride 1 reference: test {res['baseline_stride1']['test']:.4f}  tokens/image {res['baseline_stride1']['tokens_per_image']:.1f}")
del t1_tr, t1_te

# ================================================================= arm 2: store
res["store"] = {}
for how, mode, n_names, rounds in (("freq", "replace", 300, ROUNDS), ("gain", "replace", 300, ROUNDS),
                                   ("freq", "add", 300, 1), ("gain", "add", 300, 1),
                                   ("freq", "replace", 1000, 1), ("freq", "add", 1000, 1)):
        N_NAMES = n_names
        rows = [dict(round=0, test=res["baseline"]["test"], tokens_per_image=per_img, names=0, vocab=K, accepted_per_image=0.0)]
        cur_tr, cur_va, cur_te, V = tok_tr, tok_va, tok_te, K
        for r in range(1, rounds + 1):
            names = choose_names(pair_matches(cur_tr, V), ytr, V, how)
            new_tr, n_tr = rewrite(cur_tr, V, names, mode)
            new_va, _ = rewrite(cur_va, V, names, mode)
            new_te, n_te = rewrite(cur_te, V, names, mode)
            V2 = V + len(names)
            if mode == "replace":
                cur_tr, cur_va, cur_te = new_tr, new_va, new_te
            else:                                               # names beside the parts, one round only
                cur_tr, cur_va, cur_te = merge(cur_tr, new_tr), merge(cur_va, new_va), merge(cur_te, new_te)
            V = V2
            # counting and reading over a wider token array: positions repeat, cells too
            if mode == "add":
                CELL_SAVE = CELL; globals()["CELL"] = np.concatenate([CELL_SAVE] * (cur_tr.shape[1] // len(CELL_SAVE)))
            Tn = table_from(count(cur_tr, ytr, V))
            a_te = acc(scores(cur_te, Tn), yte); a_va = acc(scores(cur_va, Tn), yva)
            if mode == "add":
                globals()["CELL"] = CELL_SAVE
            tpi = float((cur_te >= 0).sum(1).mean())
            rows.append(dict(round=r, test=a_te, choose=a_va, tokens_per_image=tpi, names=int(len(names)), vocab=int(V),
                             accepted_per_image=float(n_te / len(cur_te))))
            say(f"arm 2 store/{how}/{mode}/{n_names} round {r}: test {a_te:.4f}  tokens/image {tpi:.1f}  "
                f"names {len(names)}  vocab {V}  pairs replaced/image {n_te / len(cur_te):.1f}   [{time.time() - T0:.0f}s]")
        res["store"][f"{how}_{mode}_{n_names}"] = rows

# ================================================================= arm 3: substitute
menu = [["none"]] + [[m] for m in MOVES] + [[a, b] for i, a in enumerate(MOVES) for b in MOVES[i + 1:]]
menu_name = ["+".join(s) for s in menu]
menu_cost = np.array([len(s) - (s[0] == "none") for s in menu], np.float32)
say(f"arm 3: {len(menu) - 1} moved versions of every image ({len(MOVES)} single moves, {len(menu) - 1 - len(MOVES)} pairs)")
S3 = {}
for split, X in (("tr", Xtr), ("va", Xva), ("te", Xte)):
    S = np.zeros((len(X), len(menu), NL), np.float32)
    for k, seq in enumerate(menu):
        S[:, k] = scores(tokens(moved(X, seq), W), T, per_token=True)
    S3[split] = S
say(f"       scored   [{time.time() - T0:.0f}s]")

res["substitute"] = {}
for menu_kind, keep in (("singles", np.arange(len(MOVES) + 1)), ("singles+pairs", np.arange(len(menu)))):
    U = learn_U(S3["tr"][:, keep], ytr, keep)
    grid = {}
    best = None
    for c in COSTS:
        for t in TRIGGERS:
            a_va = acc(decide(S3["va"][:, keep], U, c, t, menu_cost[keep]), yva)
            a_te = acc(decide(S3["te"][:, keep], U, c, t, menu_cost[keep]), yte)
            grid[f"cost {c:g} trigger {t:g}"] = dict(choose=a_va, test=a_te)
            if best is None or a_va > best[0]:
                best = (a_va, c, t, a_te)
    a_va, c, t, a_te = best
    Ste = decide(S3["te"][:, keep], U, c, t, menu_cost[keep])
    p0, p1 = S3["te"][:, 0].argmax(1), Ste.argmax(1)
    flipped_right = int(((p0 != yte) & (p1 == yte)).sum()); flipped_wrong = int(((p0 == yte) & (p1 != yte)).sum())
    moves_taken = float((Ste.max(1) != S3["te"][:, 0].max(1)).mean())
    # oracle: the move that most helps the true class, an upper bound on the menu
    S_or = S3["te"][:, keep]
    m_true = np.stack([S_or[:, :, g] - np.delete(S_or, g, 2).max(2) for g in range(NL)], 2)   # (n, M, 10)
    best_m = m_true[np.arange(len(yte)), :, yte].argmax(1)
    oracle = acc(S_or[np.arange(len(yte)), best_m], yte)
    free = acc(decide(S3["te"][:, keep], np.ones_like(U), 0.0, np.inf, menu_cost[keep]), yte)   # every move, no cost, no U
    res["substitute"][menu_kind] = dict(chosen=dict(cost=c, trigger=(None if np.isinf(t) else t), choose=a_va, test=a_te),
                                        flipped_right=flipped_right, flipped_wrong=flipped_wrong,
                                        moves_taken_frac=moves_taken, oracle=oracle, free_moves_no_cost=free,
                                        grid=grid, U={MOVES[m - 1] if m else "none": U[:, i].round(3).tolist() for i, m in enumerate(keep) if m <= len(MOVES)})
    say(f"arm 3 substitute, menu {menu_kind}: chosen cost {c:g} trigger {t:g} -> test {a_te:.4f}  "
        f"(baseline {res['baseline']['test']:.4f}; flipped right {flipped_right}, wrong {flipped_wrong}; "
        f"moves taken on {moves_taken:.1%}; oracle {oracle:.4f}; every move free {free:.4f})   [{time.time() - T0:.0f}s]")
    if menu_kind == "singles":
        U_singles = U

# controls: the static merge, every image moved the same way, table counted on moved images too
res["always"] = {}
for m in MOVES:
    tt = tokens(moved(Xtr, [m]), W); te = tokens(moved(Xte, [m]), W)
    res["always"][m] = acc(scores(te, table_from(count(tt, ytr, K))), yte)
    say(f"control always-{m}: test {res['always'][m]:.4f}")

# ================================================================= figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for key, st in (("freq_replace_300", "-"), ("gain_replace_300", "--")):
    rows = res["store"][key]
    ax[0].plot([r["round"] for r in rows], [r["test"] for r in rows], st + "o", label=f"store, {key.split('_')[0]} pairs, replace")
    ax[1].plot([r["round"] for r in rows], [r["tokens_per_image"] for r in rows], st + "o", label=f"{key.split('_')[0]}")
for key, mk in (("freq_add_300", "s"), ("gain_add_300", "^"), ("freq_replace_1000", "D"), ("freq_add_1000", "v")):
    r = res["store"][key][-1]
    ax[0].plot([r["round"]], [r["test"]], mk, label=f"store, {key.replace('_', ' ')} names")
b = res["baseline"]["test"]; se = float(np.sqrt(b * (1 - b) / len(yte)))
ax[0].axhline(b, color="grey", lw=0.8, label="baseline")
ax[0].axhspan(b - 2 * se, b + 2 * se, color="grey", alpha=0.15, label="baseline, two standard errors")
ax[0].set_xlabel("round of storing"); ax[0].set_ylabel("test accuracy"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("round of storing"); ax[1].set_ylabel("tokens per test image"); ax[1].legend(fontsize=8)
fig.suptitle("Arm 2: naming neighbouring pairs and counting the table again")
plt.tight_layout(); plt.savefig(OUT / "store_rounds.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(8, 5.5))
Um = U_singles[:, 1:]
v = np.abs(Um).max()
im = ax.imshow(Um, cmap="RdBu_r", vmin=-v, vmax=v)
ax.set_xticks(range(len(MOVES))); ax.set_xticklabels(MOVES, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(NL)); ax.set_yticklabels([f"goal {g}" for g in range(NL)])
for g in range(NL):
    for j in range(len(MOVES)):
        ax.text(j, g, f"{Um[g, j]:+.1f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="lift of the goal's margin: true digits minus the rest")
ax.set_title("Arm 3: which move pays for which goal (learned from the 50k, single moves)")
plt.tight_layout(); plt.savefig(OUT / "goal_by_move.png", dpi=130); plt.close()

fig, ax = plt.subplots(3, len(MOVES) + 1, figsize=(15, 4.6))
for i, k in enumerate([3, 17, 42]):
    for j, m in enumerate(["none"] + MOVES):
        ax[i, j].imshow(moved(Xte[k:k + 1], [m])[0], cmap="gray_r", vmin=0, vmax=1)
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if i == 0:
            ax[i, j].set_title(m, fontsize=9)
fig.suptitle("The menu of moves, three test digits")
plt.tight_layout(); plt.savefig(OUT / "moves.png", dpi=110); plt.close()

json.dump(res, open(OUT / "metrics.json", "w"), indent=1, default=float)
say(f"done   [{time.time() - T0:.0f}s]")
