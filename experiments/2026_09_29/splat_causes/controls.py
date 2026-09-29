"""Controls for run.py, on the table it trained (results/table.pt).

    parts      the same table read with only some of the cost: the pixel term
               alone, the tree's surprise alone
    pixels     the same learning rule on raw pixels: one pass over the 50k,
               store an image when the nearest stored image names it wrongly
               (Hart's condensed nearest neighbour, 1968), same steps of 32;
               and the nearest of as many random training images as the table
               has causes
    hidden     both pixel stores on the same hidden-part tests as run.py
    pilot      two smaller SIG_PIX, since the pilot chose the edge of its grid

Writes results/controls.json.
"""
import json, time
import torch
import run as R

t0 = time.time()
out = {}
R.SIG_PIX = 0.2
R.SIG0V, _ = R.starting_spreads()
tab = torch.load(R.OUT / "table.pt")
C = len(tab["tree"])
R.T = R.Table()
for k, v in tab.items():
    getattr(R.T, k)[:C] = v.to(R.DEV)
R.T.C = C
for s in range(0, C, 256):
    R.T.rerender(torch.arange(s, min(s + 256, C), device=R.DEV))

# ---- which part of the cost does the work
ones = torch.ones(1, 784, device=R.DEV)
right = {"full cost": 0, "pixel term only": 0, "tree surprise only": 0}
for s in range(0, len(R.Xte), 128):
    W, y = R.Xte[s:s + 128], R.yte[s:s + 128]; B = len(W)
    cand = R.shortlist(W, ones, R.S_SHORT, R.root_pose(W))
    k = cand.shape[1]; cp = cand.reshape(-1); Wp = W.repeat_interleave(k, 0)
    ex = R.explain(Wp, R.T.tree[cp], R.T.mean[cp], torch.zeros(len(cp), R.NN, dtype=torch.bool, device=R.DEV))
    pr = R.T.prior()[cp]
    po = R.pose_cost(ex["rel"], R.T.tree[cp], cp).sum(1)
    px = ((Wp - ex["R"]) ** 2).sum(1) / (2 * R.SIG_PIX ** 2)
    for name, cost in (("full cost", pr + po + px), ("pixel term only", pr + px), ("tree surprise only", pr + po)):
        best = cand[torch.arange(B, device=R.DEV), cost.view(B, k).argmin(1)]
        right[name] += int((R.T.counts[best].argmax(1) == y).sum())
out["parts"] = {k: v / len(R.Xte) for k, v in right.items()}
print("parts:", out["parts"])


# ---- the same rule on pixels
def nearest(Q, S, vis=None):
    v = torch.ones_like(Q) if vis is None else vis
    return ((v * Q * Q).sum(1, keepdim=True) - 2 * (v * Q) @ S.T + v @ (S * S).T).argmin(1)


store = [0]
for s in range(1, len(R.Xtr), R.CH):
    W, y = R.Xtr[s:s + R.CH], R.ytr[s:s + R.CH]
    S = torch.tensor(store, device=R.DEV)
    wrong = R.ytr[S][nearest(W, R.Xtr[S])] != y
    store += (torch.arange(s, s + len(W), device=R.DEV)[wrong]).tolist()
cond = torch.tensor(store, device=R.DEV)
g = torch.Generator(device="cpu").manual_seed(3)
rand = torch.randperm(len(R.Xtr), generator=g)[:C].to(R.DEV)
stores = {"condensed pixels (Hart)": cond, f"{C} random training images": rand}
out["pixels"] = {}
for name, idx in stores.items():
    a = float((R.ytr[idx][nearest(R.Xte, R.Xtr[idx])] == R.yte).float().mean())
    out["pixels"][name] = dict(stored=len(idx), numbers=len(idx) * 784, test=a)
print("pixels:", out["pixels"])

# ---- hidden parts, the same masks as run.py
g = torch.Generator(device="cpu").manual_seed(2)
fill_idx = torch.randperm(len(R.Xte), generator=g)[:R.N_FILL].to(R.DEV)
Xf, yf = R.Xte[fill_idx], R.yte[fill_idx]
N = R.N_FILL
v = torch.ones(N, 28, 28, device=R.DEV); v[:, 14:] = 0
masks = {"bottom half hidden": v.reshape(N, 784)}
v = torch.ones(N, 28, 28, device=R.DEV)
for i in range(N):
    ink = (Xf[i] > 0.5).nonzero().flatten()
    p = int(ink[torch.randint(len(ink), (1,), generator=g)])
    r0, c0 = min(max(p // 28 - 6, 0), 16), min(max(p % 28 - 6, 0), 16)
    v[i, r0:r0 + 12, c0:c0 + 12] = 0
masks["12x12 box over ink hidden"] = v.reshape(N, 784)
out["hidden"] = {}
for mname, vis in masks.items():
    hid = 1 - vis
    out["hidden"][mname] = {}
    for name, idx in stores.items():
        j = idx[nearest(Xf * vis, R.Xtr[idx], vis)]
        out["hidden"][mname][name] = dict(label=float((R.ytr[j] == yf).float().mean()),
                                          mse=float((((R.Xtr[j] - Xf) * hid) ** 2).sum() / hid.sum()))
print("hidden:", out["hidden"])

# ---- two smaller SIG_PIX on the pilot (this replaces the loaded table)
out["pilot"] = {}
for sp in (0.1, 0.15):
    R.SIG_PIX = sp
    R.train(R.Xtr[:R.PILOT_N], R.ytr[:R.PILOT_N], log_every=R.PILOT_N)
    a = R.acc(R.read(R.Xva)["pred"], R.yva)
    out["pilot"][sp] = dict(acc=a, causes=int(R.T.active[:R.T.C].sum()))
    print(f"pilot SIG_PIX {sp}: {a:.4f}, causes {out['pilot'][sp]['causes']}")

out["seconds"] = time.time() - t0
json.dump(out, open(R.OUT / "controls.json", "w"), indent=1)
print(f"done in {out['seconds']:.0f}s")
