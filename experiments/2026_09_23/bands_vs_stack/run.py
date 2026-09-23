"""Parallel bands against a stack, on the 40 CelebA attributes.

The question: does a layer need to read the layer below it, or is it enough that it reads
the image at a coarser scale?  Every arm ends in the same readout, so only the wiring differs.

    pixels   the crop shrunk to 40x32, one linear map.  The floor.
    bands1   five copies of the image (full size, half, quarter, eighth, sixteenth), each
             read by its own conv3x3 + ReLU.  No level sees another level.
    bands3   the same, but three conv layers per level.  Composition within a scale is
             allowed; composition across scales still is not.
    stack    a CNN: conv3x3 + ReLU, shrink by two, five times.  Level k reads level k-1.

Readout, identical everywhere: each level's map is average-pooled to a 5x4 grid, the grids
are concatenated, one linear map gives 40 logits.  With C channels that is 5*20*C numbers.

Conditions:
    aligned  a fixed 160x128 centre crop of the 218x178 image (CelebA faces are aligned).
    jitter   the same crop size at a random position, both in training and in testing.

Score: balanced accuracy per attribute (mean of accuracy on positives and on negatives),
averaged over the 40 attributes.  Trained with a class-balanced BCE so that a threshold of
zero is the right one.

Usage:
    python run.py --arm bands1 --cond aligned --seed 0
    python run.py --arm bands1 --cond aligned --only_level 3      # one band alone
    python run.py --arm stack --cond jitter --pool max            # the usual CNN pooling
"""
import argparse, json, time, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data/celeba"
OUT = HERE / "results/runs"; OUT.mkdir(parents=True, exist_ok=True)

H, W = 160, 128           # crop size (height, width)
FULL_H, FULL_W = 218, 178
GRID = (5, 4)             # readout grid, 32x32 pixel cells at full resolution
LEVELS = 5

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True, choices=["pixels", "bands1", "bands3", "stack"])
p.add_argument("--cond", default="aligned", choices=["aligned", "jitter"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--C", type=int, default=32)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--pool", default="avg", choices=["avg", "max"], help="stack only: how it shrinks")
p.add_argument("--only_level", type=int, default=0, help="bands only: keep one level (1=finest)")
p.add_argument("--n_train", type=int, default=0, help="0 = all")
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------- data, all on the GPU
imgs = torch.from_numpy(np.load(DATA / "gray_218x178.npy")).to(dev)          # uint8 N,218,178
attrs = torch.from_numpy(np.load(DATA / "attrs_40.npy")).float().to(dev)     # N,40
names = json.loads((DATA / "attr_names_40.json").read_text())
N = len(imgs)
n_test, n_val = 5000, 2000
idx_test = torch.arange(N - n_test, N, device=dev)
idx_val = torch.arange(N - n_test - n_val, N - n_test, device=dev)
idx_train = torch.arange(0, N - n_test - n_val, device=dev)
if args.n_train:
    idx_train = idx_train[: args.n_train]

pos = attrs[idx_train].mean(0)
pos_weight = ((1 - pos) / pos.clamp_min(1e-3)).to(dev)     # balanced BCE
MEAN, STD = 0.45, 0.25

g = torch.Generator(device=dev); g.manual_seed(1000 + args.seed)
def offsets(idx, train):
    """Crop origin per image.  aligned: the centre.  jitter: uniform over all positions."""
    B = len(idx)
    if args.cond == "aligned":
        return (torch.full((B,), (FULL_H - H) // 2, device=dev),
                torch.full((B,), (FULL_W - W) // 2, device=dev))
    if train:
        return (torch.randint(0, FULL_H - H + 1, (B,), device=dev, generator=g),
                torch.randint(0, FULL_W - W + 1, (B,), device=dev, generator=g))
    # fixed per image for evaluation, from the image index
    oy = (idx * 7919) % (FULL_H - H + 1)
    ox = (idx * 104729) % (FULL_W - W + 1)
    return oy, ox

ar_h = torch.arange(H, device=dev); ar_w = torch.arange(W, device=dev)
def batch(idx, train):
    oy, ox = offsets(idx, train)
    rows = (oy[:, None] + ar_h)[:, :, None]
    cols = (ox[:, None] + ar_w)[:, None, :]
    x = imgs[idx[:, None, None], rows, cols].float().div_(255).sub_(MEAN).div_(STD)
    return x.unsqueeze(1), attrs[idx]

# ---------------------------------------------------------------- models
def pyramid(x):
    out = [x]
    for _ in range(LEVELS - 1):
        x = F.avg_pool2d(x, 2); out.append(x)
    return out

class Head(nn.Module):
    """Every level pooled to the same grid, concatenated, one linear map."""
    def __init__(self, n_maps, C):
        super().__init__(); self.fc = nn.Linear(n_maps * C * GRID[0] * GRID[1], 40)
    def forward(self, maps):
        return self.fc(torch.cat([F.adaptive_avg_pool2d(m, GRID).flatten(1) for m in maps], 1))

def branch(C, depth):
    layers, cin = [], 1
    for _ in range(depth):
        layers += [nn.Conv2d(cin, C, 3, padding=1), nn.ReLU(inplace=True)]; cin = C
    return nn.Sequential(*layers)

class Bands(nn.Module):
    def __init__(self, C, depth, only=0):
        super().__init__()
        self.levels = [only - 1] if only else list(range(LEVELS))
        self.branches = nn.ModuleList([branch(C, depth) for _ in self.levels])
        self.head = Head(len(self.levels), C)
    def forward(self, x):
        P = pyramid(x)
        return self.head([b(P[l]) for b, l in zip(self.branches, self.levels)])

class Stack(nn.Module):
    def __init__(self, C, pool):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1 if i == 0 else C, C, 3, padding=1) for i in range(LEVELS)])
        self.pool = F.max_pool2d if pool == "max" else F.avg_pool2d
        self.head = Head(LEVELS, C)
    def forward(self, x):
        maps = []
        for i, conv in enumerate(self.convs):
            if i: x = self.pool(x, 2)
            x = F.relu(conv(x), inplace=True); maps.append(x)
        return self.head(maps)

class Pixels(nn.Module):
    def __init__(self):
        super().__init__(); self.fc = nn.Linear(40 * 32, 40)
    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(x, (40, 32)).flatten(1))

if args.arm == "pixels": model = Pixels()
elif args.arm == "bands1": model = Bands(args.C, 1, args.only_level)
elif args.arm == "bands3": model = Bands(args.C, 3, args.only_level)
else: model = Stack(args.C, args.pool)
model.to(dev)
n_params = sum(p.numel() for p in model.parameters())
n_head = sum(p.numel() for p in model.head.parameters()) if hasattr(model, "head") else n_params

# ---------------------------------------------------------------- train / eval
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
steps = args.epochs * (len(idx_train) // args.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)

@torch.no_grad()
def evaluate(idx):
    model.eval()
    tp = fp = tn = fn = torch.zeros(40, device=dev)
    for i in range(0, len(idx), 250):
        x, y = batch(idx[i:i + 250], train=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        pred = (out.float() > 0).float()
        tp = tp + (pred * y).sum(0); fp = fp + (pred * (1 - y)).sum(0)
        tn = tn + ((1 - pred) * (1 - y)).sum(0); fn = fn + ((1 - pred) * y).sum(0)
    tpr = tp / (tp + fn).clamp_min(1); tnr = tn / (tn + fp).clamp_min(1)
    bal = (tpr + tnr) / 2
    acc = (tp + tn) / (tp + tn + fp + fn)
    model.train()
    return bal.cpu().numpy(), acc.cpu().numpy()

tag = f"{args.arm}{'_L%d' % args.only_level if args.only_level else ''}{'_max' if args.arm == 'stack' and args.pool == 'max' else ''}_{args.cond}_s{args.seed}"
print(f"{tag}: {n_params} parameters ({n_head} in the head), {len(idx_train)} training images, {dev}")
log, best = [], (-1, None)
t0 = time.time()
for ep in range(args.epochs):
    perm = idx_train[torch.randperm(len(idx_train), device=dev, generator=g)]
    tot = 0.0
    for i in range(0, len(perm) - args.batch + 1, args.batch):
        x, y = batch(perm[i:i + args.batch], train=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        loss = F.binary_cross_entropy_with_logits(out.float(), y, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        tot += loss.item()
    val_bal, val_acc = evaluate(idx_val)
    test_bal, test_acc = evaluate(idx_test)
    row = dict(epoch=ep + 1, loss=tot / (len(perm) // args.batch), val=float(val_bal.mean()),
               test=float(test_bal.mean()), test_acc=float(test_acc.mean()), time=time.time() - t0)
    log.append(row)
    if row["val"] > best[0]:
        best = (row["val"], dict(epoch=ep + 1, test_bal=test_bal.tolist(), test_acc=test_acc.tolist(),
                                 test=float(test_bal.mean()), val=float(val_bal.mean())))
    print(f"  ep {ep+1:2d}  loss {row['loss']:.4f}  val {row['val']:.4f}  test {row['test']:.4f}  "
          f"acc {row['test_acc']:.4f}  {row['time']:.0f}s", flush=True)

res = dict(tag=tag, arm=args.arm, cond=args.cond, seed=args.seed, C=args.C, pool=args.pool,
           only_level=args.only_level, epochs=args.epochs, n_train=len(idx_train), params=n_params,
           head_params=n_head, names=names, best=best[1], log=log, seconds=time.time() - t0)
(OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
print(f"best epoch {best[1]['epoch']}: test balanced accuracy {best[1]['test']:.4f}  -> results/runs/{tag}.json")
