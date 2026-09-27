"""The control: a CNN with about as many parameters as the counted table, and a
linear map on pixels. Same split as run.py (50k learn, 10k choose the epoch,
10k test), one seed, 8 epochs. Writes results/cnn.json."""
import json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def read_idx(path):
    b = Path(path).read_bytes(); nd = b[3]
    dims = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(b, np.uint8, offset=4 + 4 * nd).reshape(dims)


RAW = ROOT / "data/mnist/digits/raw"
X = torch.as_tensor(read_idx(RAW / "train-images-idx3-ubyte").astype(np.float32) / 255.0)[:, None]
y = torch.as_tensor(read_idx(RAW / "train-labels-idx1-ubyte").astype(np.int64))
Xte = torch.as_tensor(read_idx(RAW / "t10k-images-idx3-ubyte").astype(np.float32) / 255.0)[:, None].to(DEV)
yte = torch.as_tensor(read_idx(RAW / "t10k-labels-idx1-ubyte").astype(np.int64)).to(DEV)
Xtr, ytr, Xva, yva = X[:50000].to(DEV), y[:50000].to(DEV), X[50000:].to(DEV), y[50000:].to(DEV)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 32, 5), nn.Conv2d(32, 64, 5)
        self.f1, self.f2 = nn.Linear(64 * 4 * 4, 64), nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))


class Linear(nn.Module):
    def __init__(self):
        super().__init__(); self.f = nn.Linear(784, 10)

    def forward(self, x):
        return self.f(x.flatten(1))


def accuracy(net, X, y):
    net.eval()
    with torch.no_grad():
        return float(sum((net(X[s:s + 2000]).argmax(1) == y[s:s + 2000]).sum() for s in range(0, len(X), 2000)) / len(X))


out = {}
for name, net in (("cnn", CNN()), ("linear", Linear())):
    net = net.to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    best = (0, 0, 0)
    t0 = time.time()
    for ep in range(8):
        net.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        for s in range(0, len(Xtr), 128):
            i = perm[s:s + 128]
            loss = F.cross_entropy(net(Xtr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        va, te = accuracy(net, Xva, yva), accuracy(net, Xte, yte)
        if va > best[0]:
            best = (va, te, ep + 1)
        print(f"{name} epoch {ep + 1}: choose {va:.4f} test {te:.4f}")
    out[name] = dict(parameters=n_par, choose=best[0], test=best[1], epoch=best[2], seconds=round(time.time() - t0))
    print(f"{name}: {n_par} parameters, best epoch {best[2]}, test {best[1]:.4f}")
json.dump(out, open(HERE / "results" / "cnn.json", "w"), indent=1)
