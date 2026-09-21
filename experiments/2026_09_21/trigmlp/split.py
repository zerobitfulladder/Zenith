"""Split MNIST, one model, trained in sequence: 0/1, then 2/3, then 4/5, then 6/7, then 8/9.

Same network as the best arm in trigmlp.py: 784 -> 256 -> 256 -> 10 with cos+sin as the
activation, pre-activation capped at one turn.  Single 10-way head throughout.  After each
task, accuracy on every task seen so far:

    class-incremental   argmax over all ten logits            (must also know WHICH task)
    task-incremental    argmax over the task's own two logits (told which task, picks within it)

The gap between the two rows is the usual story of forgetting in a shared head.

    uv run python experiments/2026_09_21/trigmlp/split.py
"""
import argparse
import numpy as np, torch, torch.nn as nn
import trigmlp as M

ap = argparse.ArgumentParser()
ap.add_argument("--act", default="trig")
ap.add_argument("--hidden", type=int, default=512)        # trig: 512 -> 256 units, 512 features
ap.add_argument("--bound", type=float, default=2 * np.pi)
ap.add_argument("--epochs", type=int, default=5, help="per task")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="mnist")
a = ap.parse_args()

torch.manual_seed(a.seed); dev = M.dev
Xtr, ytr = M.mnist("train", a.data); Xte, yte = M.mnist("test", a.data)
Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
tasks = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
net = M.mlp(a.act, a.hidden, a.bound).to(dev)
opt = torch.optim.Adam(net.parameters(), a.lr)
print(f"{a.act} H={a.hidden} bound={a.bound:.2f}: {sum(p.numel() for p in net.parameters())} params, "
      f"{a.epochs} epochs per task, single 10-way head, {dev}\n")


def acc(task, mode):
    m = (yte == task[0]) | (yte == task[1])
    with torch.no_grad():
        lg = net(Xte[m])
        if mode == "task":
            lg = lg[:, list(task)]; pred = torch.tensor(task, device=dev)[lg.argmax(1)]
        else:
            pred = lg.argmax(1)
    return (pred == yte[m]).float().mean().item()


CI, TI = np.full((5, 5), np.nan), np.full((5, 5), np.nan)
for t, task in enumerate(tasks):
    m = (ytr == task[0]) | (ytr == task[1]); X, y = Xtr[m], ytr[m]
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), a.bs):
            idx = perm[i:i+a.bs]; opt.zero_grad()
            nn.functional.cross_entropy(net(X[idx]), y[idx]).backward(); opt.step()
    net.eval()
    for s in range(t + 1):
        CI[t, s], TI[t, s] = acc(tasks[s], "class"), acc(tasks[s], "task")
    print(f"after task {t+1} ({task[0]}/{task[1]}):   class-incremental  "
          + "  ".join(f"{tasks[s][0]}/{tasks[s][1]} {CI[t,s]:.3f}" for s in range(t + 1))
          + f"   | mean {np.nanmean(CI[t]):.3f}")
    print(f"{'':>24}   task-incremental   "
          + "  ".join(f"{tasks[s][0]}/{tasks[s][1]} {TI[t,s]:.3f}" for s in range(t + 1))
          + f"   | mean {np.nanmean(TI[t]):.3f}")

# the whole test set, ten-way, after everything
with torch.no_grad():
    full = (net(Xte).argmax(1) == yte).float().mean().item()
print(f"\nfinal, all ten classes, ten-way:  {full:.3f}    "
      f"(the same network trained on all of MNIST at once: ~0.980)")
np.save(M.OUT / f"split_{a.data}_{a.act}_ci.npy", CI); np.save(M.OUT / f"split_{a.data}_{a.act}_ti.npy", TI)
