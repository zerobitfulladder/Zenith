"""Accuracy, imagination and purity against LAM -- is there a tradeoff or not."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results"
d = json.loads((OUT / "both_metrics.json").read_text())
r, lams = d["results"], d["lams"]
x = np.arange(len(lams))
acc = [r[f"{l}"]["acc"] for l in lams]
accA = [r[f"{l}+A"]["acc"] for l in lams]
gen = [r[f"{l}"]["imagination"] for l in lams]
pur = [r[f"{l}"]["purity"] for l in lams]
own = [r[f"{l}"]["own_class_err"] for l in lams]
alive = [r[f"{l}"]["alive"] for l in lams]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
ax[0].plot(x, acc, "o-", color="#1b6ca8", label="as read by the fit gate")
ax[0].plot(x, accA, "s-", color="#2f8f4e", label="+ reluctance")
ax[0].axhline(0.8186, color="#999", ls="--", lw=1)
ax[0].text(0.05, 0.822, "best label-graded arm 0.8186", fontsize=6.5, color="#666")
ax[0].set_ylabel("test accuracy"); ax[0].legend(fontsize=7)
ax[0].set_title("classification", fontsize=9)

ax[1].plot(x, gen, "o-", color="#c1462d")
ax[1].set_ylabel("imagination: cos(drawn from label, class mean)")
ax[1].set_title("generation", fontsize=9)

ax[2].plot(x, pur, "o-", color="#6a3d9a", label="purity of what each expert won")
ax[2].plot(x, np.array(alive) / 40, "^-", color="#888", label="fraction of experts alive")
ax[2].plot(x, own, "v-", color="#2f8f4e", label="own-class rebuild error")
ax[2].set_ylim(0, 1.02); ax[2].legend(fontsize=7)
ax[2].set_title("structure", fontsize=9)
for a in ax:
    a.set_xticks(x); a.set_xticklabels([str(l) for l in lams])
    a.set_xlabel("λ — how much the naming counts against the drawing")
    a.grid(alpha=.25)
fig.suptitle("score = image error + λ · (1 − confidence in the true label)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "60_lambda.png", dpi=140); plt.close(fig)
print("-> 60_lambda.png")
