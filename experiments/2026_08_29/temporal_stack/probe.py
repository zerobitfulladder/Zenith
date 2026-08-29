"""How well does a frozen stack reproduce the oracle's command?

Episode success is end-to-end and blunt — it is zero for a long time
before it is anything else. This measures the thing underneath it: on the
ORACLE's own trajectory (so distribution shift is excluded), how closely
does the frozen policy reproduce what the oracle commanded?

The number that matters is `error sd` against `oracle sd`. If they are
equal the read carries no usable signal, whatever the correlation says.

  .venv/bin/python experiments/2026_08_29/temporal_stack/probe.py [ckpt.npz]
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "temporal_drone"))      # shared rig modules
sys.path.insert(0, str(HERE.parents[1] / "2026_08_28" / "single_layer_drone"))

import sl_drone as W                       # noqa: E402
import fast_oracle as F                    # noqa: E402
import td3_stack as S                      # noqa: E402


def load(path):
    cfg = json.load(open(Path(path).with_suffix(".json")))
    z = np.load(path)
    st = S.Stack3(k1=cfg["k1"], k2=cfg["k2"], k3=cfg["k3"], seed=0)
    for lay, a, b in ((st.l1, "W1", "n1"), (st.l2, "W2", "n2"),
                      (st.l3, "W3", "n3")):
        lay.hc.W = z[a]
        lay.hc.n_boot = int(z[b])
    st.frozen = True
    return st, cfg


def probe(st, n=8000, seed=11):
    rng = np.random.default_rng(seed)
    s, tgt = W.any_init(rng)
    lv = (W.NLEV // 2, W.NLEV // 2)
    st.reset_history()
    pc, pd, oc, od, ag = [], [], [], [], []
    for _ in range(n):
        st.push(S.norm_state(s, tgt), lv)
        guess, _ = st.act()
        lv = F.teacher(s, tgt)                    # the oracle flies
        gc, gd = S.levels_to_cd(guess)
        tc, td = S.levels_to_cd(lv)
        pc.append(gc); pd.append(gd); oc.append(tc); od.append(td)
        ag.append(0.5 * ((guess[0] == lv[0]) + (guess[1] == lv[1])))
        s = W.physics(s, lv)
        if W.at_goal(s, tgt) or abs(s[0]) > 30 or abs(s[1]) > 30:
            s, tgt = W.any_init(rng)
            lv = (W.NLEV // 2, W.NLEV // 2)
            st.reset_history()
    return (np.array(pc), np.array(pd), np.array(oc), np.array(od),
            float(np.mean(ag)))


def report(path):
    st, cfg = load(path)
    pc, pd, oc, od, ag = probe(st)
    print(f"{Path(path).parent.name}/{Path(path).stem}  tick={cfg.get('tick')}"
          f"  templates={st.l3.hc.n_boot}")
    print(f"  exact-level agreement {ag:.3f}   (chance {1/W.NLEV:.3f})")
    print(f"  {'':<14}{'corr':>7}{'oracle sd':>11}{'error sd':>10}{'SNR':>7}")
    for nm, p, o in (("collective", pc, oc), ("differential", pd, od)):
        e = float((p - o).std())
        print(f"  {nm:<14}{np.corrcoef(p, o)[0, 1]:>7.3f}{o.std():>11.3f}"
              f"{e:>10.3f}{o.std() / max(e, 1e-9):>7.2f}")


if __name__ == "__main__":
    args = sys.argv[1:] or [str(HERE / "results" / "three_layer_clone2" / "checkpoint.npz")]
    for a in args:
        report(a)
