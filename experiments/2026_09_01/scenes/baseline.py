"""Can one feedforward pass answer a question whose answer-region the image picks?

Three questions of rising difficulty, three readouts each, three test splits.
The number that matters is not any accuracy -- it is the GAP between `seen` and
`order`: the same three digits, the same ink, an arrangement never trained.

Predictions, stated before running (see README):

    Q1 presence        all three fine on every split; it is a set property
    Q2 global order    logistic and mlp fine everywhere -- a fixed region
                       answers it, which is retinotopy doing binding for free.
                       The quantiser should fall on `order`.
    Q3 anchored        the quantiser fails; logistic fails; the mlp does well on
                       `seen` and degrades on `order`, because it has to have
                       seen the arrangement.

If Q3 shows no gap, the premise is wrong and the attention loop is not needed.
"""

import json, time
from pathlib import Path
import numpy as np
import scenes as S
import readouts as R

OUT = Path(__file__).resolve().parent / "results"
N_TRAIN, N_TEST = 20000, 3000
K2, EPOCHS_Q, HIDDEN, EPOCHS_N = 400, 12, 256, 25


def onehot(y, n):
    A = np.zeros((len(y), n), np.float32)
    A[np.arange(len(y)), y] = 1.0
    return A


def top3set(scores, truth):
    """Exact set match: the three highest-scoring classes, as a set."""
    p = np.sort(np.argsort(-scores, axis=1)[:, :3], axis=1)
    t = np.sort(np.argsort(-truth, axis=1)[:, :3], axis=1)
    return float((p == t).all(1).mean())


def run_task(name, base_tr, rows_tr, extra_tr, Y_tr, tests, out, mode, na, chance):
    """tests: {split: (base, rows, extra, Y)}. Returns accuracy per model per split."""
    res = {}
    A_tr = Y_tr if mode == "bce" else onehot(Y_tr, out)

    t0 = time.time()
    W, dead = R.fit_quant(base_tr, rows_tr, extra_tr, A_tr, K=K2, epochs=EPOCHS_Q)
    res["quantiser"] = {"dead": dead}
    for sp, (b, r, e, Y) in tests.items():
        P = R.predict_quant(W, b, r, e, na)
        res["quantiser"][sp] = top3set(P, Y) if mode == "bce" else float((P.argmax(1) == Y).mean())
    print(f"  {name:<12} quantiser  " +
          "  ".join(f"{k} {v:.4f}" for k, v in res['quantiser'].items() if k != 'dead') +
          f"   ({time.time()-t0:.0f}s)", flush=True)

    for label, hidden in (("logistic", 0), ("mlp", HIDDEN)):
        t0 = time.time()
        net = R.fit_net(base_tr, rows_tr, extra_tr, Y_tr, out, mode,
                        hidden=hidden, epochs=EPOCHS_N)
        res[label] = {}
        for sp, (b, r, e, Y) in tests.items():
            P = R.predict_net(net, b, r, e)
            res[label][sp] = top3set(P, Y) if mode == "bce" else float((P.argmax(1) == Y).mean())
        print(f"  {name:<12} {label:<10} " +
              "  ".join(f"{k} {v:.4f}" for k, v in res[label].items()) +
              f"   ({time.time()-t0:.0f}s)", flush=True)
    res["chance"] = chance
    return res


def main():
    t0 = time.time()
    sp = S.splits()
    X, pool_tr, pool_te = S.load_digits()
    W1 = S.load_w1()
    print({k: len(v) for k, v in sp.items()}, flush=True)

    sets = {}
    sets["train"] = S.make(N_TRAIN, sp["train"], X, pool_tr, 100)
    sets["seen"] = S.make(N_TEST, sp["train"], X, pool_te, 200)
    sets["order"] = S.make(N_TEST, sp["order"], X, pool_te, 300)
    sets["triple"] = S.make(N_TEST, sp["triple"], X, pool_te, 400)

    M, C = {}, {}
    for k, (imgs, cls, xs) in sets.items():
        te = time.time()
        M[k] = S.l1_pooled(W1, imgs)
        C[k] = cls
        print(f"  encoded {k:<7} {M[k].shape}  ({time.time()-te:.0f}s)", flush=True)
    TESTS = ["seen", "order", "triple"]
    res = {"splits": {k: len(v) for k, v in sp.items()},
           "n_train": N_TRAIN, "n_test": N_TEST, "dims": int(M["train"].shape[1])}

    # ---- Q1 presence -------------------------------------------------------
    all_rows = {k: np.arange(len(M[k])) for k in M}
    res["Q1_presence"] = run_task(
        "Q1 presence", M["train"], all_rows["train"], None, S.q_presence(C["train"]),
        {k: (M[k], all_rows[k], None, S.q_presence(C[k])) for k in TESTS},
        out=10, mode="bce", na=10, chance=1 / 120)

    # ---- Q2 global order ---------------------------------------------------
    for h, side in enumerate(["leftmost", "rightmost"]):
        col = 0 if side == "leftmost" else -1
        res[f"Q2_{side}"] = run_task(
            f"Q2 {side}", M["train"], all_rows["train"], None,
            C["train"][:, col].astype(np.int64),
            {k: (M[k], all_rows[k], None, C[k][:, col].astype(np.int64)) for k in TESTS},
            out=10, mode="softmax", na=10, chance=0.1)

    # ---- Q3 anchored relation ---------------------------------------------
    A = {k: S.q_anchored(C[k], 0) for k in M}
    res["Q3_anchored"] = run_task(
        "Q3 anchored", M["train"], A["train"][0], onehot(A["train"][1], 10), A["train"][2],
        {k: (M[k], A[k][0], onehot(A[k][1], 10), A[k][2]) for k in TESTS},
        out=11, mode="softmax", na=11, chance=1 / 3)
    res["Q3_anchored"]["chance_note"] = "1/3 -- from presence alone the answer is one of the two other digits or 'none'"

    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "baseline.json").write_text(json.dumps(res, indent=2))
    print(f"\ndone in {res['seconds']:.0f}s -> results/baseline.json")


if __name__ == "__main__":
    main()
