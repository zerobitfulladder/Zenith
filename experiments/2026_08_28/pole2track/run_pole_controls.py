"""Controls for the pole imitation gap: MLP and 1-NN on the same codes.

Regenerates the exact demo set (same seed), splits 80/20, and measures
held-out expert-action agreement for:
  mlp    two-layer network (64 hidden), the user's question made number
  1nn    nearest raw training sample (no learned rows at all)
  (the K=512 bank sits at ~0.5-0.6 from run_pole2track's tf metric)

MLP high + 1nn high + bank low  -> bank capacity/averaging is the gap
all low                          -> the encoding is information-starved

Run:  .venv/bin/python experiments/2026_08_28/pole2track/run_pole_controls.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_angle"))
sys.path.insert(0, str(ROOT / "experiments" / "2026_08_26" / "pole_swingup"))

from sklearn.neural_network import MLPClassifier          # noqa: E402

from run_pole_swingup import any_init, make_expert, run_episode  # noqa: E402
from run_pole2track import AngleHist, N_DEMOS                    # noqa: E402


def main():
    rng = np.random.default_rng(0)
    best_sign, best_ok = 1, -1
    for sign_k in (1, -1):
        ex = make_expert(sign_k)
        ok = sum(int(run_episode(lambda s, t: ex(s), any_init(
            np.random.default_rng(50)))[1]) for _ in range(20))
        if ok > best_ok:
            best_sign, best_ok = sign_k, ok
    expert = make_expert(best_sign)

    X, y = [], []
    n = 0
    while n < N_DEMOS:
        _, ok, trace = run_episode(lambda s, t: expert(s), any_init(rng))
        if not ok:
            continue
        n += 1
        hist = AngleHist()
        for s, lvl in trace:
            hist.see(s[2])
            X.append(hist.code())
            y.append(lvl)
            hist.did(lvl)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"samples {len(X)} dim {X.shape[1]}", flush=True)

    srng = np.random.default_rng(1)
    perm = srng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = perm[:cut], perm[cut:]

    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                        random_state=0)
    mlp.fit(X[tr], y[tr])
    print(f"MLP held-out agreement: {mlp.score(X[te], y[te]):.3f}",
          flush=True)
    import joblib
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    joblib.dump(mlp, out / "mlp.joblib")
    (out / "expert_sign.txt").write_text(str(best_sign))
    print("saved mlp.joblib + expert_sign.txt", flush=True)

    hits = 0
    sub = te[:2000]
    Xtr = X[tr]
    for i in sub:
        j = int(np.argmax(Xtr @ X[i]))
        hits += int(y[tr][j] == y[i])
    print(f"1-NN held-out agreement: {hits / len(sub):.3f}", flush=True)


if __name__ == "__main__":
    main()
