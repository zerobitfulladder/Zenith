"""Off-the-shelf classifiers on Fashion-MNIST, same 20000/5000 split we use.

Run as:  python baselines.py [fashion_mnist|mnist]

The point is not to win; it is to know what a plain method gets on this data
before the architecture is asked for a number, and to price how much harder
Fashion-MNIST is than MNIST under identical code.
"""

import json, sys, time
from pathlib import Path
import numpy as np
from common import load, CLASSES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
DS = sys.argv[1] if len(sys.argv) > 1 else "fashion_mnist"


def nearest_centroid(Xtr, ytr, Xte):
    C = np.stack([Xtr[ytr == c].mean(0) for c in range(10)])
    return ((Xte @ C.T) - 0.5 * (C * C).sum(1)[None]).argmax(1)


def main():
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.svm import LinearSVC, SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    Xtr, ytr, Xte, yte = load(DS)
    print(f"{DS}: train {Xtr.shape}  test {Xte.shape}", flush=True)
    res = {}

    t = time.time()
    acc = float((nearest_centroid(Xtr, ytr, Xte) == yte).mean())
    res["nearest centroid"] = {"acc": acc, "seconds": round(time.time() - t, 1)}
    print(f"  {'nearest centroid':<22} {acc:.4f}   [{time.time()-t:.0f}s]", flush=True)

    models = [
        ("logistic (SGD)", lambda: SGDClassifier(loss="log_loss", max_iter=30,
                                                 tol=None, random_state=0)),
        ("logistic (lbfgs)", lambda: LogisticRegression(max_iter=300, n_jobs=-1)),
        ("linear SVM", lambda: LinearSVC(C=0.01, dual="auto", max_iter=3000)),
        ("kNN k=3", lambda: KNeighborsClassifier(3, n_jobs=-1)),
        ("random forest 300", lambda: RandomForestClassifier(300, n_jobs=-1,
                                                             random_state=0)),
        ("MLP 256", lambda: MLPClassifier((256,), max_iter=60, random_state=0)),
        ("MLP 512-256", lambda: MLPClassifier((512, 256), max_iter=60, random_state=0)),
        ("SVM rbf", lambda: SVC(C=10.0, gamma="scale")),
    ]
    for name, mk in models:
        t = time.time()
        m = mk().fit(Xtr, ytr)
        acc = float((m.predict(Xte) == yte).mean())
        res[name] = {"acc": acc, "seconds": round(time.time() - t, 1)}
        print(f"  {name:<22} {acc:.4f}   [{time.time()-t:.0f}s]", flush=True)

    best = max(res, key=lambda k: res[k]["acc"])
    print(f"\nbest off-the-shelf: {best} {res[best]['acc']:.4f}")

    # per-class accuracy of the best linear model, to see WHERE the difficulty is
    m = LogisticRegression(max_iter=300, n_jobs=-1).fit(Xtr, ytr)
    p = m.predict(Xte)
    per = {CLASSES[c] if DS.startswith("fashion") else str(c):
           float((p[yte == c] == c).mean()) for c in range(10)}
    print("\nlogistic per class:")
    for k, v in sorted(per.items(), key=lambda kv: kv[1]):
        print(f"  {k:<14} {v:.4f}")
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (yte, p), 1)

    (OUT / f"baselines_{DS}.json").write_text(json.dumps(
        {"dataset": DS, "n_train": len(Xtr), "n_test": len(Xte),
         "results": res, "logistic_per_class": per,
         "logistic_confusion": cm.tolist()}, indent=2))
    print(f"\n-> {OUT}/baselines_{DS}.json")


if __name__ == "__main__":
    main()
