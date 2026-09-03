"""XOR with templates + tally. Two conventions: the rig's (mean-centre each
input, drop flat ones) and the same with a constant channel appended (a bias
input). Winner-only learning from random init, then a counted table."""
import numpy as np
X = np.array([[0,0],[0,1],[1,0],[1,1]], float); Y = np.array([0,1,1,0])
def run(K, bias, seed, epochs=200, recruit=False):
    rng = np.random.default_rng(seed)
    Z = np.hstack([X, np.ones((4,1))]) if bias else X
    Z = Z - Z.mean(1, keepdims=True) if not bias else Z          # rig centres; with bias we do not
    nrm = np.linalg.norm(Z, axis=1); keep = nrm > 0.05
    Zn = Z / np.maximum(nrm, 1e-9)[:, None]
    W = rng.standard_normal((K, Z.shape[1])); W /= np.linalg.norm(W, axis=1, keepdims=True)
    n = np.zeros(K)
    for _ in range(epochs):
        for i in rng.permutation(4):
            if not keep[i]: continue
            sc = Zn[i] @ W.T; w = (sc**2).argmax()
            if recruit and sc[w]**2 < 0.9 and (n == 0).any():
                w = int(np.argmax(n == 0)); W[w] = Zn[i]
            n[w] += 1
            W[w] += max(1/n[w], 0.02) * (np.sign(sc[w]) * Zn[i] - W[w]); W[w] /= np.linalg.norm(W[w])
    N = np.zeros((K, 2))
    for i in range(4):
        if keep[i]: N[((Zn[i] @ W.T)**2).argmax(), Y[i]] += 1
    T = np.log((N+1)/(N.sum(0)+K)) - np.log((N.sum(1,keepdims=True)+2)/(N.sum()+2*K))
    pred = [T[((Zn[i] @ W.T)**2).argmax()].argmax() if keep[i] else -1 for i in range(4)]
    return np.mean(np.array(pred) == Y), int(keep.sum())
for recruit in (False, True):
    for bias in (False, True):
        for K in (2, 3, 4, 8):
            accs = [run(K, bias, s, recruit=recruit) for s in range(20)]
            print(f"recruit={recruit!s:5} bias={bias!s:5}  K={K}  inputs kept={accs[0][1]}  "
                  f"XOR over 20 seeds: mean {np.mean([a for a,_ in accs]):.2f}, "
                  f"solved {sum(a==1.0 for a,_ in accs)}/20")
