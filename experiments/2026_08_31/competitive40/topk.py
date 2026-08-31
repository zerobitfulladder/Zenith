"""Don't just take the best-fitting expert -- let the best few vote."""
import sys, json, numpy as np
from pathlib import Path
H_=Path.cwd(); sys.path[:0]=[str(H_), str(H_.parent/"competitive")]
from compete40 import H, K, OUT, predict
from compete import load, join, EPS

Xtr,ytr,Xte,yte = load(); n_img=Xtr.shape[1]; z=np.load(OUT/"weights.npz")
res={}
for mode in ("competitive","conscience"):
    W=z[mode].astype(np.float64); wins=z[mode+"_wins"]
    live=np.nonzero(wins.sum(1)>wins.sum()*0.002)[0]
    dom=wins.argmax(1)[live]                       # each expert's claimed digit
    Q=join(Xte); _,R=predict(W,Q,n_img)
    err=(np.linalg.norm(Q[None,:,:n_img]-R,axis=2)/
         np.maximum(np.linalg.norm(Q[None,:,:n_img],axis=2),EPS))[live]   # (L, n)
    L,n=err.shape; order=np.argsort(err,axis=0); ix=np.arange(n)
    r={}
    # --- hard vote among the top-k best fitters, ties broken by rank ---
    for k in (1,2,3,4,5,7,10):
        k=min(k,L); tal=np.zeros((n,10))
        for rank in range(k):
            h=order[rank]
            tal[ix,dom[h]] += 1.0 + 1e-6*(k-rank)   # tiny rank bonus breaks ties
        r[f"hard top-{k}"]=float((tal.argmax(1)==yte).mean())
    # --- soft vote: weight every expert by exp(-err/T) ---
    for T in (0.005,0.01,0.02,0.05,0.1):
        w=np.exp(-(err-err.min(0))/T); tal=np.zeros((n,10))
        np.add.at(tal.T, dom, w)
        r[f"soft all T={T}"]=float((tal.argmax(1)==yte).mean())
    # --- pool a digit's experts: mean of its m best fits ---
    for m in (1,2,3):
        sc=np.full((10,n),np.inf)
        for c in range(10):
            e=err[dom==c]
            if len(e)==0: continue
            e=np.sort(e,axis=0)[:min(m,len(e))]
            sc[c]=e.mean(0)
        r[f"per-digit mean of best {m}"]=float((sc.argmin(0)==yte).mean())
    res[mode]=r
    print(f"\n=== {mode} ({L} live experts, {len(set(dom.tolist()))} digits) ===")
    for kk,v in r.items():
        star=" <--" if v==max(r.values()) else ""
        print(f"  {kk:<26} {v:.4f}{star}")
(OUT/"topk.json").write_text(json.dumps(res,indent=2))
