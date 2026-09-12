import sys, time, importlib.util, numpy as np
spec = importlib.util.spec_from_file_location('hrun', 'run.py')
hrun = importlib.util.module_from_spec(spec); spec.loader.exec_module(hrun)
sys.path.insert(0, '../sparse_patterns')
from world import *
from engine import Hashed
rng = np.random.default_rng(1000); atoms = make_world(0)
X, _ = sample(rng, atoms, 1500)
print(f"{'K':>5}{'secs':>7}{'slots':>8}{'spec':>7}{'both':>7}{'A+B->C':>9}{'F1->F2':>9}{'amb':>7}", flush=True)
for K in [20, 60, 120]:
    t = time.time(); e = Hashed(N_BITS, K=K).fit(X)
    r = hrun.probes(e, atoms, np.random.default_rng(13), 1.0)
    print(f"{K:>5}{time.time()-t:>7.1f}{len(e.meaning):>8}{r['speculates']:>7.2f}"
          f"{r['both_BC']:>7.2f}{r['cue2_addsC']:>9.2f}{r['F1_addsF2']:>9.2f}"
          f"{r['ambiguity']:>7.2f}", flush=True)
