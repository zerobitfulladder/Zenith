"""Is the leaf count tracking the product of independent choices, or the noise?"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from run import free_world
from tree import CatTree

W = lambda f, **kw: free_world(np.random.default_rng(7), f, 8000, **kw)
print(f"{'blocks':>7}{'legal':>7}{'clean':>8}{'drop .05':>10}{'drop+noise':>12}"
      f"{'   <- leaves the tree builds'}")
for f in [1, 2, 3, 4]:
    a = CatTree(min_size=20).fit(W(f, noise=0, drop=0.0))
    b = CatTree(min_size=20).fit(W(f, noise=0, drop=0.05))
    c = CatTree(min_size=20).fit(W(f, noise=2, drop=0.05))
    print(f"{f:>7}{4**f:>7}{len(a.leaves()):>8}{len(b.leaves()):>10}"
          f"{len(c.leaves()):>12}", flush=True)
