"""The fourth cell: the PATCH rig with the class pressure switched off.

whole.py --beta 0 collapsed -- churn 0.582 and 19.6 points of representation
damage -- where whole.py at beta 1.5 did not. So on whole digits the pressure,
not the receptive field, is what keeps drift harmless. That leaves the original
patch result ambiguous:

    receptive field   beta   churn   representation damage
    5x5 patch          1.5   0.542            +0.0002      (2026-09-01 forget.py)
    28x28 whole        1.5   0.211            -0.0148      (whole.py)
    28x28 whole        0.0   0.582            +0.1963      (whole.py --beta 0)
    5x5 patch          0.0     ?                 ?         <- this file

If the patch rig at beta 0 keeps high churn AND stays undamaged, then small
generic features are intrinsically safe to drift and the pressure is a second,
independent protection. If it collapses too, then beta was doing the work in
both rigs, the receptive field explains nothing, and "class-agnostic features
make drift lateral" is the wrong reading of 2026-09-01.

Runs 2026_09_01/stack/forget.py unmodified with G.BETA = 0, writing
into THIS folder so the original results are not touched.

Usage:  uv run python patch_beta0.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK = HERE.parents[1] / "2026_09_01" / "stack"
sys.path.insert(0, str(STACK))
sys.argv = [sys.argv[0]]                    # gpu_stack reads argv[1] as dataset

import forget as F

F.G.BETA = 0.0
F.OUT = HERE / "results"
F.OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    F.main()
    src = F.OUT / "forget_mnist.json"
    for ext in ("json", "log"):
        s = F.OUT / f"forget_mnist.{ext}"
        if s.exists():
            s.rename(F.OUT / f"patch_rig_beta0.{ext}")
    print(f"renamed to {F.OUT / 'patch_rig_beta0.json'}")
