# Plain k-means rungs, identity only

2026-08-31, evening. The redesign at the end of the day README, built with the
simplest possible rule: online spherical k-means. Patches are mean-centred and
L2-normalised, so the inner product with a template is their Pearson
correlation; one hypercolumn per patch position, its minicolumns compete, only
the winner learns, and the message up is the winner's index (with the patch's
contrast as its magnitude).

No writeup was kept for this folder. What each script tests, and the numbers its
result files hold:

| script | what it tests | result |
|---|---|---|
| `km.py` | the 5x5 patch codebook on MNIST | `results/km_mnist.json`, `km_mnist.npz` (loaded by `../../2026_09_01`) |
| `show.py` | the templates, and an image rebuilt from template identities | `results/templates_mnist_K*.png`, `rebuild_mnist.png` |
| `tiled.py` | reconstruction from non-overlapping 5x5 patches | `results/tiled.json`: relative error 0.238 / 0.162 / 0.120 at K = 16 / 64 / 256 |
| `joint.py` | one layer, whole images joined with the label | `results/joint_mnist.json`: 0.9086 at K=100 |
| `stack.py` | two rungs, L1 identity map unpooled | `results/stack.json`: 0.7504 |
| `stack_pool.py` | two rungs, L1 map max-pooled first | `results/stack_pool.json`: 0.9492 at pool 4 (103,400 parameters), 0.9364 / 0.926 / 0.518 at 3 / 2 / 1; one layer on pixels 0.921, logistic 0.9074 |
| `gen_pool.py` | generation from the pooled stack through a separate, finer downward memory | `results/gen_pool.npz`, `stack_pool_generate*.png` |
| `three.py`, `three_sweep.py` | three rungs; is L2 the bottleneck | `results/three.json`: 0.9302; `three_sweep.json`: 0.9224 / 0.9274 / 0.9052 at K2 = 128 / 256 / 512 |
| `celeba.py`, `celeba_gen.py` | the two-rung stack on 48x48 CelebA with the 40 attributes as a second stream; faces drawn back out | `results/celeba.json`: mean attribute accuracy 0.8044 against a 0.8078 baseline; `celeba_generate.png`, `celeba_ask.png` |
| `compose.py`, `diff.py`, `ortho.py`, `topk.py` | composition (a woman with a mustache) by arithmetic on the graded L1 map, by differentials, by stripping the male direction, and by activating a population at the top | `results/compose.json`, `diff.json`, `ortho.json`, `topk.json` and the matching `celeba_*.png` |

`results/celeba_sample.png` and `results/celeba_closeup.png` are plain samples
of the 48x48 CelebA data (no script for them was kept; they were at the day's
top level).

Run any of them as `.venv/bin/python experiments/2026_08_31/kmeans/<script>.py`.
`km.py` must run first; the stacks load its codebook.
