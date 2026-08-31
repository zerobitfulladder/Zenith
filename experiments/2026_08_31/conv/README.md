# Part three — the convolutional rung (`conv/`)

A **unit** is H hypercolumns, each holding K minicolumns, all looking at the
same 5x5 patch, slid over the image with stride 1 and shared weights. The rule
is the settled base and nothing else: reconstruction error minus conscience,
winner takes one geodesic step. No label exists for a patch, so the grading knob
reduces to the drawing score.

Minicolumns do **not** compete. They span a subspace together and emit a dense
code; the hypercolumn is the unit of identity. Their templates therefore look
like noise, and that is correct — a subspace has no preferred basis. Measured:
each hypercolumn overlaps the global top-K principal subspace by 0.58-0.73 and
its neighbours by 0.42-0.65, so the layer is local PCA with mild specialisation.

| H | K | rebuild | busiest | overlap | probe |
|---|---|---|---|---|---|
| 8 | 3 | 0.3755 | 16.5% | 0.287 | 0.5900 |
| 8 | 8 | 0.2027 | 15.7% | 0.497 | 0.7575 |
| 8 | 32 | 0.1010 | 19.6% | 0.805 | 0.8180 |
| 4 | 8 | 0.2407 | 34.5% | 0.552 | 0.6135 |
| 32 | 8 | 0.1569 | 5.0% | 0.427 | 0.8450 |
| 64 | 8 | 0.1414 | 4.3% | 0.387 | 0.8775 |

The probe rises with everything, but that is not evidence the parts are good —
a bigger subspace preserves more input and a linear readout picks up whatever
survives.

## What the unit should send up

| format | dims (2x2 pooled) | probe |
|---|---|---|
| identity only, H=8 | 32 | 0.7775 |
| identity only, H=32 | 128 | 0.9070 |
| identity only, H=64 | 256 | 0.9370 |
| identity + dense, H=8/K=6 | 192 | 0.9445 |
| identity + dense, H=32/K=8 | 1024 | 0.9605 |
| dense only, H=32/K=8 | 32 | 0.6825 |
| *raw pixels* | *784* | *0.8925* |

Identity+dense on a 2x2 grid is the first message in this project to **beat raw
pixels** — 0.9445 from 192 numbers against 0.8925 from 784.

Identity-only nearly matches it at a comparable budget (0.9370 at 256 dims), so
the dense block is mostly a way of spending dimensions; you can spend them on
more hypercolumns instead and get a six-times sparser alphabet. **But the
message format and H are coupled**: identity-only needs H=32-64, not 8.

Dense-only — the code without any indication of who produced it — is the risky
one. The winner is recoverable from the code alone at 0.5080 against a chance
rate of 0.0312, so the bases do leave a signature, but half the time the
upper layer would be reading a vector whose basis it can only guess.

## Two things to remember about this rung

A hypercolumn never gets a free perfect rebuild: reconstruction is a single
matched-filter pass `WᵀW q`, not a least-squares fit, so with K > 25 templates
in a 25-dimensional patch space it *over-counts* and error rises again (0.101 at
K=32, 0.272 at K=64).

And 44% of 5x5 patches on MNIST are flat background, dropped before training —
L2-normalising an empty patch amplifies noise into a fake edge.

## The stack on Fashion, and what actually helps (`conv/`)

Same split as the baselines, 20000 train / 5000 test, everything teacher-free:

| | Fashion |
|---|---|
| stack, grid 12 pooling, H2=40 | **0.7950** |
| stack, grid 8, H2=60 | 0.7826 |
| stack, grid 8, H2=20 | 0.7524 |
| stack, grid 6, H2=20 | 0.7190 |
| the same L2 unit on raw pixels | 0.6466 |
| one layer, teacher-free | 0.6868 |
| one layer, with a teacher | 0.8278 |
| logistic on pixels | 0.8512 |

Two things fall out, and they decided the redesign below.

**Spatial resolution is worth ten points; capacity is worth nothing.** Holding
L2 at 40 hypercolumns and pooling 6 -> 8 -> 12 gives 0.6932 -> 0.7470 -> 0.7950.
Holding the grid at 8 and going 20 -> 40 -> 60 hypercolumns gives
0.7524 -> 0.7470 -> 0.7826 while the parameters triple from 659K to 1.98M.

**About thirteen experts do the work, always.** "Really live" was 12-15 in every
row, whether L2 had 20 hypercolumns or 60. At H2=60 that is 46 hypercolumns
holding 1.5M parameters and doing nothing. L1, by contrast, is 6,400 parameters
total and shared across all 576 positions.

So the top unit holds 99.8% of the parameters, is mostly idle, and is the part
that isn't working. No cortical area looks at everything at once.

## Files

`conv1.py` trains rung one (`run_sweep.sh` runs it on MNIST and Fashion);
`message.py` / `formats.py` compare what the unit sends up; `l2.py`,
`l2_identity.py`, `l2_teacher.py` are layer two on the dense, identity and
teacher-assisted messages; `fashion_stack.py` and `sweep2.py` are the Fashion
stack and its pooling/width sweep; `imagine.py`, `render_l2.py`, `show*.py`
draw templates and generations. Everything writes to `results/`. The Fashion
loader is imported from `../fashion/common.py`.
