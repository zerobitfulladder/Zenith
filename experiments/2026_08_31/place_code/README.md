# The place code, on images and on a motor channel

2026-08-31. The encoder from `sl_drone.py` / `pop_regress.py`, pulled out,
pointed at images, and measured. A value stops being a number and becomes
**which cells are lit** — a bump on that channel's ladder. Nothing in the
file is ever negative, which is the point.

Two questions were being asked of it. Does mean-centring and splitting an
image into its positive and negative halves give the same thing? And what
does `nb`, the cells per channel, actually buy?

## The rule

Each channel owns a contiguous block of `nb` cells; each cell stands for one
value. A value lights the cells around it with a raised cosine measured in
**value units, not cell units**, so the bump slides smoothly instead of
snapping — a value halfway between two cells lights both, and a hair of
movement moves the code by a hair. `01_one_channel.png`, top left.

Blocks are contiguous and disjoint. The old encoders scattered each
channel's cells to random positions in one big array and had to work hard to
avoid collisions — `pop_regress.py` measures those collisions as the only
catastrophic errors that rig ever made, RMSE 0.157 -> 0.107 from the
de-collision line alone. But once no two channels share a position, that
scatter is **a permutation of coordinates, and every operation downstream is
permutation-equivariant** — dot products, normalisation, geodesic rotation.
So it is dropped. `profiles()` is a reshape.

## nb=2 is the ON/OFF split, and the two versions differ in one place

Rectifying a mean-centred image into `(relu(v), relu(-v))` is a two-cell
place code, so the guess was right. But `02_baseline_vs_unknown.png` shows
where they part:

| | at baseline | for a cell nobody wrote |
|---|---|---|
| `relu` split | silent | silent |
| nb=2 bump | both cells at 0.75 | silent |

**Rectifying makes "this pixel is exactly at the mean" indistinguishable
from "I have no information about this pixel."** The crossfading bump keeps
them apart, because at baseline both cells still speak. For a stack whose
read is *write half the vector and ask for the rest*, that distinction is
the mechanism, not a nicety.

## What nb buys is not fidelity. It is silence.

The surprise. Reading back with the centroid, **nb=2 already reconstructs an
image at 0.0397 RMSE.** Two crossfading cells carry the value almost
perfectly; more cells barely improve it.

| nb | 2 | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|
| image -> code -> image, centroid | 0.0397 | 0.0134 | 0.0063 | 0.0035 | 0.0024 |
| image -> code -> image, top-1 cell | 0.0909 | 0.0335 | 0.0143 | 0.0071 | 0.0049 |
| **fraction of cells lit** | **1.000** | 0.517 | 0.262 | 0.132 | 0.088 |
| **cosine of the two extreme values** | **0.471** | 0.000 | 0.000 | 0.000 | 0.000 |
| value gap until two codes are disjoint | never | 0.68 | 0.29 | 0.14 | 0.09 |

At nb=2 **every cell is lit for every value**, and black and white still
score 0.47 against each other. That is a dense code. It is nonnegative, so
it fixes the sign problem, and it fixes nothing else — no cell is ever
silent, so no cell can stand for anything in particular.

By nb=8 only a quarter of the cells are lit and any two values more than
0.29 apart are **exactly orthogonal**. That is the property the rest of the
project needs: a cell that says nothing about most of the range, and can
therefore mean something specific about the part it does cover.

So `nb` is the identity knob, and the reconstruction column is a red
herring — it was never the thing in short supply.

## halfw is the generalisation width, and 1.0 is wrong

`halfw` sets the fall-off radius in cells. It decides how fast two codes
stop overlapping (`01`, bottom middle), and it also decides whether the code
has a constant length as the value slides (`01`, bottom right):

| halfw | 1.0 | 1.5 | 3.0 | 6.0 |
|---|---|---|---|---|
| longest code / shortest, nb=16 | 1.414 | **1.029** | 1.177 | 1.279 |

At **halfw=1.0** the norm swings by √2 — exactly the partition-of-unity
case, where the weights sum to 1 but the *squares* run from 1.0 (value on a
cell) to 0.5 (value between two cells). A unit-normalised score would then
be reading the sub-cell phase as if it were signal.

The larger widths are worse for a different reason, and it is an **edge**
effect, not a ripple: a wide bump near the end of the ladder runs off it and
loses energy — the green curve is flat across the middle and collapses at
both ends. So either pad the ladder past the data range, or keep halfw near
1.5. The earlier claim that a place code is automatically norm-constant was
too strong; it is norm-constant *in the interior, at the right halfw*.

## Centre-dark, surround-bright — with no negative weight

`05_positive_template.png`. A 3x3 patch, nine channels, nb=8. The template
is the encoding of "surround bright, centre dark", normalised. Its weight
map is the whole answer: **positive weight on the bright cells of the eight
surround channels, and positive weight on the dark cells of the centre
channel.** Nothing negative anywhere.

| patch | score |
|---|---|
| centre dark, surround bright | **1.000** |
| centre bright, surround dark | **0.000** |
| flat mid grey | 0.000 |
| all bright | 0.889 |
| all dark | 0.111 |

The inverted patch scores **exactly zero** — the sign moved into position,
and a negative surround weight was never needed. Honest note on the margin:
all-bright scores 0.889, because it agrees with eight of the nine channels.
Centre-surround discrimination is one channel's worth, and that is a fact
about 3x3 patches, not about the code.

## A motor channel

`06_motor_channel.png`. One channel, nb=24 over thrust 0..8, a command
sweeping through it. The code field is a single bump tracking the command.
Read back, with cell spacing 0.348:

* top-1 cell: RMSE 0.105 — visibly a staircase, quantised at the cell
* centroid: RMSE 0.040 — **sub-cell**, about a ninth of the spacing

Which is the same trade as everywhere else in this project, at the smallest
possible scale: the peak gives you a name, the centroid gives you a number,
and you can have both off the same code.

## Files

| | |
|---|---|
| `place_code.py` | the encoder, `relu_split`, `patches_of` — no negatives anywhere |
| `run_place_code.py` | every figure and `results/summary.json` |
| `results/01_one_channel.png` | the bump, the nb dial, the similarity kernel, the norm ripple |
| `results/02_baseline_vs_unknown.png` | rectify vs crossfade at baseline |
| `results/03_image_bands.png` | a digit as nb nonnegative planes |
| `results/04_reconstruction.png` | what intensity resolution costs |
| `results/05_positive_template.png` | centre-surround with only positive weights |
| `results/06_motor_channel.png` | a command in and back out |

    python run_place_code.py     # ~20 s

## Next

`nb` and `halfw` are now knobs with measured meanings, and layer one can be
fed this instead of raw signed pixels. The open question is the one this
folder does not touch: what a hypercolumn *learns* on top of a place code,
with top-k speaking instead of one or all.
