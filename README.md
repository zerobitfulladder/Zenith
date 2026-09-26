# Zenith

A personal research project on learning without gradients.

A layer is a set of stored patterns, called **templates**, each a unit-length vector. Templates
compete for every input, and only the ones that win move a little toward what they saw, by
rotating on the unit sphere. Everything the system knows is read back out of those same
templates: what an input is, what it looks like, and what comes next. There is no loss
function, no backpropagation and no batch. Inputs arrive one at a time, and learning something
new should not erase what was learned before.

The words are borrowed from the cortex. A template is also called a **minicolumn**; a
**hypercolumn** is a group of templates competing over the same input; a **layer** is one or
more hypercolumns. The goal is for each template to mean something on its own (a stroke, a
part, a whole digit, a moment in time), and for more precision to come from more templates
rather than from denser codes.

---

## Highlights

Test accuracy unless stated, with learning switched off while testing. Single-seed numbers are
marked. Every folder is under `experiments/`.

| Task | Result | How | Folder |
|---|---|---|---|
| MNIST | 0.9869 ± 0.0009 (3 seeds) | one wide layer of 5x5 patch templates, win counts kept per image position, one linear map on top | [`2026_09_01/stack/`](experiments/2026_09_01/stack/) |
| MNIST, nothing fitted | 0.9730 | the same layer, read only by counting which templates won for which digit | [`2026_09_01/`](experiments/2026_09_01/) |
| MNIST, small | 0.9492 with 103k parameters | two layers of competing templates, each passing on only which template won | [`2026_08_31/kmeans/`](experiments/2026_08_31/kmeans/) |
| MNIST, whole-digit templates | 0.946 (3 seeds) | one hypercolumn over image + label, step size set by how sure each template is | [`2026_09_03/purity_plasticity/`](experiments/2026_09_03/purity_plasticity/) |
| Fashion-MNIST | 0.8989 ± 0.0017 (3 seeds); linear model on pixels 0.8512 | as for MNIST | [`2026_09_01/stack/`](experiments/2026_09_01/stack/) |
| No forgetting: digits 0–4, then 5–9 | old digits 0.9757 → 0.9526 (a standard network: 0.9765 → 0.0000) | 40 hypercolumns; only the winning hypercolumn learns | [`2026_08_31/continual_fixed/`](experiments/2026_08_31/continual_fixed/) |
| Filling in a deleted slice of a curve | error 0.033 vs 0.149 for a standard network | a second layer over windows of the first layer's answers | [`2026_08_29/two_layer/`](experiments/2026_08_29/two_layer/) |
| Simulated drone, chasing a target | 100/100 episodes, teacher parity | a slow counted memory (10 Hz) steering a fast reflex (50 Hz) | [`2026_08_29/cascade/`](experiments/2026_08_29/cascade/) |
| Cart-pole swing-up | 87/100 (single run) | two-track pole rig, imitation of an energy-pumping teacher | [`2026_08_28/pole2track/`](experiments/2026_08_28/pole2track/) |

---

## Day by day

Each day folder has a README indexing its experiments; each experiment folder has its scripts,
a README with the full writeup, and a `results/` folder.

**March — the AxonForge era.** Built as nodes in AxonForge, a personal node-graph tool.

- [**2026-03-16**](experiments/2026_03_16/) — The first learning rules for one layer on MNIST,
  ending in contrast-residual learning (0.917 MNIST, 0.826 Fashion with a linear readout).
- [**2026-03-19**](experiments/2026_03_19/) — Signed residual learning: every template takes
  part with its signed similarity, tracked for how dense the code becomes.
- [**2026-03-20**](experiments/2026_03_20/) — Top-down gain feedback in a two-layer stack; the
  stack without feedback read MNIST better in all five settings.
- [**2026-03-22**](experiments/2026_03_22/) — The label as an extra input row, read back from
  how the layer fills it in (0.77 with 100 templates, near chance with 500).
- [**2026-03-26**](experiments/2026_03_26/) — Competition as partial correlations: diverse
  templates, poor reconstruction (notes only).
- [**2026-03-28**](experiments/2026_03_28/) — Plain top-1: only the best-matching template
  learns, and no single template takes over (notes only).
- [**2026-03-29**](experiments/2026_03_29/) — The top-1 unit as a memory on synthetic sparse
  patterns: completion, generalisation, and a hidden permutation.

**August — rebuilding the unit, time and control.**

- [**2026-08-05**](experiments/2026_08_05/) — The hidden-permutation arc: recognising pairs
  does not turn into generating them, but giving each hypercolumn a local window nearly
  doubled generation.
- [**2026-08-23**](experiments/2026_08_23/) — Top-down feedback tested and closed; the unit
  rebuilt from patches into the consolidated unit, trained on the GPU, used on CelebA faces.
- [**2026-08-24**](experiments/2026_08_24/) — Sparse messages between layers, the top-layer
  memory, and a program of feedback experiments on the spatial rig.
- [**2026-08-26**](experiments/2026_08_26/) — Bits per weight, five rigs for learning over
  time, and first motor control (cart-pole, pole from angle only, a reaching arm).
- [**2026-08-27**](experiments/2026_08_27/) — Can the arrow of time come from completion up
  the layer stack instead of explicit "next" slots?
- [**2026-08-28**](experiments/2026_08_28/) — Reconstruction quality becomes the objective;
  a deep cart-pole swing-up and the first drone rigs.
- [**2026-08-29**](experiments/2026_08_29/) — Filling in a curve with one, two and
  convolutional layers; then the drone over time, up to the cascade that chases 100/100.
- [**2026-08-30**](experiments/2026_08_30/) — Templates that explain the input together:
  crisp rebuilds, meaningless templates, and what can still be built on them.
- [**2026-08-31**](experiments/2026_08_31/) — Competition (only the winner learns) is what
  creates identity; every representation has a readout it is good at.

**September — counting, search, and a second line of work.**

- [**2026-09-01**](experiments/2026_09_01/) — One wide layer plus counted tables is the best
  thing so far (0.9869 MNIST, 0.8989 Fashion), and nothing stacked on top beat it.
- [**2026-09-03**](experiments/2026_09_03/) — Which templates may move: forgetting, how much
  learning the templates is worth, and a drone that sees.
- [**2026-09-07**](experiments/2026_09_07/) — Olshausen & Field's sparse coding rerun on our
  patches, and why most of its templates die.
- [**2026-09-12**](experiments/2026_09_12/) — A sparse code that names the groups it finds,
  then groups of groups, by counting and search.
- [**2026-09-13**](experiments/2026_09_13/) — Search in place of backprop, settling in place
  of search; strokes appear when active cells share the input.
- [**2026-09-14**](experiments/2026_09_14/) — A theory-first reset, then a second layer as a
  counted belief network.
- [**2026-09-15**](experiments/2026_09_15/) — A machine that writes its own programs, learned
  by counting.
- [**2026-09-16**](experiments/2026_09_16/) — Start of a line imported from another project:
  a memory that looks at an image a little at a time (74.6% MNIST).
- [**2026-09-17**](experiments/2026_09_17/) — A library of parts built by counting alone.
- [**2026-09-18**](experiments/2026_09_18/) — Codes as angles moved by prediction error, and
  whether credit survives four levels.
- [**2026-09-20**](experiments/2026_09_20/) — What a layer of angle-coded units learns to
  measure, and a two-track build that reads MNIST at 92.4%.
- [**2026-09-21**](experiments/2026_09_21/) — Networks made of angles, trained with
  gradients: an exactly invertible network at 96.2% MNIST.
- [**2026-09-23**](experiments/2026_09_23/) — The basics again, in conversation: cortical
  layers, thalamic loops, bands and phases. Then a test of the idea that a level needs only a
  coarser image, not the level below. It lost to a CNN with everything else equal, and with
  one filter bank per scale it lost to raw pixels. Dropped.
- [**2026-09-26**](experiments/2026_09_26/) — Three days on reasoning: composition in time
  over live maps, attention as a window, pooling made dynamic by gain and normalisation, and
  the feedback path as a decoder. Then three tests: a backward path trained only by local
  mismatch renders and imagines from a 4x4 top map once its unpool is learned; a bag of
  features plus one pointer for the whole object cannot render at all; and attending
  windows one at a time and sweeping them rebuilds the digit sharper than one look, down
  to a quarter of the pixel error with sixteen windows of 8. Last, both directions trained
  at once with no labels: "copy your twin" collapses to silence, "rebuild the level below"
  gives a stack that renders, imagines, and reads out at 94% without being asked to. And
  faces at full crop: the decoder renders attributes from their mean codes and sharpens an
  attended mouth, but a mustache added to a woman brings the man with it, sampling a
  window's missing detail from a squared-error model gives an average, not a crisp part, and
  a render-read-correct loop helps for one cycle and then satisfies the stack instead of the
  face. Last, a painter that draws parts from a library and keeps the ones that make the
  whole read as the goal: sharp and consistent faces, real and imagined, for the first time;
  a faint mustache on a woman who stays a woman; and, given only lines and ovals, a drawing
  the stack reads as the face and a person does not.

---

## Notes

- This is a personal research log, not a library. Code is written to answer one question at a
  time and is not maintained afterwards.
- The folders were reorganised on 2026-09-23. Scripts that import code from another
  experiment folder may still point at the old layout, so some will not run as they are.
- The March node files (`learning*.py` and the helpers in `2026_03_16/shared_nodes/`) only ran
  inside AxonForge and are kept as a record. Most March scripts also read MNIST from
  AxonForge's data cache, which no longer exists.
- The 2026-09-16 to 09-21 scripts come from another project and read data from that
  project's layout. `torch` was added to the project on 2026-09-23.
- Trained weights and checkpoints (`*.npz`, `*.npy`, `*.pt`) are not in git. Datasets go in
  `data/` (`mnist/digits`, `mnist/fashion`, `celeba`, `cifar10`).
- `viewer.py` replays any trained drone or pole controller. `READING_LIST.md` gives the
  established names and papers for the ideas used here.
