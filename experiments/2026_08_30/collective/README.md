# Templates that explain the digit together

2026-08-30. Testing the doubt directly: **are minicolumns a codebook, where
each one has to account for the input alone, or a dictionary, where they add
up?** The reconstructions in the old `learning*.py` runs were low-error and
crisp, and that is real. This asks what that crispness was actually evidence
for.

One layer, one hypercolumn, 144 templates, MNIST. Every template sees every
digit. Preprocessing is the usual one: subtract the image's own mean, scale to
unit length.

## The rule

Score, rebuild, and hand each template what the others left over:

    s      = W x̂                    one score per template, signed
    recon  = Wᵀ s                    everyone's contribution added up
    others = recon − sᵢ wᵢ           what the *others* built
    rᵢ     = x̂ − others = e + sᵢ wᵢ  the residue left for template i

then rotate wᵢ toward rᵢ along the sphere. This already existed in the repo —
`experiments/2026_03_19/srl/learning4.py` is this rule exactly. Three ways to take the step were
run against each other.

### The literal reading cancels itself out

Rotating toward `rᵢ` as-is means projecting it onto the tangent plane at `wᵢ`
first, and there the `sᵢ wᵢ` term **disappears entirely**:

    τᵢ = rᵢ − (rᵢ·wᵢ) wᵢ = e − gᵢ wᵢ,     gᵢ = e·wᵢ

The per-template residue is gone. Every template chases the *same* leftover
`e` and differs only by where it happens to be sitting. Averaged over data the
drift is `(I − wᵢwᵢᵀ)(I − WᵀW) μ` — a first-moment rule, so it only ever sees
the mean image. `check_math.py` verifies the identity and that the fast path
matches a literal transcription of the definition to 4e-8.

That is `raw` in the table, and it is broken as predicted: it never gets past
0.71 error. Turning the step up doesn't rescue it, it detonates — the
templates pile into a shared direction and the rebuild *overshoots the digit
by eight times its own size*:

| `raw` step | 0.05 | 0.30 | 1.00 | 3.00 |
|---|---|---|---|---|
| rebuild error | 0.816 | **8.20** | **9.67** | **6.21** |

The fix is to weight the leftover by how loudly the template claimed the
digit — rotate toward `sᵢ·rᵢ`, which reverses for `sᵢ < 0`. Same direction,
but now carrying `sᵢ`, so it is second-moment driven and is the actual
gradient of the rebuild error. That is `weighted`, and it works. `unit` is
learning4.py's own answer to the same problem: normalize `others` to unit
length before subtracting, which smuggles `sᵢ` back in through the sign and
the `1/‖others‖`.

Both `raw` and `weighted` need no (144×784) scratch array at all — the update
is `W ← diag(a)W + b eᵀ`, about 590 µs per digit.

## It works. Beautifully. On the wrong thing.

30k digits, 3 epochs, held-out test set. Error is relative, lower is better.

| | rebuild error | effective rank | overlap with best 144 directions | largest similarity between two templates |
|---|---|---|---|---|
| best possible (top-144 subspace) | **0.200** | — | 1.000 | — |
| **weighted** | **0.228** | 144.0 / 144 | **0.974** | **0.004** |
| unit (learning4.py) | 0.252 | 143.8 / 144 | 0.887 | 0.017 |
| codebook — winner alone | 0.585 | — | — | — |
| raw | 0.711 | 121.2 / 144 | 0.217 | 0.164 |
| untrained, 144 random directions | 0.925 | — | 0.181 | 0.163 |

So the doubt was well-founded on the facts. Collective explanation **more than
halves** the codebook's error and lands within 14% of the best that any 144
directions could possibly do. In `rebuilds.png` the collective rows are sharp
where the codebook row is a blur — the codebook has to answer a `5` with the
nearest single stored `5`, so it returns an average of fives.

**And then look at `templates_weighted_final.png`.** Speckle. Every one of the
144. Not a stroke, not a loop, not a fragment of a digit — 144 panels of
red-and-blue noise. `templates_codebook.png` next to it is 144 legible
handwritten digits.

The numbers say the same thing the picture does. Largest similarity between
any two templates: **0.004** — they are orthogonal. Effective rank **144.0 out
of 144**, overlap with the best 144-dimensional subspace **0.974**. The layer
did not learn 144 parts. It learned an arbitrary orthonormal basis of MNIST's
principal subspace, and any other basis of that subspace would have done just
as well.

## Why nothing could have prevented it

Not a tuning failure, and not fixable by tuning. The rebuild is `WᵀW x`, and
for any rotation `R` of the templates among themselves:

    (RW)ᵀ(RW) = Wᵀ Rᵀ R W = WᵀW

The rebuild is **identical**. `check_rotation.py` scrambles the trained
templates and measures the error again: `0.228429` before, `0.228429` after, a
difference of exactly `0.00e+00`, while the templates themselves move by 0.315.

The error being minimised is blind to which basis the templates form. It
cannot prefer digit-shaped ones, so it doesn't. The training curves show no
window where it might have — the templates go from random speckle to
orthogonal speckle without ever passing through anything meaningful
(`templates_weighted/n0002400.png` is already noise).

## What this settles

**The old low reconstruction error was real, and it was not evidence of a
dictionary of parts.** 144 directions spanning MNIST's principal subspace
rebuild digits crisply. That is a fact about the subspace, not about the
templates — the crispness came for free and would have come from any basis at
all, including a randomly scrambled one.

So the two readings buy opposite things, and this run prices both:

* **codebook** (each explains alone) — 0.585 error, and 144 templates you can
  look at and read.
* **dictionary** (explain together) — 0.228 error, and 144 templates that mean
  nothing individually.

Crisp rebuilds are not the reason to prefer the dictionary reading, because
crisp rebuilds do not require the templates to have learned anything.

## The next thing to try

The rotation symmetry is the whole problem, so break it. Keep collective
explanation but stop letting all 144 templates answer every digit — force the
scores sparse (top-k, or a penalty) and `(RW)ᵀ(RW) = WᵀW` no longer holds,
because a rotation destroys which templates are the large ones. That is what
makes sparse coding produce strokes on MNIST where this produces speckle. It
is also the version that keeps contact with a hypercolumn, where a handful of
minicolumns speak and the rest stay quiet.

## Files

| | |
|---|---|
| `collective.py` | the hypercolumn, three learning rules, the codebook control |
| `run_collective.py` | training, snapshots, diagnostics, baselines |
| `check_math.py` | the fast update equals the literal rule (4e-8) |
| `check_rotation.py` | the objective cannot see which basis the templates form |
| `results/templates_<rule>/n*.png` | templates through training, 12 log-spaced points |
| `results/rebuilds.png` | held-out digits, every rebuild side by side |
| `results/training_curves.png` | error, rank, overlap, similarity vs digits seen |
| `results/summary.json` | every number above |

    python experiments/2026_08_30/collective/run_collective.py                    # everything, ~5 min
    python experiments/2026_08_30/collective/run_collective.py --smoke            # 2k digits, sanity check
