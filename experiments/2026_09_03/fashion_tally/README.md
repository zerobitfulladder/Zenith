# The tally-guided rig on Fashion-MNIST

2026-09-03. The rig from `../purity_plasticity` (purity step, competition =
similarity + 0.25 x label term + 0.5 x belief term) on Fashion, where features
are less generic than MNIST strokes. Same protocol: 12k/3k, split
{tshirt, trouser, pullover, dress, coat} -> {sandal, shirt, sneaker, bag, boot}
-> all ten, and joint 0-9 alongside. 9x9 patches (3 epochs/phase, batch 128)
and 28x28 whole images (20 epochs/phase, batch 512). 2 seeds.

```
                      JOINT recount  online    gap  dead |  SPLIT after 5-9 old/new   final  online  #com  drift  conv  dead
9x9
champion                     0.8015  0.7147  0.087     0 |         0.8172 / 0.7834  0.8063  0.4385     0
cnt/n none                   0.8068  0.8055  0.001     0 |         0.8172 / 0.7804  0.8060  0.8052     0
cnt/n  L0.25+B0.5            0.8038  0.7648  0.039     0 |         0.8159 / 0.7837  0.8040  0.6098   396  0.102  0.38     0
purity L0.25+B0.5            0.8063  0.7618  0.045     0 |         0.8205 / 0.7945  0.8048  0.5805   398  0.331  0.40     0
28x28
champion                     0.7520  0.7325  0.020   238 |         0.7846 / 0.6333  0.7243  0.7010   168  0.004  0.01     0
cnt/n none                   0.7543  0.7548 -0.001   279 |         0.7352 / 0.6026  0.6920  0.6845    22  0.026  0.05   318
cnt/n  L0.25+B0.5            0.7877  0.7785  0.009   269 |         0.7839 / 0.8252  0.8158  0.8083   196  0.000  0.01     0
purity L0.25+B0.5            0.7920  0.7830  0.009   256 |         0.8152 / 0.8107  0.8167  0.8130   210  0.002  0.01     0
```

`champion` = cnt/n step with belief 1.5, the MNIST champion's rule. `cnt/n none`
= no class pressure at all. `#com` = templates committed (row purity > 0.9)
after phase A; `drift`/`conv` = how far they moved / how many changed class
through phase B.

## 28x28: nine points on the split, four joint

The whole-image champion cannot learn the new classes in phase B: after 5-9 its
new-class accuracy is **0.6333**, and the final recount is 0.7243. The summed
bias learns them at **0.8252** and finishes at 0.8158; with the purity step,
0.8107 and **0.8167**. Old classes after phase B: purity 0.8152 against the
champion's 0.7846. The online table sits within 0.4 points of the recount for
both summed arms, against 2.3 for the champion. Both seeds agree on every
column.

Why the champion fails here is the sink from `../plasticity_rf`, seen from the
other side: its belief says a boot is a coat, the bias sends the boot to the
coat templates, and those are old and slow (cnt/n) so they neither move nor
release it. The label term is the only thing in the grid that says "not
there", and once it does, boots go to fresh templates. Joint training says the
same with nothing to forget: 0.7920 / 0.7877 against 0.7520.

## 9x9: nothing to protect, and pressure only makes the table stale

Old classes after phase B are 0.816-0.821 for all four arms. Final recount
0.804-0.806 for all four. On Fashion patches, as on MNIST patches, the
vocabulary is shared across classes, drift is free, and there is nothing for a
protection rule to win. The one large effect is the online table on the split:
with no pressure it equals the recount (**0.8052 vs 0.8060**); the champion's
belief pressure drops it to 0.4385; the summed bias to 0.58-0.61. This is
`../whole_digit` §3 again: any bias that steers winners by the table's own
beliefs invalidates earlier counts when the vocabulary is generic. For a patch
layer on a stream, the right rule is still cnt/n and no pressure.

## What Fashion adds

- **The regime line holds across datasets.** Object-scale templates: the tally
  guides and protects, and the gain grows with how much the champion was losing
  (2 points on MNIST, 9 on Fashion). Part-scale templates: nothing, and the
  purity step drifts (0.331 vs 0.102) exactly as on MNIST.
- **On Fashion the whole-image rig with the new competition matches the 9x9
  patch rig** (0.7920 vs 0.8063 joint; 0.8167 vs 0.8048 on the split). On MNIST
  it was 0.93 against 0.97. Fashion's patch rig has less of an edge to begin
  with, and the tally-guided whole-image rig closes it. This is also the
  setting for a template-count sweep: the prototype-count law says more
  whole-image templates buy diminishing but real accuracy, and here the
  starting point is level with the patches.
- `cnt/n none` at 28x28 leaves 318 dead templates after the split and learns
  the new classes worst of all (0.6026): with no pressure of any kind, nothing
  recruits and the phase-A prototypes absorb everything.

## Caveats

Two seeds, 12k/3k. The project's best Fashion number (0.8989) uses per-cell
readout and 40k images; nothing here is tuned for absolute accuracy. Label 0.25
and belief 0.5 are carried over from MNIST untouched. The 9x9 split uses 3
epochs per phase and 28x28 uses 20, following `../plasticity_rf`, so the two
sizes are not matched in updates per template.

## Files

| | |
|---|---|
| `fashion_split.py` | both sizes, four arms, joint and split. `--smoke` |
| `results/fashion_split.png` | old / new / online through the split, both sizes |
| `results/fashion.json`, `.log` | every number, per seed, per probe |

    uv run python fashion_split.py    # ~8 min
