# Winner, then residual winner (killed after the 9x9 half)

2026-09-03. The proposal: after the winner learns from the patch, compete again
on what it could not explain and let only THAT winner learn from the residual
(matching pursuit, winner-only at each stage). The table counts and reads both
winners per position. Five arms, MNIST 12k/3k, 3 epochs per phase, 2 seeds.
The 9x9 half finished; the 5x5 half was killed as uninformative and slow.

```
9x9              JOINT recount  online  dead  coher  stage2 |  SPLIT after 5-9 old/new   final  online
champion                0.9675  0.9307     0  0.880    0.00 |         0.9738 / 0.9649   0.9667  0.7025
top1 none               0.9710  0.9718    16  0.712    0.00 |         0.9721 / 0.9672   0.9713  0.9717
top2 read               0.9650  0.9695    16  0.712    0.00 |         0.9691 / 0.9580   0.9648  0.9705
chain 0.0               0.9595  0.9570     0  0.726    1.00 |         0.9661 / 0.9477   0.9583  0.9557
chain 0.5               0.9660  0.9638     0  0.727    0.54 |         0.9694 / 0.9633   0.9672  0.9647
```

`coher` = mean over templates of the largest |cos| to another template.
`stage2` = share of patches on which the residual winner fired.

- **The chain recruits everyone and costs accuracy.** Dead 16 -> 0; -1.2 points
  with the residual always learning, -0.5 with the surprise gate (residual larger
  than half the patch). Coherence did not fall. On MNIST parts what a good stroke
  leaves over is not class-informative, and the second stage never abstains, so
  a noise residual still gets a winner that is counted and read.
- **Two things dilute.** At read time the second vote is near-random; in the
  tally both stages write the same row, so a clean stroke template's row is
  blurred by the leftovers it catches as a second winner. The fix, untested:
  separate rows for "t as first winner" and "t as residual winner".
- **Reading the two nearest hurts** (-0.6): the second-nearest stroke is a worse
  stroke, not extra evidence.
- **Top-1 with no pressure is the patch-layer stream rule.** 0.9710 joint, online
  table equal to the recount, same on the split. The champion's belief pressure
  costs 4 points of staleness joint and 27 on the split.

`xor.py`: XOR with templates + tally. Fails under the rig's centre-and-drop
preprocessing (two of four inputs are flat, that is where the bias lives),
fails by hogging with a constant channel (one learned template is nearer the
other inputs than any random one), solves 20/20 with a constant channel plus
hire-on-surprise at 4 templates.

    uv run python chain.py      # ~4 min per arm at 9x9; 5x5 is 2x and memory-bound
    python3 xor.py
