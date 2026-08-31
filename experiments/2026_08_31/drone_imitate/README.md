# The drone flies — but the hypercolumns are barely doing it

2026-08-31. Four RL attempts (`../drone_rl`, `../drone_tracks`) all scored
0.000 and could not separate *the representation cannot express a good
policy* from *the learner cannot find one*. Imitation separates them: hand it
the teacher's answer and see whether it can hold it.

Two tracks and a tally, exactly as designed. The sensory track tiles states a
**competent** controller visits (which is the allocation the unsupervised RL
run got wrong); the motor track learns a vocabulary of 25 commands from the
teacher's own thrusts; the tally counts which command was used in which cell.
Reward and teacher never touch either track's templates.

## The result

| experts | cells used | **tally** | constant per cell | **local linear per cell** |
|---|---|---|---|---|
| 64 | 33 | 0.000 | 0.000 | **0.933** |
| 256 | 42 | 0.000 | 0.000 | 0.880 |
| 512 | 57 | 0.000 | 0.000 | 0.917 |
| *teacher (PID)* | | | | *0.970* |

**The architecture flies.** 0.933 against the teacher's 0.970, out of **33
cells**. So the representation was never the problem — four RL runs failed on
something else.

**And no cell-identity read-out works, at any resolution.** Not argmax over
the tally, not a blended weighted mean over motor categories, not a
temperature-sharpened blend, not the mean thrust per cell. All 0.000:

    N=64    tally argmax 0.000   blended 0.000   sharpened 0.000
    N=256   tally argmax 0.000   blended 0.000   sharpened 0.000
    N=512   tally argmax 0.000   blended 0.000   sharpened 0.000

## The control that deflates all of it

| policy | success |
|---|---|
| teacher (PID) | 0.970 |
| per-hypercolumn linear map, raw 6 sensors | 0.900 |
| per-hypercolumn linear map, full 72-d place code | 0.905 |
| per-hypercolumn linear map, winner's dense code (K=6) | **0.000** |
| **ONE global linear map, NO hypercolumns at all** | **0.875** |

**A single linear regression with no architecture gets 0.875.** The 33
hypercolumns buy 2.5 points over it. The flying is the linear map; the
partition is close to decoration.

The reason is the task: the teacher is a **PD controller, linear in the
state**, so one linear map represents it almost exactly and there is nothing
for a partition to do. This experiment cannot say anything about competitive
experts, because the problem it poses does not need them. That is a fault in
the choice of task, not a finding about the architecture.

It also explains the tally's 0.000 from the other side: the correct action is
a smooth linear function of the state, so every piecewise-constant read fails
and no partition helps, because there is no piecewise structure to exploit.

**And the winner's own dense code cannot drive the motors — 0.000.** `S =
W_n · q` is 6 numbers projected out of the 72-dim place code, and a linear
map on it cannot recover the control law, while a linear map on the full 72
can (0.905). So the failure is the 6-dimensional bottleneck, not the place
code. Worth knowing if anything downstream is ever meant to act on a
hypercolumn's dense output rather than on the raw values.

## What this settles

**Adaptive splitting would not have fixed it.** The plan before this run was
to split cells where the policy disagrees with itself. But resolution is not
the axis — 64, 256 and 512 experts all give 0.000, and the linear policy
already flies at 64. Worth having tested before building it.

**The rule, and it generalises past this task:**

> A tally works when the output is a **category**. It fails when the output
> is a **continuous quantity that must vary within a category**.

MNIST scored 0.8672 on a tally because a digit *is* a category — every image
in a cell wants the same answer. Thrust is not. The drone is unstable, so the
required differential thrust varies continuously and sensitively with tilt
and angular rate; a piecewise-constant command injects energy at every cell
boundary. You would need cells finer than the control precision, and in six
dimensions that is unreachable. The saturating `cells used` column (33 -> 42
-> 57 as N goes 64 -> 512) shows why more experts do not help: the teacher's
trajectories occupy a thin manifold that will not subdivide usefully.

**So an expert must contain a policy, not an answer.** Identity says *which
regime am I in*; the policy inside says *what to do given exactly where I am
in it*. That is the `[situation ; action]` subspace failing from the other
direction — it stored an answer per region when it needed a map per region.

## What would be needed to test the architecture at all

A task whose control law is **not** globally linear — so that different
regions genuinely need different rules and a partition earns its keep. Until
then this folder shows only that the pipeline can host a working policy, not
that the hypercolumns contribute one.

## Watching it fly

    VIEW_CKPT=experiments/2026_08_31/drone_imitate/results/imitate_64.npz \
        .venv/bin/python viewer.py

Select "pupil". This one actually flies.

## Files

| | |
|---|---|
| `imitate.py` | demos, both tracks, the tally, all three read-outs, `load_policy` |
| `results/imitate_{64,256,512}.npz` | tiling, tally, motor vocabulary, linear maps |
| `results/01_imitation.png` | the resolution curve |

    python imitate.py     # ~5 min

## Next

Put the local linear policy back into the RL setting: keep both tracks and
the identity gate, but fit each expert's policy by advantage-weighted
regression instead of counting. That is the one change the four failures and
this run jointly point at.
