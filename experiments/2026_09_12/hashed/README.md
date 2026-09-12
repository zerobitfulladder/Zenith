# No minting decision: hashed, soft, decaying

2026-09-12. Same 100-light world as [`../sparse_patterns`](../sparse_patterns/README.md).
One seed, no sweeps beyond one diagnostic, nothing here is a measurement.

Every version so far has had the same weak point: **naming is a discrete,
irreversible act.** Something either becomes a pattern or it doesn't, and once
it does it owns a slot forever. That decision has broken every version — it
needed a warm-up so the counts weren't noise, it produced duplicates because two
separate naming events could create two slots for the same thing, and a name
minted badly on input 30 was permanent.

This removes the decision entirely.

## The four changes

**Hashed.** A friendship between slot *a* and slot *b* lives at `hash(a,b)`.
Nobody decides to create it — its address is a function of what it is. The same
pair found twice lands in the same place, so **duplicates are impossible by
construction** rather than by a check that can fail.

**Paired, not set-hashed.** A group is never hashed as a whole. `{4,5,6}` is
`hash(4,5)` and then `hash(that, 6)`. If 6 drops out the first half still fires,
so a missing member costs the top of the chain instead of everything. Hashing
the set would be all-or-nothing, which is exactly what we were trying to escape.

**Soft.** A slot holds a strength, not a bit. A weak friendship sits at 0.05
rather than being either present or refused.

**Decaying.** Counts fade, using a running scale factor rather than touching
every entry. Evidence that stops arriving loses its grip, so a bad friendship
learned early is not permanent. And there is no warm-up: it learns from input 1,
because early nonsense is *quiet* nonsense rather than a structural commitment.

## What it does

```
2,000 vectors in 5 seconds
```

```
                          speculates   adds BOTH B,C   cue A+B adds C   F1 adds F2   ambiguity
hashed, 2000 vectors            0.77            0.00             0.00         0.00        0.71
hashed, 1500 vectors            0.77            0.39             0.36         0.00        1.00
../recursive (beam 4)           1.00            0.00             0.00         0.00        1.00
```

Read those first two rows together, because they are the same engine at the same
settings with 500 fewer training vectors. **The violation rate moves from 0.00 to
0.39 and the ambiguity score moves the other way, 0.71 to 1.00.** That is not a
system that enforces the constraint and needs tuning. That is noise.

The honest verdict is therefore: **it does not reliably do the thing.** On one
training length it looks clean; on another it produces the illegal triple a third
of the time. I wrote this section up as a success before running the sweep, and
that was wrong.

## Why: the hash space saturates, at every setting

```
17,462 of 20,000 hash slots addressed
```

Almost nothing recurs often enough to stabilise. My first guess was that the
top-K truncation caused this — that ranking pairs against whatever else happened
to be in that particular vector made the selection inconsistent, and that keeping
everything above a fixed strength would fix it.

`sweep.py` says the opposite. Loosening the truncation makes it categorically
worse:

```
    K   secs   slots used   speculates   both B,C   A+B->C   F1->F2   ambiguity
   20    4.5        16229         0.77       0.39     0.36     0.00        1.00
   60   36.7        20000         1.00       1.00     1.00     1.00        0.50
  120  188.3        20000         1.00       1.00     1.00     1.00        0.50
```

At K = 60 the hash space is **completely full**, every address shared by
unrelated pairs, and the engine violates every constraint including the
second-order one that plain bit-counting gets right. It also costs 8x the time.

So the truncation was not the disease, it was the only thing holding the disease
back. The disease is that pair-chaining generates far more distinct friendships
than any hash space can hold: 24 lights make 276 pairs, those make 38,000, and
raising K just lets more of that flood through. Tightening K is damage control,
and even the tightest setting tested does not saturate *less* -- 16,229 of 20,000
-- it merely fills slower.

Also worth recording: the guess threshold is completely inert — the probe
numbers are identical from 4.0 down to 0.0.

## The first attempt, which failed differently

With 2,000 hash slots instead of 20,000, **all 2,000 were addressed within a few
hundred inputs** and every slot was shared by many unrelated pairs. Expanding
one gave nonsense, and the engine happily put F2 next to F1 — a mistake even the
plain bit-counting baseline never makes. The size of the hash space is not a
detail.

That forced a second change: a dense notebook over 20,000 slots would be 3 GB,
so the counts are kept as a neighbour list per slot instead. Which is the same
constraint I flagged for MNIST, arriving a step earlier than expected.

## Verdict

The four mechanisms all work as designed and are worth keeping: hashing really
does make duplicates impossible, pair-chaining really does degrade gracefully,
decay really does remove the need for a warm-up, and the engine really does learn
from input 1.

But the thing as a whole does not hold the constraint reliably, and the reason is
that **removing the naming decision also removed the thing that kept the
vocabulary small.** A minting step with a duplicate check was doing two jobs, and
only one of them was obvious. The second was refusing to create anything new
most of the time — which is what kept 11 atoms at 13 patterns instead of 17,462.

The next idea should not be a better truncation rule. It should be a reason for
the space of friendships to stay small on its own.

## Files

```
engine.py   the whole thing
run.py      one seed, the worked example and the four probes
sweep.py    the diagnostic that refuted the top-K explanation
```
