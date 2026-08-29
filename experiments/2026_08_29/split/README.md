# The split rig

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

## The split rig (user's decision): joint encoding retired

Reasoning, recorded: the joint sensor+command template existed FOR
partial-cue command regeneration; with the cetele doing generation,
the joint distribution's only remaining effect was the attribution
trap (0.72 -> 0.00). Neuroscience framing: descriptive memory (PFC)
and action selection (basal ganglia) as separate organs — cortex
learns of the selected action through the loop, not by clustering its
state memory by command. Record precedent: sensory rows + tally beat
bound 61-to-3 (Aug 28 A/B).

`run_td_split.py`, `results/`: sensory-only top layer
(cmd_share = 0; present share 0.5 so the winning 50/50 present-
context cue ratio is preserved), tracks reused frozen from dagger3,
cetele notched every tick by the SAME masked question the read asks
(one partition), pupil flies the duty-cycle expectation read through
the tight-envelope DAgger phase. Watch-item on record: the command's
vote used to refine the partition by action; sensory aliasing at
decision states now rests entirely on the fovea/present encoding.
Smoke: agree 0.908 at 8k ticks — highest ever seen at any scale.
Target: match or beat the joint rig's 0.72 strict EVAL.

Console output `results/split.log`; smoke test `results/smoke/`. No result
for the full run was written up beyond the smoke line above.
