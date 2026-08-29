# The deep outer

*Shared modules (`td_*.py`, `fast_oracle.py`, the viewer hooks `*_view.py`) live in [`../temporal_drone/`](../temporal_drone/README.md); the whole line of work is summarised in [`../temporal_drone/ARCHITECTURE.md`](../temporal_drone/ARCHITECTURE.md).*

## The deep outer (user's request): layers restored, and the tax measured

`gpu_deep_outer.py`. The champion's outer seat given the full layered
treatment: per-channel two-layer tracks (240/720 ms), combined at an
L3 with present bumps at 50%, same cetele, same vocabulary, same
reflex. Three runs, one variable at a time:

    present (dx,dy) only  [champion]        EVAL 1.00
    + temporal history (2-layer tracks)     EVAL 0.52
    + the angle channel                     EVAL 0.05

THE RELEVANCE TAX, quantified: the outer teacher's law is a function
of dx and dy alone. Every cue bit beyond that fragments the partition
— same position, different histories/tilts land on different
templates, the cetele's notches spread across the fragments, and the
law-bearing channels are outvoted by their own context. History cost
half; the (fully irrelevant, fast-varying) angle cost nearly all the
rest. The inverse of the glide trap (where the cue was too POOR for
the law): a cue can be too RICH for it. A layer's cue must carry
exactly what its decision depends on.

Corollary for the architecture: hierarchy is not free richness —
layers pay for themselves only where the LAW of the task needs what
they extract (hidden state, composition, invariance). Here the outer
law needed two present numbers, and two present numbers win.

`gpu_deep_outer.py` writes `results/` (the run with the angle channel);
`results/notilt/` is the run without it and `results/smoke/` a smoke test.
No separate result folder was kept for the present-only champion (that is
[`../cascade/`](../cascade/README.md)). The script imports `gpu_cascade` from
`../cascade/` and `gpu_chase` from `../chase/`.
