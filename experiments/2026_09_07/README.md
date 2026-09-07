# 2026-09-07

## `sparsenet/` — Olshausen & Field 1996 on 8x8 MNIST patches

Their cost, their settling and their learning rule, on our patches, to see what
their templates look like next to ours. Run as written, the paper kills most of
its own templates (46 of 64 end at length zero and never fire), and its pooled
coefficient histogram hides this: pooled kurtosis 18.8, per template 2.6. The
choice of sparseness cost is the lever: with |x| per-template kurtosis is 103.9
against 2.3 for log(1+x^2) at the same reconstruction. The templates are
general-purpose filters, not digit parts — reconstruction fixes the subspace,
sparseness fixes the axes. Unit-length templates remove the death mode with no
re-seeding (0 dead in every arm), and unit-length patches make lambda mean one
thing everywhere. Adding sparsity or independence terms to the learning rule
buys nothing at matched reconstruction, and a random dictionary has the lowest
coefficient dependence of all, so independence is only readable at matched
reconstruction. `rebuild.py` then rebuilds whole digits from the learned
templates.

Full writeup: [`sparsenet/README.md`](sparsenet/README.md).
