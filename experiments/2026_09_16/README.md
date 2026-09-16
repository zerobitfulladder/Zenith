# 2026-09-16 — the glimpse loop (HelloInductor line)

This day starts a line of work imported from another project ("HelloInductor"), which runs
through `2026_09_21`. Its words (note, voice, choir, cell, ...) are defined once in
[`VOCABULARY.md`](VOCABULARY.md).

## `glimpse_loop/` — a memory that looks a little at a time

The idea under test, working title "learn the representation, memorize the function": no
network learns the answer; a memory with strict rules about how things are written, combined
and held (about 20 items) forces looking a little at a time, compressing what is recognised,
guessing and checking.

- **Toy world.** The loop recovered the hidden part library exactly (5 of 5 sub-assemblies,
  10 of 10 objects, zero junk), recognised 7-cell objects in about 6 glimpses with zero wrong
  declarations, and looking where hypotheses disagree was worth 87% vs 23% once the library
  was rich.
- **MNIST.** 74.6% test accuracy, 93% right when confident; rotation as an operator took
  rotated digits from 20% to 57–62%. The learned look policy matched the computed one and
  neither beat random looking: digits are not decided by one cell. The 8,000-digit run
  crashed on a ghost read from working memory, the failure the substrate itself predicts.

The README holds the session narrative for this day and the full design record (`DESIGN.md`),
which also covers the next two days' scripts.

Full writeup: [`glimpse_loop/README.md`](glimpse_loop/README.md).
