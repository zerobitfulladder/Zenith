# 2026-09-17 — counting, and names made from parts (HelloInductor line)

Words are defined in [`../2026_09_16/VOCABULARY.md`](../2026_09_16/VOCABULARY.md). The full
design record for this day (Parts XV–XVI) is kept in
[`../2026_09_16/glimpse_loop/README.md`](../2026_09_16/glimpse_loop/README.md). Both
experiments import `sdr.py` / `mnist.py` from that folder.

## `tally/` — a library built by counting only

Three scripts, each correcting the last. Naming what recurs (`tally.py`) compresses but
**costs 13 points** on the label, because what recurs most is what every digit has. Scoring
chunks by the surprise they remove (`tally2.py`) gives the first library that helps: +6 on
the label, +7 on unseen cells. Integrating as you go (`tally3.py`) works, but the library
helps prediction, not the label, in every variant; categories and as-if substitutions run as
designed and change nothing.

Full writeup: [`tally/README.md`](tally/README.md).

---

## `product_names/` — a card's name as its parts walked together

If a card's name is its parts bound together, the order of integration cannot matter. It
holds exactly: 20 orders of 5 parts give 1 name, and subtracting parts one by one recovers
the rest 100% of the time for cards of up to 20 parts, including whole sub-cards at once.
The price: product names have no graded similarity.

Full writeup: [`product_names/README.md`](product_names/README.md).
