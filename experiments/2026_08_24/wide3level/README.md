# 2026-08-24 — `wide3level/`: 3-level wide rig

Part of the day log [`../README.md`](../README.md).

## 3-level wide rig — predictions (before running, 2026-08-24)

User hypothesis: MNIST needs only 3 levels if they are wide enough.
Sharper motivation from today: every level is an interface, and the
measured costs (sparse-speech loss, expansion loss) are per-interface.
`run_wide3level.py`: L1 8x8 s1 K=512 -> wide L2 (3x3 s2, grid 10,
K2 in {256, 512}) -> memory directly on L2's code (100*K2 + 10 dims).
Arms: 256-dense, 512-dense, 256-top3 (top3 speech at BOTH interfaces).
References: 4-level champion dense .9094 / probe .9668; 4-level
all-top3 .7596; L1-only top3 .8642.

1. Dense 3-level matches the 4-level champion within ~1pp (hard ~.90,
   probe ~.965) — depth was not doing the classification work. If it
   does match, the 3-level rig becomes the preferred platform
   (simpler, one less interface).
2. Generation clean with a single expansion — possibly crisper than
   4-level (fewer hardened expansions to compound).
3. All-top3 at 3 levels lands ~.80-.83, clearly above the 4-level
   all-top3 (.7596) — one less sparsified interface, one less cost.
   If it approaches L1-only top3 (.8642), interface count is the
   dominant cost variable, confirmed.
4. K2=512 vs 256: small differences (upper-palette lesson says width
   there is cheap but not very useful).

### Outcome (same day) — 3 levels suffice for REPRESENTATION; depth
### earns its keep at the memory; the sparse cost is a FINAL-INTERFACE
### phenomenon

`results/`:

| K2 | out | probe L2 | hard | consistent |
|---|---|---|---|---|
| 256 | dense | .9660 | .8728 | 10/10 |
| 512 | dense | .9618 | **.8972** | 10/10 |
| 256 | top3 | .9644 | .7616 | 10/10 |

- P1 half: probes MATCH the 4-level champion (.966 ~ .9668) — the
  user is right that 3 wide levels carry MNIST's information. The
  lookup falls short: -3.7pp at K2=256, -1.2pp at K2=512 (borderline).
- P2 ✗: generation is legible but visibly SOFTER than 4-level —
  chunky, mushier strokes. Depth's L3 abstraction stage was earning
  its keep in prototype crispness, not in the probes.
- P3 ✗, productively: all-top3 at 3 levels = .7616 ~ the 4-level
  all-top3 (.7596). Removing an interface recovered NOTHING. Combined
  with the day's other numbers, a cleaner law emerges: what matters is
  whether the code THE MEMORY STORES is dense. Memory fed dense:
  .8642-.9094 (regardless of sparse interfaces below). Memory fed
  sparse: ~.76 (regardless of depth). The sparse-speech cost is
  concentrated at the FINAL interface, not spread per-interface —
  the compounding story needs this amendment.
- P4 ✗: K2=512 clearly beats 256 (+2.4pp hard) — in a 3-level rig L2
  is the sole compositional stage and width there pays, unlike in the
  4-level.

VERDICT: the 4-level 8x8/512 dense rig stays champion (.9094, crisp
generation); the 3-level 512-wide rig is a legitimate fast variant
(-1.2pp, one less layer) and the proof that representation needs only
3 levels. Depth's real deliverables on MNIST: lookup margin and
generation crispness.
