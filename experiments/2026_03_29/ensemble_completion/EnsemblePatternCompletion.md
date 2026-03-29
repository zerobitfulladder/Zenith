# Ensemble Pattern Completion Experiments

## Notes from experiments — 2026-03-29

---

## Single-Node Pattern Completion

### Setup

Standalone Zenith algorithm (no AxonForge framework) tested on synthetic sparse binary vectors. Sweep over template count (k), pattern count (N), sparsity, and mask fraction. Train the node, then zero out a fraction of active bits and check if the correct template still wins.

### Key findings

- At low load (N/k ~ 0.2), retrieval is perfect even at 80% masking. The templates are so far apart on the hypersphere that even heavily degraded inputs land closer to the right one.
- Capacity wall is clear: as load approaches 1.0+, accuracy drops. At load 2.0, 80% masking gives ~61% accuracy.
- Sparser patterns (5%) are more robust than denser ones (20%) — more distinguishable in high-dimensional space.
- Retention metric (masked_sim / full_sim) is important: if full-input correlation is 0.8 and masked gives 0.6, the drop is from 0.8 to 0.6, not from 1.0 to 0.6.

### Scripts

- `experiments/2026_03_29/pattern_completion/pattern_completion.py` — full sweep, produces composite figures and capacity heatmaps.

---

## Ensemble (Multiple Zenith Nodes)

### Motivation

A single node with k=10 can't distinguish 40 patterns (load=4.0). But if multiple independent nodes see the same input, each with different random templates, the *combination* of their outputs forms a distributed code. Pattern A and B might share a winner in node 1, but have different winners in node 3 — the joint code is unique even when individual nodes collide.

### Architecture

- M independent Zenith nodes, each with k templates, all receiving the same input.
- Output representation: concatenated activation vectors from all nodes (M*k dimensional).
- Retrieval: cosine similarity between masked-input activation and all stored full-input activations. Nearest neighbor = retrieved pattern.

### Results

**6 nodes, k=10, N=40 (load=4.0), sparsity=0.10:**
- 100% retrieval up to 20% masking.
- 99.75% at 40%, 95.65% at 60%, 83.70% at 80%.
- Per-node cosine sim is basically zero (~-0.04). Individual nodes learn nothing useful at this load. But the 60-dim ensemble activation is discriminative.

**50 nodes, k=10, N=40 (load=4.0), sparsity=0.05:**
- 100% retrieval up to 40% masking.
- 99.90% at 60%, 95.25% at 80%.
- 500-dim activation space gives much more room for pattern separation.

**50 nodes, k=10, N=200 (load=20.0), sparsity=0.05:**
- 100% code uniqueness — all 200 patterns get distinct winner-index codes.
- 99.6% retrieval at 40% masking. Collapses at 80% (42%).
- Shows the system can store 20x more patterns than templates per node, but heavy masking overwhelms it.

### Exact code matching vs cosine similarity

First version used exact winner-index matching: all M nodes must pick the same winner as with full input. This is too strict — accuracy drops as M increases (0.83^6 = 0.33). Switched to cosine similarity on concatenated activations, which is the right metric because it tolerates a few nodes flipping while the overall code stays close.

### What this is and isn't

**It is:** A high-capacity content-addressable memory / pattern lookup system. Given a partial input, it identifies which stored pattern it came from via activation-space nearest-neighbor.

**It is not:** A generalizer. Similar inputs produce similar outputs, but this is trivial — correlation is a continuous similarity measure, so this is just the math being smooth, not the system learning abstractions. Flipping a few bits and getting similar output doesn't demonstrate generalization, it demonstrates continuity.

**Open question:** Can we design an experiment that tests genuine generalization (learning structure beyond what raw input similarity gives)? The naive "train on prototypes, test on variants" approach doesn't work because correlation already handles that for free. Need a task where the system must learn a non-trivial mapping that raw similarity doesn't capture.

### Scripts

- `experiments/2026_03_29/ensemble_completion/ensemble_completion.py` — single-config experiment with report generation.
- Reports saved as `results_ensemble_completion/report.md`, `report2.md`, `report3.md`.
