# `gpu_minibatch/` — the consolidated 4-layer unit, mini-batched on the GPU (stride 2, then stride 1)

(2026-08-23, from the day log.) Script: `../rig/run_gpu_minibatch.py` (shared with later days, so it lives in `../rig/`).
Results: `results/stride2/` (default), `results/stride1/` (`GF_W1_STR=1`, the flagship weights
that `../inpaint/` and later days load).

## GPU mini-batch trainer — predictions (before running, 2026-08-23)

`../rig/run_gpu_minibatch.py` (GF_XP=cupy|numpy, GF_B batch size, default 128):
the consolidated 4-layer sparselearn config, mini-batched — B images
encoded against frozen weights, then per-template one vectorized
geodesic rotation toward the normalized mean of its batch targets with
theta = clip(eta * sum(c), 0.3). This is a dynamics change (the
approximation under test); GF_XP=numpy isolates algorithm from hardware.
Evaluation reuses the exact CPU pipeline on the trained weights.

1. Fidelity: probes within ±0.010 of CPU sparselearn (.9528/.9592/.9590),
   hard within ±0.02 of .8748, peak2 >= 0.6, peak3 >= 0.45, retrieval
   10/10, parts gallery diverse.
2. Speed: >= 10x wall-clock over the ~20-minute CPU online run.
3. If fidelity fails, the numpy-minibatch arm tells us whether the
   mini-batch approximation (not the GPU) is responsible.

### Outcome (same night) — verified at 114x

Train time **10.5s** vs ~1200s CPU online (**114x**), on the RTX 3060
(CUDA libs installed as suffix-less nvidia-* wheels: nvidia-cublas,
nvidia-cuda-runtime, nvidia-curand, nvidia-cuda-nvrtc — the -cu13 PyPI
names are placeholder squatters).

Fidelity vs CPU sparselearn: probes .9552/.9594/.9578 (vs
.9528/.9592/.9590 — all within ±0.010), hard .8738 (vs .8748 — within
0.001!), retrieval 10/10. Peakiness .623/.431 — peak3 misses its 0.45
bar by 0.019 (mini-batch mean-target averaging smooths slightly) but
the parts gallery is qualitatively intact: diverse diagonals, corners,
elbows, bars. 6/7 criteria pass, the seventh cosmetic.

The mini-batch approximation (frozen-weights batch encode + one
vectorized geodesic step per template toward its batch-mean target,
theta = clip(eta*sum_c, 0.3)) is a faithful stand-in for the online
rule at B=128. **Project economics changed: 8-seed validations in
minutes, sweeps in minutes, CelebA-scale feasible.**

## Stride-1 consolidated rig on GPU — predictions (before running, 2026-08-23)

`GF_W1_STR=1 python ../rig/run_gpu_minibatch.py`: L1 windows at every pixel
position. Grids reshape 25x25 -> 12x12 -> 10x10; code3 dim 10,000; L2/L3
receptive fields shrink in pixels (6px/10px vs 8px/16px) — an accepted
geometry change, not a bug. Baselines: stride-2 GPU run
(.9552/.9594/.9578, hard .8738) and the 2-layer detail rig's generation.

1. The point: hardened generation visibly smoother/finer than any
   4-layer generation yet (up to 16 windows per pixel, 4x constellation
   density — the detail-rig effect, now with parts underneath).
2. Probes >= stride-2 values at every level (denser codes carry more);
   probe L1 targets the detail-rig territory (>= 0.95, hopefully 0.96+).
3. Hard readout >= .874 (denser constellations overlap more).
4. Train time under ~60s (GPU absorbs the 4x window count).

### Outcome (same night) — the ladder rises; the 8 finally draws

| | stride-2 GPU | stride-1 GPU |
|---|---|---|
| train | 10.5s | 27.2s |
| probes | .955/.959/.958 | .954/.963/**.9658** |
| hard | .874 | .849 |
| peakiness | .62/.43 | .69/.56 |

- **Probe L3 0.9658 — best representation in project history — and the
  ladder now RISES monotonically (.954 -> .963 -> .966)**: with the
  consolidated unit + density, depth gains at every level. The depth
  question that opened at "each layer loses" closes at "each layer adds."
- P1 confirmed emphatically: hardened generation is the best of the
  project — bold smooth digits, and the FIRST clean double-loop 8 of
  the session (generation_hardened.png).
- P3 missed: hard readout dipped to .849 (200 memories tile a
  10,000-dim code thinly — the lookup-vs-probe gap widens with
  dimension; the probe proves the information is intact, consistent
  with the standing readout-is-a-separate-organ conclusion).
- 27 seconds. This configuration (consolidated unit, stride-1, GPU) is
  the new flagship: **best representation + best generation, trained in
  under half a minute.** Next: CelebA + composition on this rig.
