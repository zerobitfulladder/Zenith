# Anti-Hebbian lateral inhibition with geodesic learning

2026-08-31, last run of the day. Three local rules on MNIST patches, nothing
global:

    settle      y  <-  relu( W x  -  M y  -  t )        iterated
    feedforward each active template rotates toward x along the sphere, at a
                rate proportional to its own activity
    lateral     dM_ij = alpha ( <y_i y_j> - p^2 ), clipped at >= 0 -- units
                that fire together start suppressing each other
    threshold   dt_i  = beta ( <y_i> - p ) -- each unit driven to its target
                activity p

Competition here is not winner-take-all: several templates can be active at
once, and what is punished is redundant co-activation. Patches are mean-centred
and L2-normalised, so `W x` is a Pearson correlation.

No writeup was kept. Runs over template counts K = 8, 16, 25, 64 and, at K=64,
inhibition rates alpha = 0, 0.005, 0.02, 0.1, 0.4; each writes per-epoch active
fraction, rebuild cosine, coherence and dead count to `results/ah_*.json`, the
state to `results/ah_*.npz`, and the templates to `results/templates_*.png`
(`templates_compare.png`, `lateral.png` side by side).

Run: `AH_ALPHA=0.4 .venv/bin/python experiments/2026_08_31/antihebb/ah.py 64`
(first argument K, environment variable `AH_ALPHA` the inhibition rate).
