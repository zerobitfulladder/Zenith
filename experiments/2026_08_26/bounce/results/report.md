# Bouncing generation (iterative resolution), base weights

One-shot baseline = the base rig's standard hardened generation.

| arm | self-classification | peakiness per bounce |
|---|---|---|
| self | 1/10 | 0.016 -> 0.007 -> 0.007 -> 0.008 -> 0.009 |
| consult | 1/10 | 0.016 -> 0.008 -> 0.009 -> 0.016 -> 0.016 |

Figures: bounce_self.png, bounce_consult.png (rows = bounces, k = dense,32,8,2,1).

## Head-to-head vs one-shot (pixel-LR judge, ink-normalized)

- one-shot k=1: 7/10, mean p(intended) 0.661
- bounce k=1: 7/10, mean p(intended) 0.661

Figure: oneshot_vs_bounce.png
