# Cart-pole without imitation: expectation-gated rotation

K=256, 600 self-played episodes, theta = min(0.8*|delta|+0.01, 0.3).
Final greedy: mean 97, median 89 / 500 (random floor 23; imitation 491).
Stress: 108/400 (imitation pure 165, shoved 273).

Figure: learning_curve.png
