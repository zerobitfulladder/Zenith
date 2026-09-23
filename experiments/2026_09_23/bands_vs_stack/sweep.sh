#!/bin/bash
# The short run: the three models and the pixel floor, aligned faces, 8 epochs each (~8 min).
# sweep_full.sh has the jittered condition, each band alone and the max-pool CNN.
cd "$(dirname "$0")"
run() { uv run python run.py "$@" 2>&1 | grep --line-buffered -v Warning; }
for arm in pixels bands1 stack bands3; do run --arm $arm --cond aligned --epochs 8; done
echo SWEEP DONE
