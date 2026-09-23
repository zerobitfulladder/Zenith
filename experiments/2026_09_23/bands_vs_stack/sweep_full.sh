#!/bin/bash
# The full run, one process at a time (the GPU holds the dataset once per process).
cd "$(dirname "$0")"
run() { uv run python run.py "$@" 2>&1 | grep -v Warning; }
for cond in aligned jitter; do
  for arm in pixels bands1 stack bands3; do run --arm $arm --cond $cond; done
done
for L in 1 2 3 4 5; do run --arm bands1 --cond aligned --only_level $L; done
for cond in aligned jitter; do run --arm stack --cond $cond --pool max; done
echo SWEEP DONE
