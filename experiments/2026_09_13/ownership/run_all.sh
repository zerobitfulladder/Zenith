#!/bin/bash
# three rules at price 0.02 x two seeds, the hard rule at 0.01 and 0.04, all in parallel;
# then the completeness check on the hard 0.02 seed-0 weights, then the board.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for seed in 0 1; do
  for rule in order hard soft; do
    uv run python run.py $rule 0.02 $seed > results/log_${rule}_l0.02_s${seed}.txt 2>&1 &
  done
done
uv run python run.py hard 0.01 0 > results/log_hard_l0.01_s0.txt 2>&1 &
uv run python run.py hard 0.04 0 > results/log_hard_l0.04_s0.txt 2>&1 &
wait
grep -h "^hard\|^order\|^soft" results/log_*.txt
uv run python astar.py hard 0.02 25 400 48 2>&1 | tail -3
uv run python board.py
