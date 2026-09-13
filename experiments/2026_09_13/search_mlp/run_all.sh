#!/bin/bash
# all arms and seeds in parallel, two BLAS threads each, then the board
cd "$(dirname "$0")"
for seed in 0 1; do
  for arm in backprop search-b1 search-b4 label-b4; do
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run python run.py $arm $seed &
  done
done
wait
uv run python board.py
