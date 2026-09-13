#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for init in data order hard soft; do
  uv run python em.py $init 6 4000 > results/log_em_${init}.txt 2>&1 &
done
for rule in order soft; do
  uv run python selfprice.py $rule 0 > results/log_selfprice_${rule}.txt 2>&1 &
done
wait
grep -h "test:\|^selfprice" results/log_em_*.txt results/log_selfprice_*.txt
uv run python em_board.py
