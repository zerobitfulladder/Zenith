#!/bin/sh
# the sweep behind README.md.  15 epochs, lr 1e-4, same for every arm.
# (lr 1e-3 diverged in every arm around epoch 5-12: results/diag/)
cd "$(dirname "$0")/../../.."
P="uv run python experiments/2026_09_21/catmap/catmap.py"
{
for sp in pi tau; do
  echo "--logreg --span $sp"
  for L in 2 4 8 16; do
    echo "--depth $L --span $sp"
    echo "--depth $L --span $sp --nowrap"
    echo "--depth $L --span $sp --shift mlp"
  done
  echo "--depth 8 --span $sp --mix shuffle"
done
echo "--depth 8 --span pi --init 1"
for s in 1 2; do
  echo "--depth 8 --span pi --seed $s"
  echo "--depth 8 --span pi --seed $s --nowrap"
done
} | xargs -P 4 -I{} sh -c "$P {} 2>&1 | tail -1"
$P --plot

# --- added later: the periodic shear, sin/cos of the untouched half ---
{
for sp in pi tau; do for L in 2 4 8 16; do echo "--depth $L --span $sp --shift trig"; done; done
for s in 1 2; do echo "--depth 8 --span pi --seed $s --shift trig"; done
} | xargs -P 4 -I{} sh -c "$P {} 2>&1 | tail -1"
$P --plot
