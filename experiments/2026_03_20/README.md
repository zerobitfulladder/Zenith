# 2026-03-20

## `feedback/` — top-down gain feedback in a two-layer SRL stack

Two stacks of two signed-residual layers, one with L2 feeding a gain back onto
L1 and one without, trained in AxonForge and read out with a logistic
regression on L2. Five feedback settings; in all five the stack without
feedback read MNIST better (0.8794-0.8866 against 0.8607-0.8773).

Full writeup: [`feedback/README.md`](feedback/README.md).
