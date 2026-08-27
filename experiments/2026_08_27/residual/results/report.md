# Square, same-rate chassis + completion read

K1=64
TF plain read (chassis check, n=100): self 1.00 / fwd 0.00 / back 0.00
TF sweep  lam   mu  | forward  self  backward |  mean  | lead ahead
         1.0  0.0 |   0.29   0.25    0.46   | -1.17 |   0.16
         1.0  0.5 |   0.40   0.15    0.45   | -0.82 |   0.32
         1.0  1.0 |   0.46   0.15    0.39   | +0.59 |   0.32
         1.0  2.0 |   0.41   0.16    0.43   | +0.66 |   0.32
         2.0  0.0 |   0.22   0.74    0.04   | +1.41 |   0.23
         2.0  0.5 |   0.18   0.78    0.04   | +0.96 |   0.22
         2.0  1.0 |   0.19   0.81    0.00   | +1.50 |   0.22
         2.0  2.0 |   0.17   0.83    0.00   | +1.47 |   0.22
best (fwd-back): lam=2.0, mu=1.0
FREE-RUN 100 emissions: mean |x err| 4.98 (max 15)  (refs: arrow forms 0.00, trail-only 8.12, completion-plain 10.02, refractory 7.49)
free-run next-unit offsets: forward 0.38 / self 0.03 / backward 0.59
first 33 (gen_x, true_x): (19,20) (19,19) (19,18) (15,17) (19,16) (19,15) (15,14) (15,13) (15,12) (15,11) (15,10) (15,9) (15,8) (15,7) (15,6) (15,5) (15,4) (12,3) (15,2) (15,1) (15,0) (11,1) (11,2) (11,3) (11,4) (11,5) (11,6) (11,7) (19,8) (19,9) (19,10) (19,11) (19,12)
L2 arcs (row: home phases, final epoch):
    L2 row 0: 6 phases [21, 22, 23, 24, 25, 26]
    L2 row 1: 4 phases [20, 27, 28, 29]
    L2 row 2: 4 phases [18, 19, 30, 31]
    L2 row 4: 2 phases [32, 33]
    L2 row 5: 5 phases [13, 14, 15, 16, 17]
    L2 row 6: 4 phases [34, 35, 36, 37]
    L2 row 7: 16 phases [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 38, 39]
