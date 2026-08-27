# Square, completion arrow (no next, no lag channels)

free-run 100 emissions over 300 ticks: mean |x err| vs t 10.02 (max 20), vs t+1 10.08  (refs: arrow forms 0.00, trail-only 8.12)
L1 retrieval offsets (n=100): forward 0.44 / self 0.02 / backward 0.54, mean -1.32, median -2.0
L2 retrieval offsets (n=100): forward 0.50 / self 0.05 / backward 0.45, mean +0.05, median +0.5
L3 retrieval offsets (n=34): forward 0.53 / self 0.12 / backward 0.35, mean +0.79, median +1.0

TEACHER-FORCED (true frames driven, on-distribution):
  L1 offsets (n=100): forward 0.00 / self 1.00 / backward 0.00, mean +0.00
  L2 offsets (n=97): forward 0.13 / self 0.82 / backward 0.04, mean +0.43
  L3 offsets (n=34): forward 0.00 / self 0.97 / backward 0.03, mean -0.26
  emission lead vs present (signed, + = ahead in motion dir): mean -0.01, ahead 0.02 / at-present 0.95 / behind 0.03

first 33 free-run (gen_x, true_x): (19,18) (19,15) (20,12) (20,9) (20,6) (20,3) (20,0) (20,3) (20,6) (20,9) (20,12) (20,15) (20,18) (20,19) (20,16) (20,13) (20,10) (20,7) (20,4) (20,1) (20,2) (20,5) (20,8) (20,11) (20,14) (20,17) (20,20) (20,17) (20,14) (20,11) (20,8) (20,5) (20,2)
