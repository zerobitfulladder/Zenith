# F3: inference searchlight (no training; bottom-20% margins, top-3 candidates)

## rig: all-dense
- baseline hard 0.9080; low-margin subset (1000 queries) baseline 0.7940
- w=0.3: overall 0.9078, low-margin 0.7930
- w=0.5: overall 0.9070, low-margin 0.7890
- w=1.0: overall 0.9054, low-margin 0.7810
- w=pixel-only: overall 0.9042, low-margin 0.7750

## rig: all-top1
- baseline hard 0.7092; low-margin subset (1000 queries) baseline 0.4670
- w=0.3: overall 0.7274, low-margin 0.5580
- w=0.5: overall 0.7276, low-margin 0.5590
- w=1.0: overall 0.7276, low-margin 0.5590
- w=pixel-only: overall 0.7262, low-margin 0.5520

