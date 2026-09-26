# Results

what = grid.  Judge on real test digits: 0.9879.  Forward net on real test digits: 0.9856.

Fractions named correctly by the judge over the 10,000 test digits (concept: over 10 class renders).
recon is given as rendered / brightness-normalised; the controls and concept use the normalised render.
swap: the digit's own where with the what of the next class; 'content' = judge said the next class, 'where' = judge said the digit's own class.

own = the forward stack's own head on the render (recon / concept).  code cos = cosine between the what rendered and the what
recomputed from the render, recon (floor: cosine to a shuffled digit's what) / concept.

| where code | unpool | recon | recon mse | where alone | swap: content / where | concept | own | code cos |
|---|---|---|---|---|---|---|---|---|
| energy | product | 0.957 / 0.958 | 0.0182 | 0.953 | 0.011 / 0.886 | 1.0 / 1.0 | 0.945 / 1.0 | 0.95 (0.65) / 0.88 |
| energy | learned | 0.969 / 0.971 | 0.0190 | 0.924 | 0.143 / 0.723 | 1.0 / 1.0 | 0.965 / 1.0 | 0.97 (0.65) / 0.89 |
| energy3 | product | 0.839 / 0.840 | 0.0371 | 0.357 | 0.626 / 0.111 | 1.0 / 1.0 | 0.797 / 1.0 | 0.84 (0.60) / 0.80 |
| energy3 | learned | 0.901 / 0.914 | 0.0326 | 0.140 | 0.925 / 0.018 | 1.0 / 1.0 | 0.905 / 1.0 | 0.92 (0.62) / 0.85 |
| switch1 | product | 0.923 / 0.923 | 0.0203 | 0.566 | 0.248 / 0.358 | 1.0 / 0.9 | 0.942 / 1.0 | 0.96 (0.67) / 0.92 |
| switch1 | learned | 0.914 / 0.923 | 0.0316 | 0.119 | 0.999 / 0.000 | 1.0 / 1.0 | 0.915 / 1.0 | 0.93 (0.62) / 0.87 |
| box | product | 0.764 / 0.764 | 0.0404 | 0.098 | 0.933 / 0.000 | 1.0 / 1.0 | 0.754 / 0.9 | 0.83 (0.64) / 0.82 |
| box | learned | 0.889 / 0.909 | 0.0393 | 0.101 | 1.000 / 0.000 | 1.0 / 1.0 | 0.892 / 1.0 | 0.91 (0.61) / 0.84 |
| none | product | 0.341 / 0.389 | 0.0719 | 0.089 | 0.507 / 0.097 | 0.5 / 0.5 | 0.469 / 0.3 | 0.69 (0.54) / 0.64 |
| none | learned | 0.924 / 0.928 | 0.0234 | 0.101 | 1.000 / 0.000 | 1.0 / 1.0 | 0.922 / 1.0 | 0.93 (0.63) / 0.90 |
