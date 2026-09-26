# Results

what = global.  Judge on real test digits: 0.9881.  Forward net on real test digits: 0.9860.

Fractions named correctly by the judge over the 10,000 test digits (concept: over 10 class renders).
recon is given as rendered / brightness-normalised; the controls and concept use the normalised render.
swap: the digit's own where with the what of the next class; 'content' = judge said the next class, 'where' = judge said the digit's own class.

own = the forward stack's own head on the render (recon / concept).  code cos = cosine between the what rendered and the what
recomputed from the render, recon (floor: cosine to a shuffled digit's what) / concept.

| where code | unpool | recon | recon mse | where alone | swap: content / where | concept | own | code cos |
|---|---|---|---|---|---|---|---|---|
| energy | product | 0.943 / 0.949 | 0.0357 | 0.942 | 0.002 / 0.932 | 1.0 / 1.0 | 0.918 / 1.0 | 0.97 (0.91) / 0.94 |
| energy | learned | 0.934 / 0.938 | 0.0308 | 0.934 | 0.003 / 0.931 | 1.0 / 0.9 | 0.918 / 0.9 | 0.98 (0.91) / 0.95 |
| energy3 | product | 0.473 / 0.477 | 0.0667 | 0.439 | 0.069 / 0.415 | 0.5 / 0.4 | 0.345 / 0.1 | 0.89 (0.85) / 0.82 |
| energy3 | learned | 0.396 / 0.451 | 0.0645 | 0.438 | 0.061 / 0.431 | 0.4 / 0.5 | 0.351 / 0.3 | 0.90 (0.86) / 0.86 |
| switch1 | product | 0.357 / 0.376 | 0.0575 | 0.301 | 0.086 / 0.285 | 0.6 / 0.6 | 0.501 / 0.5 | 0.94 (0.91) / 0.90 |
| switch1 | learned | 0.162 / 0.167 | 0.0794 | 0.183 | 0.063 / 0.203 | 0.0 / 0.2 | 0.134 / 0.1 | 0.72 (0.71) / 0.70 |
| box | product | 0.117 / 0.117 | 0.0631 | 0.097 | 0.116 / 0.085 | 0.1 / 0.1 | 0.209 / 0.2 | 0.88 (0.87) / 0.89 |
| box | learned | 0.053 / 0.107 | 0.0779 | 0.097 | 0.103 / 0.098 | 0.1 / 0.1 | 0.115 / 0.1 | 0.68 (0.68) / 0.68 |
| none | product | 0.103 / 0.199 | 0.0873 | 0.101 | 0.195 / 0.101 | 0.1 / 0.2 | 0.114 / 0.1 | 0.63 (0.63) / 0.63 |
| none | learned | 0.051 / 0.073 | 0.0751 | 0.114 | 0.114 / 0.098 | 0.0 / 0.1 | 0.136 / 0.1 | 0.72 (0.71) / 0.72 |
