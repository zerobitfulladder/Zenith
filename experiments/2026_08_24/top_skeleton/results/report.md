# Twin-top: dense vs skeleton archive learning (cupy, stride-1 flagship, train 13.8s)

| arm | hard (dense query) | consistent | mean cos | mean abs cos | code cos | within-class | top peak |
|---|---|---|---|---|---|---|---|
| dense | 0.8932 | 10/10 | +0.189 | 0.197 | +0.220 | +0.379 | 0.109 |
| skel | 0.7462 | 10/10 | +0.046 | 0.049 | +0.031 | +0.077 | 0.899 |

Skeleton arm, skeleton query (secondary): hard 0.6816

Reference (saved s1 flagship, dense top): mean cos +0.115, hard .849 (f64 CPU eval; this run's dense arm is the like-for-like f32 control).
Figures: generation_{dense,skel}.png, gallery_{dense,skel}.png
