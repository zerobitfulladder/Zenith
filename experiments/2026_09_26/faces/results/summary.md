# Results

Forward stack: 0.8378 balanced accuracy on real test faces (40 attributes, mean).

## Reconstruction from the top map alone

- pixel error 0.0190
- the stack's own head on the renders: 0.6585 balanced accuracy against the true attributes (real faces 0.8378)
- mouth region pixel error: whole render 0.0161, mouth window attended and rendered 0.0101

## Concept renders: the head's probability of the attribute on the render

| attribute | render of faces with it | render of faces without it |
|---|---|---|
| Male | 0.86 | 0.00 |
| Smiling | 0.93 | 0.07 |
| Eyeglasses | 0.81 | 0.00 |
| Mustache | 0.42 | 0.00 |
| Blond_Hair | 0.94 | 0.46 |
| Wearing_Hat | 0.03 | 0.00 |
| Bald | 0.69 | 0.00 |
| Young | 0.56 | 0.26 |
| Wearing_Lipstick | 0.39 | 0.01 |

## Woman + mustache

| render | P(Male) | P(Mustache) |
|---|---|---|
| woman | 0.00 | 0.00 |
| man + mustache | 0.99 | 0.42 |
| arith | 0.00 | 0.00 |
| arith x3 | 0.73 | 0.74 |
| grid | 0.00 | 0.00 |
| window | 0.06 | 0.00 |
| window x3 | 0.84 | 0.94 |
| replace | 0.62 | 0.06 |
| real women | 0.06 | 0.01 |
| real men w/ mustache | 0.99 | 0.84 |

## Finding the mouth from the face's own map (300 test women, shifted)

Cosine of a mouth signature (mean feature vector at the mouth cells of aligned training faces) against every top-map cell; the best cell is the mouth.
row hit = the found cell is in the row holding the mouth line; cell hit = and within one column.  found rows = how many of the 300 landed in each of the five rows.

| shift | mouth row | row hit | cell hit | found rows |
|---|---|---|---|---|
| dy=+0 dx=+0 | 3 | 0.96 | 0.96 | [2, 3, 0, 289, 6] |
| dy=+0 dx=-20 | 3 | 0.95 | 0.95 | [0, 4, 1, 286, 9] |
| dy=+0 dx=+20 | 3 | 0.97 | 0.96 | [0, 2, 0, 291, 7] |
| dy=-24 dx=+0 | 4 | 0.19 | 0.17 | [1, 11, 48, 183, 57] |
| dy=+24 dx=+0 | 3 | 0.01 | 0.01 | [0, 3, 275, 4, 18] |
