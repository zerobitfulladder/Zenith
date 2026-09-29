# Results

Full MNIST: 50,000 to learn, 2,000 of the next 10,000 to choose the pixel noise on a
10,000-image pilot, the 10,000 test images for every number here. One seed. Two standard
errors on a test accuracy of 0.96 over 10,000 images is 0.004; over the 1,000 images of the
hidden-part tests it is 0.012.

## Pilot: the pixel noise SIG_PIX

Learn on 10,000, read 2,000 of the choose set. 0.2 was chosen by `run.py`; 0.1 and 0.15 were
run afterwards by `controls.py`, because 0.2 was the edge of the first grid.

| SIG_PIX | choose set | causes |
|---|---|---|
| 0.1 | 0.9360 | 960 |
| 0.15 | 0.9445 | 780 |
| 0.2 (used) | 0.9390 | 769 |
| 0.3 | 0.9150 | 992 |
| 0.5 | 0.7805 | 2,421 |

0.15 and 0.2 are within the noise of 2,000 images (two standard errors about 0.011).

## The table after 50,000 images

| | |
|---|---|
| causes | 2,628 (2,645 hired, 17 retired) |
| leaves per cause | median 7, largest 17 |
| deepest level reached | 0: 17, 1: 61, 2: 211, 3: 1,115, 4: 1,095, 5: 124, 6 (the cap): 5 |
| leaves grown after hiring | 93 |
| numbers stored | 424,968 |
| learning time | 342 s, 6.8 ms per image |
| right before learning each image, last 5,000 | 0.952 |

## Naming the digit, label hidden

| reader | test | numbers stored |
|---|---|---|
| cheapest cause | 0.9593 | 424,968 |
| blend over the 16 shortlisted causes | 0.9593 | |
| fast path alone: nearest typical drawing | 0.8774 | |
| same table, pixel term only | 0.6407 | |
| same table, tree surprise only | 0.8164 | |
| pixels, the same rule (Hart's condensed nearest neighbour): 4,362 images kept | 0.9322 | 3,419,808 |
| pixels, 2,645 random training images | 0.9196 | 2,073,680 |
| pixels, all 50,000 training images | 0.9666 | 39,200,000 |
| counted table, 2026-09-27 | 0.9725 | 144,000 |
| CNN, 2026-09-27 | 0.9908 | 118,346 |

## Hidden parts: 1,000 test images, label hidden too

Label: right answers. Fill error: mean squared error over the hidden pixels.

| bottom half hidden | label | fill error |
|---|---|---|
| cheapest cause, fitted to visible + imagined ink | 0.841 | 0.0659 |
| blend over the shortlist | 0.845 | |
| the same images whole, cheapest cause | 0.956 | |
| nearest of all 50,000 training images, on the visible pixels | 0.915 | 0.0668 |
| nearest of 2,645 random training images | 0.856 | 0.0754 |
| nearest of Hart's 4,362 | 0.789 | 0.0841 |
| the average training image | | 0.0703 |
| blank | | 0.1204 |

| 12x12 box over ink hidden | label | fill error |
|---|---|---|
| cheapest cause, fitted to visible + imagined ink | 0.862 | 0.1141 |
| blend over the shortlist | 0.863 | |
| the same images whole, cheapest cause | 0.956 | |
| nearest of all 50,000 training images, on the visible pixels | 0.931 | 0.1088 |
| nearest of 2,645 random training images | 0.864 | 0.1287 |
| nearest of Hart's 4,362 | 0.798 | 0.1485 |
| the average training image | | 0.1374 |
| blank | | 0.2511 |
