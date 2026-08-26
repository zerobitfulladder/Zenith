# Compositional generalization: 4 lines x 4 direction bits

Trained on <=2-bit combos (11 of 16); held out: 1110, 1111.

| test | arm | decoded | correct |
|---|---|---|---|
| 1110 | G | 1100 | 3/4 |
| 1110 | Gq | 1110 | 4/4 |
| 1110 | H | 0x00 | 1/4 |
| 1110 | A | 1110 | 4/4 |
| 1110 | J | 1110 | 4/4 |
| 1110 | L | 1110 | 4/4 |
| 1111 | G | 0011 | 2/4 |
| 1111 | Gq | 1111 | 4/4 |
| 1111 | H | 0100 | 1/4 |
| 1111 | A | 1111 | 4/4 |
| 1111 | J | 0100 | 1/4 |
| 1111 | L | 1111 | 4/4 |

Files: input_1100.gif, gen_{G,L}_*.gif, filmstrip.png
