# Results

Unique: a Gaussian over the real top codes.  The 64-component version keeps 0.52 of the variance.
Mean pixel distance between rendered faces (diversity): real renders 0.259, diag 0.079, pca64 0.168

Rebuild of 1000 test faces from nine windows.  Pixel error to the real face; sharpness = mean absolute Laplacian.

| way | pixel error | sharpness |
|---|---|---|
| real | 0.0000 | 0.0607 |
| blur | 0.0192 | 0.0126 |
| from_blur | 0.0203 | 0.0119 |
| mean | 0.0205 | 0.0114 |
| sample | 0.0228 | 0.0114 |
| nearest | 0.0328 | 0.0162 |
| from_real | 0.0088 | 0.0173 |

Imagined faces (pca64 samples): sharpness coarse 0.0091, from_blur 0.0077, sample 0.0083, nearest 0.0119, two samples differ by 0.0370.

## Two fixes: feathered blending, and 40x32 windows on a 7x7 grid

| windows | source | pixel error | sharpness |
|---|---|---|---|
| 80x64 hard | from_blur | 0.0203 | 0.0119 |
| 80x64 hard | sample | 0.0227 | 0.0114 |
| 80x64 hard | from_real | 0.0088 | 0.0173 |
| 80x64 feathered | from_blur | 0.0204 | 0.0077 |
| 80x64 feathered | sample | 0.0228 | 0.0076 |
| 80x64 feathered | from_real | 0.0090 | 0.0130 |
| 40x32 hard | from_blur | 0.0193 | 0.0128 |
| 40x32 hard | sample | 0.0220 | 0.0155 |
| 40x32 hard | from_real | 0.0038 | 0.0237 |
| 40x32 feathered | from_blur | 0.0192 | 0.0078 |
| 40x32 feathered | sample | 0.0231 | 0.0095 |
| 40x32 feathered | from_real | 0.0038 | 0.0190 |

Imagined faces, fixed: sharpness coarse 0.0091, 80x64 feathered, from_blur 0.0047, 80x64 feathered, sample 0.0056, 40x32 feathered, from_blur 0.0051, 40x32 feathered, sample 0.0076.
