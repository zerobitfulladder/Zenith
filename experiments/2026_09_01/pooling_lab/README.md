# Pooling lab — an interactive kernel and pooling playground

Not an experiment and no results: a small window for building intuition about
what pooling does to a filtered picture.

Three panels:

- **left** — the source picture, `cat.jpg`, in greyscale
- **middle** — the picture after one kernel is swiped over it (a convolution)
- **right** — that result after pooling

Controls: the kernel type (lines, a cross, blurs, edges, laplacian, sharpen,
emboss, identity) and its size (1 to 31); whether the filtered output is kept
signed, rectified (`relu`) or made absolute (`abs`); the pooling mode (max, avg,
min), window (1 to 32) and stride, with an option for the stride to follow the
window. Everything recomputes as you move a control.

Run (needs Tk, Pillow and scipy):

    .venv/bin/python experiments/2026_09_01/pooling_lab/pooling_lab.py
