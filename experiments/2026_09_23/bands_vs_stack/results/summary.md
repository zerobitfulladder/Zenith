# Results

Balanced accuracy on the 5,000 held-out images, mean over the 40 attributes, at the epoch
with the best validation score. `acc` is plain accuracy.

| arm | params | aligned | acc | jitter | acc |
|---|---|---|---|---|---|
| pixels, linear | 51,240 | 0.7764 | 0.7699 | — | — |
| bands, 1 conv each | 129,640 | 0.7721 | 0.7644 | — | — |
| bands, 3 convs each | 222,120 | 0.8192 | 0.8155 | — | — |
| stack (CNN, avg pool) | 165,352 | 0.8288 | 0.8234 | — | — |

## Per attribute, aligned

| attribute | pixels | bands 1 conv | bands 3 convs | stack | stack - bands1 |
|---|---|---|---|---|---|
| Eyeglasses | 0.883 | 0.855 | 0.939 | 0.970 | +0.115 |
| Blurry | 0.652 | 0.704 | 0.796 | 0.819 | +0.115 |
| Sideburns | 0.865 | 0.828 | 0.913 | 0.929 | +0.102 |
| No Beard | 0.842 | 0.811 | 0.877 | 0.904 | +0.093 |
| Goatee | 0.884 | 0.824 | 0.891 | 0.916 | +0.092 |
| Mustache | 0.877 | 0.812 | 0.878 | 0.902 | +0.090 |
| Wearing Earrings | 0.728 | 0.705 | 0.777 | 0.791 | +0.086 |
| Male | 0.901 | 0.858 | 0.938 | 0.941 | +0.082 |
| 5 o Clock Shadow | 0.816 | 0.782 | 0.840 | 0.856 | +0.074 |
| Bushy Eyebrows | 0.764 | 0.743 | 0.824 | 0.814 | +0.071 |
| Smiling | 0.884 | 0.827 | 0.874 | 0.896 | +0.070 |
| Young | 0.722 | 0.729 | 0.783 | 0.797 | +0.068 |
| Bags Under Eyes | 0.726 | 0.692 | 0.742 | 0.760 | +0.068 |
| Narrow Eyes | 0.695 | 0.633 | 0.686 | 0.699 | +0.066 |
| Brown Hair | 0.629 | 0.695 | 0.752 | 0.761 | +0.066 |
| Mouth Slightly Open | 0.810 | 0.777 | 0.818 | 0.842 | +0.065 |
| Wearing Lipstick | 0.868 | 0.850 | 0.904 | 0.910 | +0.059 |
| Double Chin | 0.822 | 0.792 | 0.853 | 0.851 | +0.059 |
| Arched Eyebrows | 0.748 | 0.729 | 0.780 | 0.786 | +0.057 |
| High Cheekbones | 0.824 | 0.790 | 0.838 | 0.845 | +0.055 |
| Receding Hairline | 0.766 | 0.801 | 0.834 | 0.855 | +0.054 |
| Bangs | 0.858 | 0.874 | 0.918 | 0.928 | +0.054 |
| Heavy Makeup | 0.842 | 0.835 | 0.881 | 0.887 | +0.052 |
| Wearing Necktie | 0.787 | 0.783 | 0.826 | 0.827 | +0.044 |
| Attractive | 0.751 | 0.768 | 0.801 | 0.811 | +0.043 |
| Bald | 0.874 | 0.911 | 0.952 | 0.954 | +0.043 |
| Rosy Cheeks | 0.817 | 0.814 | 0.849 | 0.855 | +0.041 |
| Big Nose | 0.710 | 0.711 | 0.751 | 0.752 | +0.041 |
| Wearing Hat | 0.832 | 0.895 | 0.931 | 0.935 | +0.040 |
| Chubby | 0.804 | 0.784 | 0.819 | 0.823 | +0.039 |
| Big Lips | 0.625 | 0.630 | 0.651 | 0.665 | +0.036 |
| Blond Hair | 0.851 | 0.882 | 0.914 | 0.915 | +0.034 |
| Gray Hair | 0.860 | 0.866 | 0.891 | 0.899 | +0.032 |
| Pale Skin | 0.772 | 0.779 | 0.808 | 0.808 | +0.029 |
| Wearing Necklace | 0.656 | 0.675 | 0.695 | 0.703 | +0.028 |
| Oval Face | 0.637 | 0.635 | 0.660 | 0.661 | +0.026 |
| Straight Hair | 0.588 | 0.628 | 0.664 | 0.652 | +0.024 |
| Black Hair | 0.747 | 0.797 | 0.814 | 0.818 | +0.022 |
| Wavy Hair | 0.677 | 0.717 | 0.739 | 0.738 | +0.021 |
| Pointy Nose | 0.664 | 0.663 | 0.667 | 0.674 | +0.010 |
