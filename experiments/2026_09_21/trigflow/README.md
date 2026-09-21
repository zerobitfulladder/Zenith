# 2026-09-21 — trigflow: the capped sine/cosine shear, invertible, and what it generates

**[Lavender]** Take what worked in [`../trigmlp/`](../trigmlp/README.md) (cos and sin as the activation, the turn
capped) and make it an invertible network. Then run it backwards from a label plus noise.

Script: `trigflow.py`. Checkpoint and figure in `results/`.

## The network

Every pixel is an angle in [−π, π), span π so the data sits in [−π/2, π/2]. A layer leaves
half the angles alone (checkerboard, alternating) and shifts the other half:

```
s        =  W [cos θ_A, sin θ_A] + b
s        =  2π · tanh(s / 2π)              at most one turn of movement per layer
θ_B     <-  θ_B + s   (mod 2π)
```

Inverse is subtraction of the same shift. The state is 784 angles at every layer; it has to
be, since a map that reduces the count throws information away and cannot be undone. The
width *inside* the shift function is free, because the shift is a quantity computed from the
untouched half, not part of the state. Here it is a single linear map of the 784 cos/sin
features, so the network has no hidden layer in the ordinary sense.

Readout: mean cosine of all 784 angles against ten fixed random chords, softmax. Trained for
classification only, Adam 1e-4, 15 epochs, depth 8.

## Classification [MEASURED]

| | test | train | shift per layer | roundtrip |
|---|---|---|---|---|
| depth 8, cap one turn | **0.979** | 0.997 | 0.04–0.06 turns, flat over training | 2e-6 rad, float32 |

The uncapped version in [`../catmap/`](../catmap/README.md) scored 0.983 with shifts at 0.1 turns. The cap costs
0.4 points here and was never active: 0.06 turns is nowhere near one. It is a guarantee, not
a gain, on this data.

## Generation from chord + noise [MEASURED, NEGATIVE]

[`mnist_L8_span1pi_cap2pi_generated.png`](results/mnist_L8_span1pi_cap2pi_generated.png): each row a class, the real digit at left, then
three inversions of `chord_y + N(0, sd)` for sd = 0, 0.1, 0.3, 0.6, 1.0.

The digits are there and readable for most classes at sd ≤ 0.3, under a speckle of white
pixels. Noise past 0.6 destroys them. Two measurements explain the picture.

**Where real latents sit relative to their chord.** Over 5000 test images, the per-angle
deviation from the class chord has mean 0.74 rad, median 0.55, 90th percentile 1.71, and its
spread across images is 0.95 rad per angle. Mean cosine to the own chord is only 0.64. The
classifier never asked images to land *on* the chord, only nearer to it than to the other
nine, so the data cloud around a chord is wide and the chord itself is not a typical member.
A noise ball of sd 0.3 is far smaller than the cloud, and a ball of sd 1.0 is isotropic
where the cloud is not.

**Where the speckle comes from.** Decoded angles that land past +π/2 clip to white:

| source | angles on the white side | angles on the black side | white pixels |
|---|---|---|---|
| real image, exact | 0.2% | 35.0% | 7.9% |
| real latent + N(0, 0.3) | 3.8% | 41.8% | 6.4% |
| chord | **8.2%** | 36.8% | 9.4% |
| chord + N(0, 0.3) | 7.8% | 37.6% | 9.0% |

The black side is harmless: black pixels sit at exactly −π/2, the edge of the arc, so any
disturbance puts a third of all angles nominally "out of band" and they clip straight back to
black. **[CONFOUND]** The "41% out of band" figure in [`../nice_flow/README.md`](../nice_flow/README.md) was this
edge effect and not evidence of anything. The white side is the speckle: 8% of pixels
scattered at random, against 0.2% for a real image. A real latent with noise of the same
size keeps the white side at 4% and stays a clean digit.

So: invertibility gives an exact inverse of every real image and a recognisable but speckled
inverse of the chord. It does not give a generator, because nothing in a classification loss
places the chord inside the data cloud or shapes the cloud into something a Gaussian can
sample. Options that stay inside this framing: invert a real latent moved partway toward
another class's chord, or add noise to real latents, both of which decode cleanly. A fitted
distribution over the cloud is the step that turns this into a density model.

## Pulling the cloud onto the chord [MEASURED]

**[Lavender]** "We don't train them to be separated, we train them to be near the chord."
Not with softmax cross-entropy: that is a relative loss, it stops pushing once the own score
leads, and 0.64 is where it stopped. "Near the chord" is its own term:

```
loss  =  cross-entropy  +  λ · (1 − mean_i cos(θ_i − chord_y,i))        λ = 1
```

`--pull 1`. Same network, same settings, one run. [`mnist_L8_span1pi_cap2pi_pull1_generated.png`](results/mnist_L8_span1pi_cap2pi_pull1_generated.png).

| | test | cos to own chord | per-angle dev | spread across images | chord → white side | L1(chord image, class mean image) |
|---|---|---|---|---|---|---|
| classification only | 0.979 | 0.636 | 0.74 rad | 0.95 rad | 8.2% | 0.216 |
| **+ pull** | **0.981** | **0.862** | **0.37** | **0.56** | **0.7%** | **0.083** |

- **Every chord now decodes to a clean, readable digit.** All ten classes, no speckle. The
  chord images look like blurred class prototypes, and by the last column they are: their
  L1 distance to the class mean image dropped by a factor of 2.6.
- **Classification did not pay for it.** Test accuracy went up by 0.2. Near implies
  separated.
- **The cloud halved.** Per-angle deviation 0.74 → 0.37 rad, spread 0.95 → 0.56. It cannot
  collapse, the map preserves volume, but it can concentrate in most directions by stretching
  along a few, and it did. The chord is now inside the cloud rather than in a hollow.
- **Noise at sd 0.1–0.3 gives the prototype with small perturbations**, not variety. The
  samples in a row are nearly identical. Variety would need noise shaped like the cloud, not
  an isotropic ball, and that is where a fitted distribution would come in.
- Noise at sd 0.6 speckles, at 1.0 destroys. Same as before; the pull moved the chord into
  the cloud, it did not widen the region the inverse handles well.

So the label plus small noise now generates a digit, exactly as the original vision had it,
once the loss asks for what the vision assumed. The maximum-likelihood objective in
[`../nice_flow/`](../nice_flow/README.md) contained this term (a von Mises log-density is κ times this cosine); it was
the right pull attached to the wrong build.

## Does one latent angle encode a style? [MEASURED, NEGATIVE]

**[Lavender]** Move a single latent angle of the chord and draw the result, as if each angle
were a style. `traverse.py --cls 3`, [`traverse_3_image.png`](results/traverse_3_image.png), [`traverse_3_diff.png`](results/traverse_3_diff.png). The
angles moved are the ten with the largest spread across real threes (1.4–1.8 rad), at
±1 and ±2 spreads, plus three smallest-spread angles as controls.

No. Moving one latent angle changes essentially **one pixel**, the pixel at that angle's own
position, plus a faint halo around it. Pixel L1 change per +2 spreads is 0.009–0.016 for
the wide angles and 0.001 for the controls, which are background pixels. The digit does not
change slant, width or shape. The latent coordinates are still pixels with a small nonlinear
correction, which is what shifts of 0.05 turns per layer amount to: the network is close to
the identity map with a class-dependent nudge on each pixel.

That is also consistent with the noise result. Style is a direction spanning hundreds of
angles at once, the principal directions of the class cloud, and no single coordinate is one.
Nothing in the loss asks for one angle to mean one thing, so none does. An axis-aligned
style code would need a term that asks for it, or a much less identity-like map.

## Open

- Span 2π with the cap: the seam arm that collapsed in every uncapped run.
- Sweep λ: at what pull strength does classification start to pay, and does the chord image
  sharpen further.
- Style along the cloud's principal directions instead of single angles.
- Variety: noise along the cloud's principal directions instead of an isotropic ball, or
  interpolating between two real latents of one class.
