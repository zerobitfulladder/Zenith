# 2026-09-21 — grok: does a network made of angles have anything to grok?

**[Lavender]** Grokking is the click: train accuracy saturates early, test accuracy sits at
chance for thousands of steps, then jumps. Nanda et al. showed that what a network groks on
modular addition is a *circle*: residues embedded as angles, added, read out by a cosine
cleanup. So a network that is already made of angles, with addition and cleanup built in,
should have nothing to grok. Does it?

Task: `(a + b) mod 97`, a random 30% of the 9409 pairs for training, the rest held out,
full batch, AdamW lr 1e-3, one seed. Script `grok.py`, figure [`grok.png`](results/grok.png).

| arm | params | weight decay | train ≥ 99% | test ≥ 50% | test ≥ 99% | plateau | final test |
|---|---|---|---|---|---|---|---|
| ReLU MLP | 194k | 1 | step 100 | 7950 | 9900 | 9800 | 0.999 |
| ReLU MLP | 194k | 0 | 100 | never | never | ∞ | 0.000 |
| cos+sin MLP, capped | 128k | 1 | 100 | 1600 | **2850** | 2750 | 1.000 |
| **angles + addition + cleanup** | **6.2k** | 0 | 1850 | **1800** | 10300 | **none** | 0.992 |

"Plateau" is the gap between train and test reaching 99%: the time spent memorised but not
generalised.

## What happened [MEASURED]

- **The ReLU MLP groks exactly as in the literature.** Memorised by step 100, test at
  chance until step 8000, then a jump to 99% within 2000 steps. Without weight decay it
  memorises and never generalises; test accuracy 0.000, which is below chance, so the
  memorised function is actively wrong off the training set.
- **The cos+sin activation shortens the plateau 3.5×** but does not remove it. Same shape:
  memorise at 100, generalise at 2850. The activation makes the circle easier to find. It
  does not stop the network from taking the memorisation route first.
- **The torus arm has no plateau.** Train and test rise *together*: test passes 50% at step
  1800 and train passes 99% at step 1850. It cannot memorise, since the only functions it
  can express are "cleanup of an angle sum", so there is no memorisation phase to be stuck
  in. That is the prediction confirmed. With 32 times fewer parameters than the ReLU MLP.
- **But it is slow, twice over.** Nothing moves until step 500, then train climbs to 99% by
  1850, then test crawls from 0.93 to 0.99 over the next 8000 steps and ends at 0.992. The
  click is there, in both curves at once, but the run-up to it is long and the tail is
  longer.

## What it learned [MEASURED]

Take the learned angle of each residue, exp(iφ_r), and Fourier transform it over r. If the
network learned φ_r = 2π·m·r/p + c, a pure frequency, the transform is a single spike:

| | purity, mean | purity, min | frequencies used |
|---|---|---|---|
| input angles, 32 of them | 0.75 | 0.26 | 24 distinct m from 2 to 45 |
| output chords | 0.71 | 0.27 | input and output agree on 97% of angles |

So it learned Nanda's algorithm directly: each angle is a frequency, residue a sits at angle
m·a, the sum sits at m·(a+b), and the chord for y sits at m·y on the same frequency. Most
angles are close to pure. The few impure ones (purity 0.26) are the ones still converging,
and they are why the final test accuracy is 0.992 and not 1.000. κ grew from 10 to 53: the
cleanup sharpened as the frequencies locked in.

## Reading

**The click is the cleanup crossing its threshold.** In the torus arm there is no
memorisation to hide behind, so the click shows in train and test at the same moment: the
frequencies sharpen continuously, and the discrete readout flips from wrong to right for
most pairs within a few hundred steps. That is the "now I understand" moment with nothing in
front of it. The sparse output, a definite residue, is the readout of a dense thing, the
purity of a frequency, and the jump is in the readout, not in the thing.

**Grokking's plateau is the cost of being able to memorise.** The ReLU network can fit the
training set with a lookup and does so first, because that is the fastest way down the
loss. The circle is found later and only under weight decay. Remove the ability to memorise
and the plateau vanishes, but the structure that removes it also makes the landscape harder
to descend: 32 angles each hunting for an integer frequency from a random start is a
non-convex problem with many wrong assignments to get stuck in, and the slow start and slow
tail are that.

**[CONFOUND]** Weight decay differs across arms. It is what makes the ReLU arm grok at
all, and it is meaningless on angle parameters, so the torus arm ran without it. The
plateau-versus-no-plateau comparison stands regardless, since the torus arm's train and test
were simultaneous with or without decay. The speed comparison is not clean: the MLPs had
20–30× the parameters and a smoother landscape. One seed throughout.

## Open

- Seeds, and the frequency-assignment problem: does a better init for the angles (spread
  the frequencies deliberately) remove the slow start and the tail?
- Fewer angles: how many frequencies does p = 97 need? Nanda's network used about 5.
- The same arm on a task that is *not* a group operation, to see it fail: the torus arm
  should be unable to learn (a·b) mod p as a sum of angles unless it discovers discrete logs.
