# Zenith Ensemble — Fixed Permutation Relation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Part dimensionality | 128 |
| Total pair dimensionality | 256 |
| Sparsity | 0.1 |
| Active bits per part | 12 |
| Training positive pairs | 400 |
| Test sources | 120 |
| Templates per node (k) | 10 |
| Ensemble nodes (M) | 30 |
| Learning rate | 0.03 |
| Training epochs | 800 |
| Candidate trials per test source | 20 |

## Task

Training data contains only positive pairs [X, P(X)], where P is one fixed hidden permutation.
At test time, X is unseen. The model must rank the true Y = P(X) against distractor Y values.

## Methods

- **Raw pair NN**: score candidate [X, Y] by its maximum cosine similarity to any training pair.
- **Activation-space NN**: score candidate [X, Y] by the cosine similarity of its ensemble activation to stored training activations.
- **Direct template score**: score candidate [X, Y] by the mean top-1 |correlation| across Zenith nodes, without nearest-neighbor lookup.

## Results

| Distractors | Method | Accuracy | Avg Rank | Avg Margin | Chance |
|-------------|--------|----------|----------|------------|--------|
| 1 | Raw pair NN | 98.54% | 0.01 | +0.1048 | 50.00% |
| 1 | Activation-space NN | 53.75% | 0.46 | +0.0097 | 50.00% |
| 1 | Direct template score | 99.58% | 0.00 | +0.0501 | 50.00% |
| 3 | Raw pair NN | 96.33% | 0.04 | +0.0782 | 25.00% |
| 3 | Activation-space NN | 25.17% | 1.43 | -0.0482 | 25.00% |
| 3 | Direct template score | 98.67% | 0.01 | +0.0376 | 25.00% |
| 7 | Raw pair NN | 91.88% | 0.10 | +0.0584 | 12.50% |
| 7 | Activation-space NN | 12.33% | 3.31 | -0.0838 | 12.50% |
| 7 | Direct template score | 96.50% | 0.04 | +0.0296 | 12.50% |
| 15 | Raw pair NN | 86.38% | 0.20 | +0.0433 | 6.25% |
| 15 | Activation-space NN | 4.17% | 7.02 | -0.1113 | 6.25% |
| 15 | Direct template score | 92.54% | 0.09 | +0.0227 | 6.25% |

## Interpretation

- **Activation-space NN > Raw NN** means the learned activation manifold organizes valid permutation pairs better than raw input similarity does.
- **Direct template score > Raw NN** is stronger: it suggests the templates themselves capture some of the fixed transformation, not only nearest-neighbor memory.
- **Avg rank** asks where the true candidate falls among all candidates. 0 means best.
- **Avg margin** is true score minus best false score. Positive is good.

The direct template score beats raw NN in the hardest setting. That is evidence that Zenith templates learned something about the fixed permutation itself.

### Caveat

A fixed permutation is a very specific linear relation. Success here would not imply general relational reasoning.
It would only show that Zenith can internalize one reusable transformation on sparse vectors.
