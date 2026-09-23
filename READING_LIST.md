# Reading list — the stepping stones under Zenith

Everything in this repository was built from intuition first and named
afterwards. This list gives each of those intuitions its established name and
the paper where the idea was first pinned down. It is deliberately limited to
foundations: the works that *define* a method, a metric, or a problem that this
project uses. Incremental results are left out on purpose; branch from these.

Each entry says, in one line, which part of the project it grounds. Read the
"why" column against `README.md`, which walks through what was actually built.

Suggested order for a first pass, if you read nothing else:
Rumelhart & Zipser 1985 → MacQueen 1967 → Kohonen 1982 → Grossberg 1976 →
Olshausen & Field 1996 → Földiák 1990 → Krotov & Hopfield 2019 →
Coates, Lee & Ng 2011 → Csurka et al. 2004 → McCloskey & Cohen 1989 →
Carpenter & Grossberg 1987 → Rao & Ballard 1999 → Hinton et al. 1995 →
Lillicrap et al. 2020.

---

## 1. Competitive learning, winner-take-all, self-organisation

The unit at the heart of the project (templates compete, one winner learns, the
winner rotates toward the input) is *competitive learning*. These are its
origins.

| Year | Paper | Why it matters here |
|---|---|---|
| 1973 | von der Malsburg, *Self-organization of orientation sensitive cells in the striate cortex*, Kybernetik | First self-organising competitive model with normalised weights and lateral inhibition; orientation detectors emerge unsupervised, exactly the L1 edge vocabulary. |
| 1976 | Grossberg, *Adaptive pattern classification and universal recoding, I: parallel development and coding of neural feature detectors*, Biological Cybernetics | Formal competitive learning; introduces the stability–plasticity dilemma that the whole continual-learning strand is about. |
| 1985 | Rumelhart & Zipser, *Feature discovery by competitive learning*, Cognitive Science | The canonical statement: winner-take-all, winner moves toward the input, "leaky" learning for losers. The monopolist / dead-unit problem is described here. |
| 1982 | Kohonen, *Self-organized formation of topologically correct feature maps*, Biological Cybernetics | Self-organising maps; winner-plus-neighbourhood learning, the reference point for every "how many units, how close" question. |
| 1990 | Kohonen, *The self-organizing map*, Proceedings of the IEEE | Consolidated SOM and *Learning Vector Quantization* (LVQ, LVQ2.1): the supervised push-the-wrong-winner-away rule the project rediscovered as "repulsion". |
| 1988 | DeSieno, *Adding a conscience to competitive learning*, IEEE ICNN | The "conscience" (win-frequency penalty) used to keep hypercolumns from starving; also the reason equal-usage pressure fights density adaptation. |
| 1990 | Ahalt, Krishnamurthy, Chen & Melton, *Competitive learning algorithms for vector quantization*, Neural Networks | Frequency-sensitive competitive learning; the family of "who gets to learn" rules compared against plain winner-take-all. |
| 1991 | Martinetz & Schulten, *A "neural-gas" network learns topologies*, ICANN | Rank-based soft competition (the "soft losers" idea) framed as vector quantisation without a fixed grid. |
| 1995 | Fritzke, *A growing neural gas network learns topologies*, NIPS | Growing the unit count on demand: the ancestor of "hire a template on error". |
| 2000 | Maass, *On the computational power of winner-take-all*, Neural Computation | Why a WTA layer is a legitimate compute primitive, not a hack. |

## 2. Vector quantisation, k-means, codebooks

"Identity-only coding" is vector quantisation. The online rule with step 1/n is
MacQueen's sequential k-means; the error law used throughout is Zador's.

| Year | Paper | Why it matters here |
|---|---|---|
| 1967 | MacQueen, *Some methods for classification and analysis of multivariate observations*, Berkeley Symposium | Online (sequential) k-means with per-cluster step 1/n — the learning rule of the current L1. |
| 1982 | Lloyd, *Least squares quantization in PCM*, IEEE Trans. Information Theory (written 1957) | Batch k-means / Lloyd's algorithm; the fixed point the online rule approaches. |
| 1980 | Linde, Buzo & Gray, *An algorithm for vector quantizer design*, IEEE Trans. Communications | The LBG codebook algorithm; "codebook" as a term comes from here. |
| 1984 | Gray, *Vector quantization*, IEEE ASSP Magazine | The tutorial; rate–distortion framing of "precision costs templates, not coefficients". |
| 1982 | Zador, *Asymptotic quantization error of continuous signals and the quantization dimension*, IEEE Trans. Information Theory | Distortion ∝ K^(−2/d): the dimensionality law that says reduce d before adding templates. |
| 2001 | Dhillon & Modha, *Concept decompositions for large sparse text data using clustering*, Machine Learning | Spherical k-means: k-means with cosine similarity on unit vectors — the project's core, named. |
| 2005 | Banerjee, Dhillon, Ghosh & Sra, *Clustering on the unit hypersphere using von Mises-Fisher distributions*, JMLR | The probabilistic reading of spherical k-means; what "temperature" and soft assignment mean on a sphere. |
| 2007 | Arthur & Vassilvitskii, *k-means++: the advantages of careful seeding*, SODA | The seeding that removed every dead template; init from data, not noise. |
| 2011 | Jégou, Douze & Schmid, *Product quantization for nearest neighbor search*, IEEE TPAMI | Several independent small codebooks whose joint index forms the code — the "ensemble of nodes" pattern-completion design. |
| 2011 | Coates, Lee & Ng, *An analysis of single-layer networks in unsupervised feature learning*, AISTATS | k-means on image patches + pooling + linear classifier beats deep models at the time: the two-rung k-means stack, published. |
| 2012 | Coates & Ng, *Learning feature representations with K-means*, Neural Networks: Tricks of the Trade | The practical recipe (whitening, patch size, pooling, dictionary size) behind the same result. |

## 3. Hebbian rules, weight normalisation, local plasticity

Unit-norm templates, rotation instead of addition, residual (deflation)
learning, and anti-Hebbian lateral inhibition all have named ancestors.

| Year | Paper | Why it matters here |
|---|---|---|
| 1949 | Hebb, *The Organization of Behavior* (book) | The postulate. |
| 1982 | Oja, *A simplified neuron model as a principal component analyzer*, J. Mathematical Biology | Hebbian learning with an implicit norm constraint; the weight stays on the sphere — the additive cousin of the geodesic step. |
| 1989 | Sanger, *Optimal unsupervised learning in a single-layer linear feedforward neural network*, Neural Networks | Generalised Hebbian algorithm: each unit learns from the residual left by the previous units — the "sequential residual, units explain alone" scheme. |
| 1990 | Földiák, *Forming sparse representations by local anti-Hebbian learning*, Biological Cybernetics | Anti-Hebbian lateral weights + threshold homeostasis; tested directly in the project and found to lose to residual learning. |
| 1982 | Bienenstock, Cooper & Munro, *Theory for the development of neuron selectivity*, J. Neuroscience | BCM sliding threshold; the reference for any activity-dependent threshold or "adaptation" mechanism. |
| 2008 | Turrigiano, *The self-tuning neuron: synaptic scaling of excitatory synapses*, Cell | The biology behind "a template has a fixed norm budget to spend". |
| 2019 | Krotov & Hopfield, *Unsupervised learning by competing hidden units*, PNAS | The modern local competitive-Hebbian benchmark on MNIST/CIFAR; closest published relative of the unit. |
| 2022 | Moraitis, Toichkin, Journé, Chua & Guo, *SoftHebb: Bayesian inference in unsupervised Hebbian soft winner-take-all networks*, Neuromorphic Computing and Engineering | Soft WTA as Bayesian inference; where the "contrast message" idea comes from. |
| 2023 | Journé, Garcia Rodriguez, Guo & Moraitis, *Hebbian deep learning without feedback*, ICLR | Stacked SoftHebb with pooling; already read against the project (see memory). |
| 2019 | Illing, Gerstner & Brea, *Biologically plausible deep learning — but how far can we go with shallow networks?*, Neural Networks | Systematic benchmark of local rules with linear readouts on MNIST; calibrates what a gradient-free stack should reach. |

## 4. Sparse coding, dictionaries, pursuit and settling

"Dictionary", "matching pursuit", "LCA settling", "coherence" and "overcomplete"
all come from this literature.

| Year | Paper | Why it matters here |
|---|---|---|
| 1961 | Barlow, *Possible principles underlying the transformation of sensory messages*, in Sensory Communication | Redundancy reduction / efficient coding; the principle behind contrast extraction and sparse messages. |
| 1994 | Field, *What is the goal of sensory coding?*, Neural Computation | Sparse versus compact codes; why few-active-units is the target rather than low dimension. |
| 1996 | Olshausen & Field, *Emergence of simple-cell receptive field properties by learning a sparse code for natural images*, Nature | The founding sparse-coding paper; edges as a learned overcomplete dictionary. |
| 1997 | Olshausen & Field, *Sparse coding with an overcomplete basis set: a strategy employed by V1?*, Vision Research | The fuller treatment, including the inference/learning split the project mirrors as read/learn. |
| 1993 | Mallat & Zhang, *Matching pursuits with time-frequency dictionaries*, IEEE Trans. Signal Processing | Matching pursuit: the sequential greedy residual encoder used for learning. |
| 1993 | Pati, Rezaiifar & Krishnaprasad, *Orthogonal matching pursuit*, Asilomar Conference | The orthogonalised variant; the partial-correlation view of "who explains what". |
| 2008 | Rozell, Johnson, Baraniuk & Olshausen, *Sparse coding via thresholding and local competition in neural circuits*, Neural Computation | LCA: parallel lateral-inhibition settling; the read-time encoder of "pursuit learns, settling reads". |
| 2006 | Aharon, Elad & Bruckstein, *K-SVD: an algorithm for designing overcomplete dictionaries for sparse representation*, IEEE Trans. Signal Processing | Dictionary learning as alternating sparse code / atom update; the batch reference point. |
| 2010 | Mairal, Bach, Ponce & Sapiro, *Online dictionary learning for sparse coding*, JMLR | Online dictionary learning; the same one-sample-at-a-time constraint as this project. |

## 5. Lateral inhibition, normalisation, cortical columns

The vocabulary minicolumn / hypercolumn / layer, and the inhibition models
(subtractive, shunting, divisive) tested in the earliest experiments.

| Year | Paper | Why it matters here |
|---|---|---|
| 1962 | Hubel & Wiesel, *Receptive fields, binocular interaction and functional architecture in the cat's visual cortex*, J. Physiology | Simple/complex cells: the template-then-pool motif. |
| 1977 | Hubel & Wiesel, *Functional architecture of macaque monkey visual cortex* (Ferrier Lecture), Proc. Royal Society B | The hypercolumn. |
| 1997 | Mountcastle, *The columnar organization of the neocortex*, Brain | Minicolumns inside hypercolumns; the project's naming. |
| 2004 | Douglas & Martin, *Neuronal circuits of the neocortex*, Annual Review of Neuroscience | The canonical microcircuit; local inhibition, long-range excitation (the reason sign lives inside a hypercolumn, not in the message). |
| 1992 | Heeger, *Normalization of cell responses in cat striate cortex*, Visual Neuroscience | Divisive (shunting) normalisation, the second inhibition model tried. |
| 2012 | Carandini & Heeger, *Normalization as a canonical neural computation*, Nature Reviews Neuroscience | The review; why L2-normalising every layer is defensible. |
| 1995 | Desimone & Duncan, *Neural mechanisms of selective visual attention*, Annual Review of Neuroscience | Biased competition: top-down gain biases a local competition — exactly the gain-feedback design. |
| 2009 | Reynolds & Heeger, *The normalization model of attention*, Neuron | Attention as multiplicative gain inside a normalisation circuit. |
| 1992 | Atick & Redlich, *What does the retina know about natural scenes?*, Neural Computation | Centre-surround as decorrelation/whitening; the justification for the LGN mean-centring stage. |

## 6. Convolution, pooling, invariance

Weight sharing, max pooling, and why an identity code needs pooling before it
can be compared.

| Year | Paper | Why it matters here |
|---|---|---|
| 1980 | Fukushima, *Neocognitron: a self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position*, Biological Cybernetics | Unsupervised competitive S-cells plus pooling C-cells: the gradient-free convolutional stack, forty-five years early. |
| 1998 | LeCun, Bottou, Bengio & Haffner, *Gradient-based learning applied to document recognition*, Proceedings of the IEEE | CNNs and MNIST itself. |
| 1999 | Riesenhuber & Poggio, *Hierarchical models of object recognition in cortex*, Nature Neuroscience | HMAX: template matching alternated with MAX pooling. |
| 2010 | Boureau, Ponce & LeCun, *A theoretical analysis of feature pooling in visual recognition*, ICML | Max versus average pooling and when each wins. |
| 2006 | Lazebnik, Schmid & Ponce, *Beyond bags of features: spatial pyramid matching*, CVPR | Counting codewords per spatial cell — the per-cell tally readout, named. |
| 2009 | Jarrett, Kavukcuoglu, Ranzato & LeCun, *What is the best multi-stage architecture for object recognition?*, ICCV | Random filters + rectification + pooling already work; the "dense layers are channels" finding. |
| 2011 | Saxe, Koh, Chen, Bhand, Suresh & Ng, *On random weights and unsupervised feature learning*, ICML | Why an untrained layer preserves geometry and what learning actually buys. |

## 7. Bag-of-words, n-tuple and counting readouts

The "çetele" / tally: count which template fired with which label, read the
ratio. This is the bag-of-visual-words pipeline and, older still, the n-tuple
classifier.

| Year | Paper | Why it matters here |
|---|---|---|
| 1959 | Bledsoe & Browning, *Pattern recognition and reading by machine*, Eastern Joint Computer Conference | The n-tuple classifier: lookup tables of observed sub-patterns, counted per class. Gradient-free, one-pass, no forgetting by construction. |
| 1984 | Aleksander, Thomas & Bowden, *WISARD: a radical step forward in image recognition*, Sensor Review | The hardware n-tuple machine; weightless neural networks. |
| 2003 | Sivic & Zisserman, *Video Google: a text retrieval approach to object matching in videos*, ICCV | Quantise local descriptors into "visual words", count them. |
| 2004 | Csurka, Dance, Fan, Willamowski & Bray, *Visual categorization with bags of keypoints*, ECCV Workshop | Bag-of-visual-words classification: k-means codebook + histogram + classifier. |
| 1997 | Domingos & Pazzani, *On the optimality of the simple Bayesian classifier under zero-one loss*, Machine Learning | Why counting P(y|t)/P(y) works even when independence is false. |
| 1967 | Cover & Hart, *Nearest neighbor pattern classification*, IEEE Trans. Information Theory | The nearest-template read, with its error bound. |
| 2016 | Alain & Bengio, *Understanding intermediate layers using linear classifier probes*, arXiv | The linear probe as a measurement tool, and its limits (it cannot judge a message format a reconstructive layer must use). |
| 1972 | Spärck Jones, *A statistical interpretation of term specificity and its application in retrieval*, Journal of Documentation | Inverse document frequency; the weighting used for attribute retrieval and the P(y|t)/P(y) ratio in the table. |
| 2004 | Hoyer, *Non-negative matrix factorization with sparseness constraints*, JMLR | The Hoyer sparseness measure used as a diagnostic (and NMF as the nonnegative-dictionary reference point). |

## 8. Associative memory, pattern completion, sparse distributed codes

The OR'ed array, partial-cue reads, and capacity walls.

| Year | Paper | Why it matters here |
|---|---|---|
| 1969 | Willshaw, Buneman & Longuet-Higgins, *Non-holographic associative memory*, Nature | Binary OR-stored associative memory; the ancestor of writing several channels into one sparse array. |
| 1971 | Marr, *Simple memory: a theory for archicortex*, Phil. Trans. Royal Society B | Pattern completion from a partial cue, sparse codes, capacity. |
| 1982 | Hopfield, *Neural networks and physical systems with emergent collective computational abilities*, PNAS | Content-addressable memory, energy, spurious attractors — the "energy peaks above truth" seen in the permutation arc. |
| 1983 | Hopfield, Feinstein & Palmer, *"Unlearning" has a stabilizing effect in collective memories*, Nature | Dream/unlearning of spurious states; the "dream" experiments. |
| 1983 | Crick & Mitchison, *The function of dream sleep*, Nature | The same idea from the biology side. |
| 1988 | Kanerva, *Sparse Distributed Memory* (MIT Press) | High-dimensional sparse addresses, graceful degradation, best-match reads. |
| 2009 | Kanerva, *Hyperdimensional computing: an introduction to computing in distributed representation with high-dimensional random vectors*, Cognitive Computation | Binding and superposition in high dimensions; why concatenated channels behave well and where they don't. |
| 1986 | Hinton, McClelland & Rumelhart, *Distributed representations*, in Parallel Distributed Processing vol. 1 | Coarse coding: a value written as an overlapping patch of cells — the place code. |
| 2016 | Krotov & Hopfield, *Dense associative memory for pattern recognition*, NIPS | Modern high-capacity associative memory; where nearest-template reads and softmax reads meet. |

## 9. Population codes, place codes, tile coding

The raised-cosine bump per channel, and density-adaptive tiling.

| Year | Paper | Why it matters here |
|---|---|---|
| 1975 | Albus, *A new approach to manipulator control: the cerebellar model articulation controller (CMAC)*, J. Dynamic Systems, Measurement, and Control | Coarse coding of continuous inputs into overlapping cells, for motor control. |
| 1986 | Georgopoulos, Schwartz & Kettner, *Neuronal population coding of movement direction*, Science | Population vector: a value carried by a bump over tuned cells. |
| 2000 | Pouget, Dayan & Zemel, *Information processing with population codes*, Nature Reviews Neuroscience | Reading a value (peak, centroid) out of a bump, and the noise trade-offs of bump width. |
| 1996 | Sutton, *Generalization in reinforcement learning: successful examples using sparse coarse coding*, NIPS | Tile coding as the function approximator for RL — the drone/pole memories. |
| 1996 | McCallum, *Reinforcement learning with selective perception and hidden state* (PhD thesis, U. Rochester) | U-Tree: split the state space where it matters — adaptive, density-following tiling. |

## 10. Continual learning, forgetting, stability–plasticity

The no-forgetting property of winner-only learning, and the baselines it is
measured against.

| Year | Paper | Why it matters here |
|---|---|---|
| 1989 | McCloskey & Cohen, *Catastrophic interference in connectionist networks: the sequential learning problem*, Psychology of Learning and Motivation | Names the problem and the split-task protocol. |
| 1999 | French, *Catastrophic forgetting in connectionist networks*, Trends in Cognitive Sciences | Why shared weights forget and localised codes do not. |
| 1987 | Carpenter & Grossberg, *A massively parallel architecture for a self-organizing neural pattern recognition machine*, Computer Vision, Graphics, and Image Processing | ART-1: match → vigilance → commit a fresh unit. "Hire on error" and "specialisation protects itself" are ART. |
| 1995 | McClelland, McNaughton & O'Reilly, *Why there are complementary learning systems in the hippocampus and neocortex*, Psychological Review | Fast episodic store plus slow consolidation; the framing for "freeze the vocabulary, stream the counts" and for sleep. |
| 2017 | Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, PNAS | EWC: the standard gradient-side baseline the project's tables should eventually be compared to. |
| 2019 | Parisi, Kemker, Part, Kanan & Wermter, *Continual lifelong learning with neural networks: a review*, Neural Networks | The map of the field (regularisation, replay, architectural growth). |
| 1991 | Jacobs, Jordan, Nowlan & Hinton, *Adaptive mixtures of local experts*, Neural Computation | Experts + gate; the hypercolumn-as-specialist design and why the gate criterion must match the experts. |

## 11. Feedback, predictive coding, analysis by synthesis, generation

Top-down gain, residual surfacing, and one set of weights serving recognition
and generation.

| Year | Paper | Why it matters here |
|---|---|---|
| 1992 | Mumford, *On the computational architecture of the neocortex II: the role of cortico-cortical loops*, Biological Cybernetics | Feedback carries the higher layer's hypothesis; the lower layer returns the residual. |
| 1999 | Rao & Ballard, *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects*, Nature Neuroscience | Predictive coding: exactly "residual surfacing". |
| 2003 | Lee & Mumford, *Hierarchical Bayesian inference in the visual cortex*, JOSA A | Ping-pong settling between layers as inference. |
| 2005 | Friston, *A theory of cortical responses*, Phil. Trans. Royal Society B | The free-energy version; "energy search" in the permutation arc is this. |
| 1995 | Hinton, Dayan, Frey & Neal, *The wake-sleep algorithm for unsupervised neural networks*, Science | Recognition and generation pathways trained locally from each other — the "generation is a diagnostic" stance. |
| 2002 | Hinton, *Training products of experts by minimizing contrastive divergence*, Neural Computation | Contrastive divergence: the "negatives from the model's own samples" training tried on the permutation task. |
| 2009 | George & Hawkins, *Towards a mathematical theory of cortical micro-circuits*, PLoS Computational Biology | Hierarchical temporal memory: columns, sparse codes, temporal pooling, belief propagation up and down. |
| 2017 | Hawkins, Ahmad & Cui, *A theory of how columns in the neocortex enable learning the structure of the world*, Frontiers in Neural Circuits | The cortical-column-as-unit view the project shares. |

## 12. Learning without backpropagation

The constraint the whole project lives under, and the published alternatives.

| Year | Paper | Why it matters here |
|---|---|---|
| 1989 | Crick, *The recent excitement about neural networks*, Nature | The original biological objection to backprop. |
| 2016 | Lillicrap, Cownden, Tweed & Akerman, *Random synaptic feedback weights support error backpropagation for deep learning*, Nature Communications | Feedback alignment; separate downward weights (cf. "feedback is not the transpose"). |
| 2020 | Lillicrap, Santoro, Marris, Akerman & Hinton, *Backpropagation and the brain*, Nature Reviews Neuroscience | The survey of what a brain-plausible credit assignment could be. |
| 2019 | Whittington & Bogacz, *Theories of error back-propagation in the brain*, Trends in Cognitive Sciences | Predictive coding and other local approximations, compared. |
| 2017 | Scellier & Bengio, *Equilibrium propagation*, Frontiers in Computational Neuroscience | Settling-based credit assignment; the closest published relative of "ping-pong with a clamped label". |
| 2022 | Hinton, *The forward-forward algorithm: some preliminary investigations*, arXiv | Layer-local positive/negative training with no backward pass. |
| 2016 | Frémaux & Gerstner, *Neuromodulated spike-timing-dependent plasticity, and theory of three-factor learning rules*, Frontiers in Neural Circuits | Three-factor rules: local Hebbian term gated by a global scalar — the "value only at the selector" law. |

## 13. Geometry on the sphere

Rotation instead of addition, and why the constraint is exact.

| Year | Paper | Why it matters here |
|---|---|---|
| 1840 | Rodrigues, *Des lois géométriques qui régissent les déplacements d'un système solide dans l'espace…*, J. de Mathématiques Pures et Appliquées | The rotation formula the update is named after. |
| 1998 | Edelman, Arias & Smith, *The geometry of algorithms with orthogonality constraints*, SIAM J. Matrix Analysis and Applications | Geodesics, tangent projection and retraction on spheres and Stiefel manifolds. |
| 2008 | Absil, Mahony & Sepulchre, *Optimization Algorithms on Matrix Manifolds* (Princeton) | The textbook; "supervised rotation is Riemannian SGD" cites it. |
| 2013 | Bonnabel, *Stochastic gradient descent on Riemannian manifolds*, IEEE Trans. Automatic Control | Convergence of online geodesic steps. |
| 2016 | Salimans & Kingma, *Weight normalization*, NIPS | The unit-norm weight constraint in modern nets, for contrast. |

## 14. Statistics under the hood

Pearson correlation, partial correlation, whitening, the n^(−1/d) law.

| Year | Paper | Why it matters here |
|---|---|---|
| 1895 | Pearson, *Note on regression and inheritance in the case of two parents*, Proc. Royal Society | Correlation as the match score once both sides are centred and unit length. |
| 1980 | Stone, *Optimal rates of convergence for nonparametric estimators*, Annals of Statistics | Nearest-neighbour style error falls only as n^(−1/d); the "prototype memory dimensionality law". |
| 1957 | Bellman, *Dynamic Programming* (Princeton) | The curse of dimensionality, named. |
| 1997 | Bell & Sejnowski, *The "independent components" of natural scenes are edge filters*, Vision Research | Whitening + independence produces edges; why "learning buys whitening" is a real result. |
| 2000 | Hyvärinen & Oja, *Independent component analysis: algorithms and applications*, Neural Networks | Whitening, decorrelation, and the equal-norm frame the geodesic rule settles into. |

## 15. Binding, composition, disentanglement

Why an index cannot say "woman with a mustache" and what would.

| Year | Paper | Why it matters here |
|---|---|---|
| 1988 | Fodor & Pylyshyn, *Connectionism and cognitive architecture: a critical analysis*, Cognition | Systematicity: "john loves alice" versus "alice loves john" — the composition wall. |
| 1990 | Smolensky, *Tensor product variable binding and the representation of symbolic structures in connectionist systems*, Artificial Intelligence | Binding by outer product; what a message would need to express relations. |
| 1995 | Plate, *Holographic reduced representations*, IEEE Trans. Neural Networks | Binding by circular convolution in a fixed-width vector. |
| 2013 | Bengio, Courville & Vincent, *Representation learning: a review and new perspectives*, IEEE TPAMI | Disentangled factors as the goal; where "delta with the male axis projected out" sits. |

## 16. Categorisation, prototypes and exemplars (cognitive science)

The project's "template = the picture" is prototype theory; the "many styles of
a 2" argument is exemplar theory.

| Year | Paper | Why it matters here |
|---|---|---|
| 1972 | Reed, *Pattern recognition and categorization*, Cognitive Psychology | Prototype models of categorisation. |
| 1986 | Nosofsky, *Attention, similarity, and the identification–categorization relationship*, J. Experimental Psychology: General | Exemplar model (GCM): classification by summed similarity to stored instances, with attention weights on dimensions. |
| 1992 | Kruschke, *ALCOVE: an exemplar-based connectionist model of category learning*, Psychological Review | Exemplar nodes + learned attention + learned association weights — a tally readout over a prototype layer, in cognitive-science clothing. |

## 17. Learning from a second modality (the label as a sensory stream)

| Year | Paper | Why it matters here |
|---|---|---|
| 1992 | Becker & Hinton, *Self-organizing neural network that discovers surfaces in random-dot stereograms*, Nature | Two streams agree with each other; no external target. |
| 1994 | de Sa, *Learning classification with unlabeled data*, NIPS | Minimising disagreement between two modalities as the supervisory signal — the "label is co-occurring input, never a verdict" stance. |
| 1998 | Blum & Mitchell, *Combining labeled and unlabeled data with co-training*, COLT | The same idea as an algorithm with guarantees. |

## 18. Sequences, lags, motor control and reinforcement

For the drone, pole and arm rigs and the lag-structured memories.

| Year | Paper | Why it matters here |
|---|---|---|
| 1981 | Takens, *Detecting strange attractors in turbulence*, Dynamical Systems and Turbulence (Springer LNM 898) | Delay embedding: a present over lagged keys is a state — the theory behind "no next-slots". |
| 1989 | Waibel, Hanazawa, Hinton, Shikano & Lang, *Phoneme recognition using time-delay neural networks*, IEEE Trans. ASSP | Lagged taps as input channels. |
| 1990 | Elman, *Finding structure in time*, Cognitive Science | Context as recurrent state; the alternative the project chose not to take. |
| 2016 | Hawkins & Ahmad, *Why neurons have thousands of synapses, a theory of sequence memory in neocortex*, Frontiers in Neural Circuits | Minicolumn-based sequence memory with local rules. |
| 1997 | Schultz, Dayan & Montague, *A neural substrate of prediction and reward*, Science | Dopamine as reward-prediction error; the scalar that may touch the selector only. |
| 1992 | Watkins & Dayan, *Q-learning*, Machine Learning | Value on state–action templates. |
| 2018 | Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (MIT Press) | Tile coding, eligibility traces, SARSA — the whole toolkit named. |
| 2011 | Ross, Gordon & Bagnell, *A reduction of imitation learning and structured prediction to no-regret online learning* (DAgger), AISTATS | The teacher-in-the-loop imitation scheme the drone rigs used. |
| 1989 | Pomerleau, *ALVINN: an autonomous land vehicle in a neural network*, NIPS | Behaviour cloning, the imitation baseline. |
| 1991 | Sutton, *Dyna, an integrated architecture for learning, planning, and reacting*, SIGART Bulletin | Planning with a learned model; the queued lookahead fix for the babbled world model. |
| 1999 | Ng, Harada & Russell, *Policy invariance under reward transformations: theory and application to reward shaping*, ICML | Potential-based shaping, as used on the drone. |
| 1960 | Kalman, *Contributions to the theory of optimal control*, Boletín de la Sociedad Matemática Mexicana | The LQR teacher on the pole. |
| 1956 | Hassenstein & Reichardt, *Systemtheoretische Analyse der Zeit-, Reihenfolgen- und Vorzeichenauswertung bei der Bewegungsperzeption des Rüsselkäfers Chlorophanus*, Zeitschrift für Naturforschung B | The Reichardt correlator: direction selectivity from delayed copies, which the delay-line slices reproduce. |

## 19. Datasets

| Year | Paper | Why it matters here |
|---|---|---|
| 1998 | LeCun, Bottou, Bengio & Haffner (above) | MNIST. |
| 2017 | Xiao, Rasul & Vollgraf, *Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms*, arXiv | Fashion-MNIST. |
| 2015 | Liu, Luo, Wang & Tang, *Deep learning face attributes in the wild*, ICCV | CelebA and its 40 attributes. |

---

## Reading by project theme (cross-index)

- **The unit itself** — §1, §2 (Dhillon & Modha, MacQueen), §3 (Oja, Sanger, Krotov & Hopfield), §13.
- **Online / gradient-free / no backprop** — §3, §12, §13 (Bonnabel), §2 (MacQueen), §4 (Mairal).
- **No forgetting** — §10 entire, §1 (Grossberg 1976, Fritzke), §7 (Bledsoe & Browning: counting never forgets).
- **Counting / tally readout** — §7 entire, §6 (Lazebnik), §16 (Kruschke).
- **Dictionary / codebook** — §2 entire, §4 entire.
- **Messages between layers, pooling, invariance** — §6, §8 (Hinton et al. 1986 coarse coding), §9.
- **Feedback and generation** — §11, §12 (Lillicrap 2016 on separate downward weights).
- **Composition** — §15.
- **Temporal / motor / RL** — §18, §9 (Albus, Sutton, McCallum).
