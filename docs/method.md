# Method configuration

## Grain graphs

Alpha and reference gamma segmentation use four-neighbor connectivity with cubic disorientation thresholds of 3.0 and 7.5 degrees. Components smaller than two pixels are removed. The lowest-MAD pixel supplies each representative orientation. Area, centroid and grain-mean quality fields define the other attributes. Edge interface size is the number of shared four-neighbor pixel pairs.

Each node has 16 features: nine rotation-matrix entries, log1p area, two within-map centroid coordinates scaled to [-1, 1], and mean MAD, BC, BS and Error. Each edge has ten features: child disorientation, log1p shared interface count, minimum and second candidate disorientations, their difference, normalized matching entropy, maximum matching weight and matching fractions at 1.0, 2.5 and 5.0 degrees. Gaussian matching weights use angular scale 2.5 degrees.

Node and edge standardization use the population mean and standard deviation across all respective training nodes or edges. Standard deviations have a floor of 0.0001. Each checkpoint carries its own normalizer, which is used without refitting at inference.

## Orientation relationships

Calibration weights high-confidence grain pairs by overlap coverage times assignment purity. Up to 30000 pairs are sampled without replacement. KS and NW initializations each undergo three symmetry-assignment and rotation-projection updates. The fit with smaller median residual is selected, with KS preferred when the difference is at most 0.01 degrees.

Each fold calibrates its OR only on its inner-training maps using seed 42 plus the fold number. Candidate generation deduplicates child-symmetry operators up to cubic parent symmetry; the fitted ORs give 24 candidates per grain.

The common orientation relationship used for regional parent-orientation estimation is calibrated on the development maps listed in `configs/splits/common_or_split.json`, using seed 42. The five fold-specific relationships are used for learned edge features. For new datasets, preparation fits the common relationship on the supplied development maps.

## Residual models and training

The physics boundary logit is `(minimum_candidate_disorientation - 3.5) / 1.0`, with angles in degrees. The network predicts an additive correction. The last decoder layer is zero initialized. Symmetric decoding uses the absolute difference and elementwise product of endpoint embeddings, concatenated with the edge features. All dropout probabilities are 0.15.

| Setting | MLP | GNN2 |
| --- | --- | --- |
| Hidden width | 96 | 128 |
| Message-passing layers | 0 | 2 |
| Decoder widths | 202, 96, 48, 1 | 266, 128, 64, 1 |
| Epoch limit | 40 | 120 |
| Initial learning rate | 0.0003 | 0.001 |
| Batch size in maps | 4 | 4 |
| AdamW weight decay | 0.0001 | 0.0001 |

Learning rates follow cosine annealing over the configured epoch limit. CUDA training uses automatic mixed precision with gradient scaling; CPU training uses full precision. The checkpoint with highest inner-validation edge average precision is retained. Seeds are 43 through 47 for folds 1 through 5. Floating-point, device and library differences can affect training trajectories.

The total loss is the class-balanced BCE plus 0.01 times mean squared residual logit, 0.05 times triangle transitivity loss and 0.05 times probability calibration MSE. BCE positive weight is the number of labeled same-parent training edges divided by the number of labeled boundary training edges. All terms use labeled edges; triangle consistency uses triangles whose three edges are labeled. For triangle edges with same-parent probabilities a, b and c, the consistency penalty is `a*b*(1-c) + a*c*(1-b) + b*c*(1-a)`, averaged over triangles. The probability MSE uses boundary targets. See `pag_recon/training.py` and `pag_recon/graph/physics.py` for the loss implementation.

## Partitioning and parent orientation

Inference constructs features with each fold's OR, then applies that fold's MLP and GNN2 with their own saved normalizers. Five probabilities are averaged separately for each family. Fusion is 0.4 times MLP plus 0.6 times GNN2 in probability space.

GAEC clips boundary probabilities to [0.00001, 0.99999]. Signed interface cost is shared interface count multiplied by `log((1-p)/p) - log(0.4/0.6)`. The neutral boundary probability is 0.6. Contraction selects the largest positive interface cost and adds costs to common neighbors. A contraction is rejected when the combined region area would exceed 80 percent of the total retained alpha-grain pixel area. Minimum-region cleanup is disabled.

The partition and common OR remain fixed for orientation reconstruction. Every candidate of the largest-area child grain initializes a separate fit. Candidate and cubic-symmetry assignment alternates with area-weighted Huber matrix averaging and proper-rotation projection. Huber delta is 5 degrees. Iteration stops at a direct angular change of 0.0001 degrees or 30 updates. The final orientation minimizes area-weighted mean Huber loss over the initializations. Ambiguity diagnostics do not alter this selection.
