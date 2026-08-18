# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of bngo
- Core graph structures (DAG, undirected graphs)
- Discrete factor operations
- Tabular CPD implementation
- Bayesian Network model with discrete variables
- Variable Elimination inference algorithm
- PC structure learning algorithm with Meek's orientation rules
- Maximum Likelihood parameter learning
- Data simulation via ancestral sampling
- Prediction using MAP inference
- Independence tests (Chi-square, Fisher's Z)
- Example models (Student, Alarm, Cancer, Sprinkler)
- Comprehensive test suite
- Example application demonstrating all features
- Full documentation (README, IMPLEMENTATION guide)
- Score-based structure learning: Hill Climbing and MMHC
- Decomposable network scores: BIC, AIC, BDeu and K2
- G-squared conditional independence test

### Features

#### Graph Operations
- DAG with cycle detection
- Topological sorting
- Ancestor/descendant queries
- Moral graph construction
- Graph copying and manipulation

#### Factor Operations
- Factor multiplication
- Marginalization
- Reduction with evidence
- Normalization
- Max-marginalization for MAP queries

#### Bayesian Networks
- Network definition with edges and CPDs
- Model validation
- Data simulation
- Parameter learning (MLE with Laplace smoothing)
- Prediction for missing values
- Full inference integration

#### Inference
- Variable Elimination (exact inference)
- Posterior probability queries
- MAP (Maximum A Posteriori) queries
- Evidence conditioning
- Mixed Variable Elimination for CLG (Conditional Linear Gaussian) models
- Continuous variable queries with discrete/continuous evidence
- Discrete variable queries with continuous evidence (Bayesian updating)
- Joint Gaussian construction from Linear Gaussian CPDs
- Moment-matched Gaussian mixture approximation for hidden discrete variables

#### Structure Learning
- PC algorithm with constraint-based learning
- Chi-square independence test
- V-structure detection
- Meek's orientation rules (R1-R4)
- Configurable significance levels
- `HillClimbSearch`: greedy score-based search over edge additions, deletions and
  reversals, with optional caps on the number of parents, on the candidate parents of
  each variable, and on the number of steps. Every accepted change strictly increases
  the score, so the search terminates by construction
- `MMHCEstimator`: Max-Min Hill Climbing (Tsamardinos, Brown and Aliferis, 2006).
  MMPC narrows the candidate parents of every variable by local independence tests,
  then greedy search picks and orients the edges among them. `LearnSkeleton` exposes
  the MMPC phase on its own
- `StructureEstimator`, the interface every structure learner satisfies, so one can be
  swapped for another
- Restricting the search is what suits MMHC to large networks. Measured on a sparse
  synthetic network of binary variables over 2000 rows, MMHC overtakes unrestricted
  hill climbing at around a hundred variables: at 160 variables it takes 43 s against
  2 min 10 s, and at 240 variables 1 min 8 s. Below that size hill climbing is faster,
  and the more accurate of the two throughout

#### Network Scores
- `NewBIC` and `NewAIC`: log likelihood penalised per free parameter
- `NewBDeu` and `NewK2`: marginal likelihood under a Dirichlet prior
- `ScoreDAG`: the score of a whole network, as the sum of its local scores
- Local scores are cached, since search prices the same family many times over
- Sufficient statistics cover only the parent configurations that occur, so a wide
  parent set costs memory in proportion to the sample rather than to the product of
  the parent cardinalities
- Declared cardinalities override the states read off the data, which keeps a state
  absent from the sample from changing the penalty term

#### Independence Tests
- `GSquareTest`: the likelihood ratio test, with degrees of freedom adjusted to the
  states actually observed in each stratum. Without that adjustment a thinly populated
  conditioning set inflates the degrees of freedom and the test stops rejecting
  anything
- Counting uses a flat contingency table while it stays small and a sparse one when it
  would not, so high cardinality conditioning sets stay affordable

#### Performance
- Observations are transposed into one slice per variable before learning starts.
  Structure learning reads every value of a variable thousands of times, and reaching
  those values through a map per row made hashing rather than counting the dominant
  cost: on a 80 variable network MMHC went from 3 min 53 s to 16 s

#### Utilities
- DataFrame for data handling
- CSV import/export
- Data type conversions

## [0.1.0] - 2025-10-23

### Added
- Initial project structure
- Basic graph algorithms
- Factor algebra
- Bayesian Network implementation
- Variable Elimination inference
- PC structure learning
- Example models
- Test suite
- Documentation

[Unreleased]: https://github.com/JohnPierman/bngo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JohnPierman/bngo/releases/tag/v0.1.0

