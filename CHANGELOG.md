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
- Support for non-numerical fields: categorical and binary variables carry labels
- `categorical` package with the `StateNames` and `Codebook` value objects
- Label valued CSV loading with configurable missing value markers
- Label aware fitting, simulation, prediction and inference on `BayesianNetwork`
- Labelled query results via `inference.QueryLabeled` and `inference.MAPLabeled`

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

#### Utilities
- DataFrame for data handling
- CSV import/export
- Data type conversions

#### Categorical and Binary Fields
- `categorical.StateNames`: an ordered, immutable set of labels for one field,
  validating that labels are non-empty and unique
- `categorical.Codebook`: one `StateNames` per variable, encoding and decoding
  whole rows between labels and integer states
- Deterministic label ordering, so the same data always encodes to the same
  integers: recognised binary pairs (`no`/`yes`, `false`/`true`, `0`/`1`,
  `off`/`on`, `absent`/`present`, ...) are ordered negative first so the positive
  state is state 1; all numeric labels sort numerically (`2` before `10`);
  anything else sorts lexicographically
- An empty field counts as unobserved rather than as a state of its own, and
  `NA` and `?` stay ordinary labels unless configured as missing markers
- `utils.CategoricalFrame`, `utils.LoadCategoricalCSV` and
  `utils.LoadCategoricalCSVWithOptions` for label valued CSV files
- `BayesianNetwork.DeclareStates`, `AddCategoricalCPD`, `FitCategorical`,
  `SimulateCategorical`, `PredictCategorical` and `ValidateCodebook`
- `AddCategoricalCPD` takes cardinalities from the declared states, so defining a
  CPD no longer needs a hand written cardinality map
- Parameter learning honours declared cardinalities, so a state that never occurs
  in the sample keeps its column in the CPD instead of being dropped
- `inference.LabeledDistribution` with `MostLikely`, `Probability` and `String`
- `examples.GetWeatherModel` and `examples.DemonstrateCategoricalNetwork`, a
  labelled network mixing a three state categorical field with two binary fields

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

