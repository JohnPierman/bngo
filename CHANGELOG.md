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

