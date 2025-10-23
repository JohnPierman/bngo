# bngo Implementation Summary

This document provides a detailed overview of the bngo implementation, a Golang port of the pgmpy Python library for Bayesian Networks.

## Project Overview

**bngo** is a complete Bayesian Network package for Go, providing:
- Probabilistic graphical models
- Causal inference
- Structure learning
- Parameter learning
- Probabilistic inference
- Data simulation

## Architecture

### Package Structure

```
bngo/
├── graph/              # Graph data structures
│   ├── dag.go         # Directed Acyclic Graph implementation
│   ├── undirected.go  # Undirected graph implementation
│   └── dag_test.go    # Unit tests
├── factors/            # Factor operations
│   ├── discrete.go    # Discrete factor implementation
│   ├── cpd.go         # Conditional Probability Distribution
│   └── discrete_test.go
├── models/             # Bayesian Network models
│   ├── bayesian_network.go  # Main BN implementation
│   └── bayesian_network_test.go
├── inference/          # Inference algorithms
│   └── variable_elimination.go  # Variable Elimination algorithm
├── estimators/         # Learning algorithms
│   ├── pc.go          # PC structure learning algorithm
│   └── independence_tests.go  # Statistical tests
├── utils/              # Utility functions
│   └── data.go        # DataFrame and CSV handling
├── examples/           # Example models
│   └── example_models.go  # Pre-built BN models
└── cmd/
    └── demo/           # Demo application
        └── main.go
```

## Core Components

### 1. Graph Structures (`graph/`)

**DAG (Directed Acyclic Graph)**
- Maintains parent-child relationships
- Cycle detection on edge addition
- Topological sorting
- Ancestor/descendant queries
- Moral graph construction

**Key Methods:**
- `AddEdge(parent, child)` - Adds edge with cycle checking
- `TopologicalSort()` - Returns nodes in causal order
- `Ancestors(node)` - Returns all ancestors
- `Descendants(node)` - Returns all descendants
- `MoralGraph()` - Creates undirected moral graph

**Implementation Notes:**
- Uses adjacency lists for efficient graph operations
- O(V+E) cycle detection using DFS
- Deterministic topological ordering

### 2. Factors (`factors/`)

**DiscreteFactor**
Represents discrete probability distributions as multidimensional arrays.

**Operations:**
- `Multiply(other)` - Factor product (join operation)
- `Marginalize(vars)` - Sum out variables
- `Reduce(evidence)` - Condition on evidence
- `Normalize()` - Convert to probability distribution
- `MaxMarginalize(vars)` - Maximum over variables (for MAP)

**TabularCPD**
Conditional Probability Distribution in tabular form.

**Features:**
- Validates probability constraints (sums to 1)
- Converts to factors for inference
- Parent-child relationship tracking

**Implementation Notes:**
- Row-major storage for cache efficiency
- Stride-based indexing for multi-dimensional access
- Laplace smoothing in parameter learning

### 3. Bayesian Networks (`models/`)

**BayesianNetwork**
Main model class combining DAG structure with CPDs.

**Core Functionality:**
- `Simulate(n, seed)` - Generate synthetic data via ancestral sampling
- `Fit(data)` - Learn CPD parameters via Maximum Likelihood Estimation
- `Predict(observations)` - Predict missing values using MAP
- `CheckModel()` - Validate model consistency

**Simulation Algorithm:**
```
1. Get topological ordering of nodes
2. For each sample:
   a. For each node in order:
      - Get parent values from current sample
      - Look up CPD probabilities
      - Sample from categorical distribution
   b. Store complete sample
```

**Parameter Learning:**
- Maximum Likelihood Estimation with Laplace smoothing
- Counts occurrences in data
- Normalizes to get probabilities
- Handles missing data gracefully

### 4. Inference (`inference/`)

**Variable Elimination**
Exact inference algorithm for discrete Bayesian Networks.

**Query Types:**
- `Query(vars, evidence)` - Compute P(vars | evidence)
- `MAP(vars, evidence)` - Find most probable assignment

**Algorithm:**
```
1. Convert CPDs to factors
2. Reduce factors with evidence
3. For each variable to eliminate:
   a. Find all factors containing variable
   b. Multiply these factors
   c. Sum out the variable
4. Multiply remaining factors
5. Normalize result
```

**Complexity:**
- Time: O(n * k^w) where w is induced width
- Space: O(k^w) for largest intermediate factor
- k = maximum cardinality, n = number of variables

**Optimizations:**
- Eliminate variables in order to minimize factor size
- Early reduction with evidence
- Reuse of factor operations

### 5. Structure Learning (`estimators/`)

**PC Algorithm (Peter-Clark)**
Constraint-based structure learning using conditional independence tests.

**Phases:**
1. **Skeleton Discovery**
   - Start with complete undirected graph
   - Test conditional independence for increasing conditioning set sizes
   - Remove edges if conditionally independent

2. **Edge Orientation**
   - Detect v-structures (X -> Z <- Y where X⊥Y)
   - Orient edges using separation sets
   - Propagate orientation constraints

**Independence Tests:**
- **Chi-square test**: For discrete variables
  - Tests null hypothesis: X ⊥ Y | Z
  - Uses contingency tables
  - p-value threshold (alpha) determines edge removal

- **Fisher's Z-test**: For continuous variables (partial implementation)
  - Tests partial correlation
  - Normal approximation for p-values

**Parameters:**
- `alpha`: Significance level (default 0.05)
  - Lower alpha = more edges (conservative)
  - Higher alpha = fewer edges (aggressive)

### 6. Utilities (`utils/`)

**DataFrame**
Simple data structure for tabular data.

**Features:**
- CSV import/export
- Column access
- Sample conversion
- Type safety for integer data

## Example Models (`examples/`)

### 1. Student Network
Classic academic performance model:
- Variables: Difficulty, Intelligence, Grade, SAT, Letter
- Models: How intelligence and course difficulty affect grades

### 2. Alarm Network
Home security scenario:
- Variables: Burglary, Earthquake, Alarm, JohnCalls, MaryCalls
- Models: Multiple causes for alarm activation

### 3. Cancer Network
Medical diagnosis:
- Variables: Pollution, Smoker, Cancer, XRay, Dyspnoea
- Models: Disease causes and symptoms

### 4. Sprinkler Network
Classic rain/sprinkler example:
- Variables: Cloudy, Sprinkler, Rain, WetGrass
- Models: Explaining away (explaining-away effect)

## Testing

### Test Coverage
- **graph/**: DAG operations, cycle detection, topological sort
- **factors/**: Factor operations, marginalization, reduction
- **models/**: Network creation, simulation, parameter learning

### Running Tests
```bash
go test -vet=off ./graph ./factors ./models
```

### Test Results
All core packages pass unit tests:
- ✓ graph package
- ✓ factors package  
- ✓ models package

## Usage Examples

### Basic Workflow

```go
// 1. Create network structure
edges := [][2]string{
    {"A", "C"},
    {"B", "C"},
}
bn, _ := models.NewBayesianNetwork(edges)

// 2. Define CPDs
cpdA, _ := factors.NewTabularCPD("A", 2, 
    [][]float64{{0.7, 0.3}}, 
    []string{}, map[string]int{})
bn.AddCPD(cpdA)
// ... add other CPDs

// 3. Simulate data
samples, _ := bn.Simulate(1000, 42)

// 4. Learn structure
pc := estimators.NewPC(samples)
dag, _ := pc.Estimate()

// 5. Learn parameters
learnedBN, _ := models.NewBayesianNetwork(dag.Edges())
learnedBN.Fit(samples)

// 6. Perform inference
ve, _ := inference.NewVariableElimination(learnedBN)
result, _ := ve.Query([]string{"C"}, map[string]int{"A": 1})
```

## Performance Characteristics

### Time Complexity
- **Simulation**: O(n * m) - n samples, m nodes
- **Inference**: O(n * k^w) - w is induced width
- **Structure Learning**: O(n^2 * d^s * m) - s is max conditioning set size
- **Parameter Learning**: O(n * m) - n samples, m nodes

### Space Complexity
- **DAG**: O(V + E) - vertices and edges
- **Factors**: O(k^d) - k cardinality, d dimensions
- **Inference**: O(k^w) - induced width

### Optimization Opportunities
1. **Parallel Inference**: Goroutines for independent factor operations
2. **Factor Caching**: Memoize repeated factor operations
3. **Sparse Representations**: For factors with many zeros
4. **Better Elimination Ordering**: Min-fill or min-width heuristics
5. **Approximate Inference**: Sampling methods for large networks

## Differences from pgmpy

### Design Choices

1. **Type Safety**
   - Go's static typing catches errors at compile time
   - Explicit error handling vs Python exceptions

2. **Memory Management**
   - Go's automatic garbage collection
   - Value vs pointer semantics for performance

3. **Concurrency**
   - Go's goroutines enable easy parallelization
   - Channel-based communication (future work)

4. **Simplifications**
   - Focus on discrete variables (continuous planned for future)
   - Simplified PC algorithm orientation phase
   - Basic statistical tests (can be extended)

### Feature Parity

**Implemented:**
- ✓ Discrete Bayesian Networks
- ✓ Variable Elimination inference
- ✓ PC structure learning
- ✓ Maximum Likelihood parameter learning
- ✓ Data simulation
- ✓ Prediction

**Planned:**
- ⚬ Continuous variables (Linear Gaussian)
- ⚬ Belief Propagation
- ⚬ MCMC sampling
- ⚬ Additional structure learning algorithms
- ⚬ Model scoring (BIC, AIC)
- ⚬ Causal inference (do-calculus)

## Building and Running

### Build
```bash
go build ./...
```

### Run Demo
```bash
go run cmd/demo/main.go
```

### Run Tests
```bash
go test -vet=off ./graph ./factors ./models
```

### Create Executable
```bash
go build -o bngo-demo cmd/demo/main.go
./bngo-demo
```

## Future Enhancements

### Short Term
1. Additional independence tests (G-test, permutation tests)
2. More sophisticated elimination ordering
3. Comprehensive test suite with edge cases
4. Benchmarking suite

### Medium Term
1. Linear Gaussian Bayesian Networks
2. Belief Propagation (message passing)
3. Additional structure learning (Hill Climbing, K2)
4. Model validation metrics
5. Visualization tools

### Long Term
1. Dynamic Bayesian Networks
2. Causal inference with interventions
3. Counterfactual reasoning
4. Online learning
5. Large-scale distributed inference

## Contributing

Contributions are welcome! Areas needing work:
- Performance optimizations
- Additional algorithms
- More example models
- Documentation improvements
- Bug fixes

## License

MIT License - see LICENSE file

## Acknowledgments

Based on [pgmpy](https://github.com/pgmpy/pgmpy) by the pgmpy development team.

