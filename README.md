# bngo

**bngo** is a comprehensive Bayesian Network package for Go, inspired by [pgmpy](https://github.com/pgmpy/pgmpy). It provides tools for causal inference, probabilistic modeling, and machine learning with Bayesian Networks.

[![Go Version](https://img.shields.io/badge/go-%3E%3D1.16-blue.svg)](https://golang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

| Feature | Description |
|---------|-------------|
| **Structure Learning** | Learn the model structure from data: PC (constraint-based), Hill Climbing (score-based) and MMHC (hybrid, for large networks) |
| **Network Scores** | Score a structure against data with BIC, AIC, BDeu or K2 |
| **Parameter Learning** | Estimate model parameters (CPDs) from observed data using Maximum Likelihood Estimation |
| **Probabilistic Inference** | Compute posterior distributions using Variable Elimination algorithm |
| **Simulations** | Generate synthetic data from Bayesian Networks |
| **Prediction** | Predict missing values in partial observations |
| **Causal Models** | Support for directed acyclic graphs (DAGs) and causal reasoning |

## Installation

```bash
go get github.com/JohnPierman/bngo
```

## Quick Start

### Creating a Bayesian Network

```go
package main

import (
    "fmt"
    "github.com/JohnPierman/bngo/models"
    "github.com/JohnPierman/bngo/factors"
)

func main() {
    // Define the structure
    edges := [][2]string{
        {"Cloudy", "Sprinkler"},
        {"Cloudy", "Rain"},
        {"Sprinkler", "WetGrass"},
        {"Rain", "WetGrass"},
    }
    
    bn, _ := models.NewBayesianNetwork(edges)
    
    // Define CPDs (Conditional Probability Distributions)
    cpdCloudy, _ := factors.NewTabularCPD("Cloudy", 2,
        [][]float64{{0.5, 0.5}},
        []string{},
        map[string]int{},
    )
    
    cpdSprinkler, _ := factors.NewTabularCPD("Sprinkler", 2,
        [][]float64{
            {0.5, 0.5},  // Cloudy=0
            {0.9, 0.1},  // Cloudy=1
        },
        []string{"Cloudy"},
        map[string]int{"Cloudy": 2},
    )
    
    bn.AddCPD(cpdCloudy)
    bn.AddCPD(cpdSprinkler)
    // ... add other CPDs
    
    fmt.Printf("Network: %v\n", bn.Edges())
}
```

### Simulating Data

```go
// Simulate 1000 samples from the network
samples, err := bn.Simulate(1000, 42)
if err != nil {
    panic(err)
}

fmt.Printf("Generated %d samples\n", len(samples))
fmt.Printf("First sample: %v\n", samples[0])
```

### Probabilistic Inference

```go
import "github.com/JohnPierman/bngo/inference"

// Create inference engine
ve, _ := inference.NewVariableElimination(bn)

// Query: P(Rain | WetGrass=1)
evidence := map[string]int{"WetGrass": 1}
result, _ := ve.Query([]string{"Rain"}, evidence)

fmt.Printf("P(Rain=0 | WetGrass=1) = %.4f\n", result.Values[0])
fmt.Printf("P(Rain=1 | WetGrass=1) = %.4f\n", result.Values[1])
```

### Mixed Network Inference

```go
import "github.com/JohnPierman/bngo/inference"

// Create mixed inference engine (works with discrete, continuous, or mixed networks)
mve, _ := inference.NewMixedVariableElimination(bn)

// Query continuous variable given discrete evidence
result, _ := mve.QueryContinuous(
    []string{"Temperature"},
    inference.MixedEvidence{Discrete: map[string]int{"Season": 2}}, // Summer
)
fmt.Printf("E[Temperature | Summer] = %.2f\n", result.Mean["Temperature"])
fmt.Printf("Var[Temperature | Summer] = %.2f\n", result.Covariance["Temperature"]["Temperature"])

// Query discrete variable given continuous evidence (Bayesian updating)
posterior, _ := mve.QueryDiscrete(
    []string{"Season"},
    inference.MixedEvidence{Continuous: map[string]float64{"Temperature": 85.0}},
)
fmt.Printf("P(Season | Temp=85) = %v\n", posterior.Values)
```

### Structure Learning

```go
import "github.com/JohnPierman/bngo/estimators"

// Learn structure from data using PC algorithm
pc := estimators.NewPC(samples)
pc.SetAlpha(0.05) // Significance level

learnedDAG, _ := pc.Estimate()
fmt.Printf("Learned structure: %v\n", learnedDAG.Edges())
```

### Parameter Learning

```go
// Create network with known structure
bn, _ := models.NewBayesianNetwork(edges)

// Learn parameters from data
err := bn.Fit(samples)
if err != nil {
    panic(err)
}

// Get learned CPD
cpd, _ := bn.GetCPD("Rain")
fmt.Printf("Learned CPD: %v\n", cpd)
```

### Prediction

```go
// Create partial observations (missing values)
testSamples := []map[string]int{
    {"Cloudy": 1, "Sprinkler": 0},
    {"Rain": 1, "WetGrass": 1},
}

// Predict missing values
predictions, _ := bn.Predict(testSamples)

fmt.Printf("Predicted Rain: %v\n", predictions["Rain"])
fmt.Printf("Predicted Sprinkler: %v\n", predictions["Sprinkler"])
```

### Score-Based Structure Learning

Score-based search asks which structure explains the data best, rather than which
independencies the data shows. It scales far better than PC and is the route to take
on a large network:

```go
import "github.com/JohnPierman/bngo/estimators"

// Greedy hill climbing over edge additions, deletions and reversals, scored by BIC
search := estimators.NewHillClimb(samples, nil)
search.SetMaxIndegree(4) // optional: bound the size of the CPDs

learnedDAG, _ := search.Estimate()

// Max-Min Hill Climbing: narrow the candidate parents first, then search
mmhc := estimators.NewMMHC(samples, nil)
mmhc.SetAlpha(0.05)

learnedDAG, _ = mmhc.Estimate()

// The skeleton MMPC found is useful on its own
skeleton := mmhc.LearnSkeleton()
fmt.Printf("Candidate edges: %v\n", skeleton.Edges())
```

Any of the four scores can drive either search, and a score can also be used on its
own to compare two structures you already have:

```go
bdeu, _ := estimators.NewBDeu(samples, nil, 10) // equivalent sample size 10
_ = search.SetScore(bdeu)

bic := estimators.NewBIC(samples, nil)
total, _ := estimators.ScoreDAG(learnedDAG, bic)
fmt.Printf("BIC of the learned network: %.2f\n", total)
```

**Which algorithm to use.** Measured on a sparse synthetic network where each variable
depends on the two before it, 2000 rows, binary variables:

| Variables | PC | Hill Climbing | MMHC |
|----------:|---:|--------------:|-----:|
| 25  | 2 m 27 s | 0.19 s | 1.3 s |
| 40  | -        | 0.78 s | 4.3 s |
| 80  | -        | 8.6 s  | 16.4 s |
| 120 | -        | 41.1 s | 29.2 s |
| 160 | -        | 2 m 10 s | 42.7 s |
| 240 | -        | 9 m 53 s | 1 m 8 s |

Hill climbing is the faster choice up to about a hundred variables, and the more
accurate one throughout: on these runs it recovered 94-98% of the true edges against
85-91% for MMHC. Past roughly a hundred variables the O(n^2) candidate changes of an
unrestricted step start to dominate, MMHC overtakes it, and the gap keeps widening:
by 240 variables MMHC is nearly nine times faster.
MMHC is also the more cautious of the two, reporting essentially no edge that is not
in the true network. PC was left out above 25 variables because it was already slower
than either at that size, and less accurate.

## Package Structure

```
bngo/
├── graph/              # Graph data structures (DAG, undirected graphs)
│   ├── dag.go
│   └── undirected.go
├── factors/            # Factor and CPD implementations
│   ├── discrete.go
│   └── cpd.go
├── models/             # Bayesian Network models
│   └── bayesian_network.go
├── inference/          # Inference algorithms
│   └── variable_elimination.go
├── estimators/         # Structure and parameter learning
│   ├── pc.go
│   ├── hill_climb.go
│   ├── mmhc.go
│   ├── score.go
│   ├── columns.go
│   ├── g_square.go
│   └── independence_tests.go
├── utils/              # Utility functions
│   └── data.go
└── examples/           # Example models and usage
    ├── example_models.go
    └── main.go
```

## Examples

The `examples/` directory contains several pre-built Bayesian Network models:

- **Student Network**: Classic student intelligence/grade model
- **Alarm Network**: Burglary/earthquake alarm system
- **Cancer Network**: Medical diagnosis network
- **Sprinkler Network**: Classic rain/sprinkler example

### Running Examples

```bash
cd examples
go run main.go example_models.go
```

## Core Components

### Graph Structures

**DAG (Directed Acyclic Graph)**
- Add/remove nodes and edges
- Check for cycles
- Topological sorting
- Find ancestors/descendants
- Create moral graphs

```go
import "github.com/JohnPierman/bngo/graph"

dag := graph.NewDAG()
dag.AddEdge("A", "B")
dag.AddEdge("B", "C")

ancestors := dag.Ancestors("C")  // ["A", "B"]
topOrder, _ := dag.TopologicalSort()
```

### Factors

**Discrete Factors**
- Factor multiplication
- Marginalization
- Reduction (evidence)
- Normalization
- Max-marginalization (for MAP queries)

```go
import "github.com/JohnPierman/bngo/factors"

factor1, _ := factors.NewDiscreteFactor(
    []string{"A", "B"},
    map[string]int{"A": 2, "B": 2},
    []float64{0.3, 0.7, 0.4, 0.6},
)

// Marginalize out variable B
marginalized, _ := factor1.Marginalize([]string{"B"})

// Reduce with evidence
evidence := map[string]int{"A": 1}
reduced, _ := factor1.Reduce(evidence)
```

**Tabular CPD**
- Conditional probability distributions in tabular form
- Convert to factors for inference
- Query specific probability values

### Models

**Bayesian Network**
- Define structure with edges
- Add CPDs for each node
- Validate model consistency
- Simulate data
- Learn parameters from data
- Make predictions

### Inference

**Variable Elimination**
- Exact inference for discrete Bayesian Networks
- Posterior probability queries
- MAP (Maximum A Posteriori) queries
- Evidence handling

**Mixed Variable Elimination**
- Exact inference for mixed discrete/continuous networks (CLG models)
- Query continuous variables given discrete and/or continuous evidence
- Query discrete variables given continuous evidence (Bayesian updating)
- Builds joint Gaussian distributions conditioned on discrete configurations
- Moment-matched mixture approximation when hidden discrete variables exist

### Structure Learning

**PC Algorithm**
- Constraint-based structure learning
- Chi-square test for conditional independence
- Learns undirected skeleton
- Orients edges based on v-structures
- Configurable significance level (alpha)

**Hill Climbing**
- Score-based greedy search over edge additions, deletions and reversals
- Prices one edge change by rescoring only the families it touches
- Optional cap on the number of parents, and on the candidate parents of each variable
- Terminates by construction: every accepted change strictly increases the score

**MMHC (Max-Min Hill Climbing)**
- Hybrid: MMPC narrows the candidate parents by local independence tests, then greedy
  search picks and orients the edges among them
- Restricting the search is what makes it suited to large networks
- The MMPC skeleton is available on its own via `LearnSkeleton`

**Network Scores**
- BIC and AIC: penalised log likelihood
- BDeu and K2: marginal likelihood under a Dirichlet prior
- Every score is decomposable and caches the families it has already priced

### Data Utilities

**DataFrame**
- Load/save CSV files
- Convert between samples and DataFrame format
- Column access and manipulation

```go
import "github.com/JohnPierman/bngo/utils"

// Load data from CSV
df, _ := utils.LoadCSV("data.csv")

// Save data to CSV
df.SaveCSV("output.csv")

// Convert to samples
samples := df.ToSamples()
```

## Advanced Usage

### Custom Model Creation

```go
// Create a complex model
edges := [][2]string{
    {"A", "C"},
    {"B", "C"},
    {"C", "D"},
    {"C", "E"},
}

bn, _ := models.NewBayesianNetwork(edges)

// Add CPDs for all nodes
// ... (define each CPD)

// Validate the model
err := bn.CheckModel()
if err != nil {
    panic(err)
}
```

### Batch Inference

```go
ve, _ := inference.NewVariableElimination(bn)

// Query multiple variables
result, _ := ve.Query([]string{"A", "B"}, evidence)

// Joint probability table
fmt.Printf("Joint distribution: %v\n", result.Values)
```

### Structure Learning Pipeline

```go
// 1. Load or generate data
bn, _ := examples.GetAlarmModel()
data, _ := bn.Simulate(1000, 123)

// 2. Learn structure
pc := estimators.NewPC(data)
dag, _ := pc.Estimate()

// 3. Create new network with learned structure
learnedBN, _ := models.NewBayesianNetwork(dag.Edges())

// 4. Learn parameters
learnedBN.Fit(data)

// 5. Use learned model
predictions, _ := learnedBN.Predict(testData)
```

## Comparison with pgmpy

bngo provides similar functionality to pgmpy but with Go's advantages:

- **Performance**: Compiled language with better performance
- **Concurrency**: Native goroutine support for parallel operations
- **Type Safety**: Static typing catches errors at compile time
- **Deployment**: Single binary with no dependencies
- **Memory Efficient**: Better memory management for large networks

## Implementation Notes

### Variable Types

bngo supports both **discrete** and **continuous** variables:

- **Discrete Variables**: Finite cardinality, represented as integers (0, 1, 2, ...)
- **Continuous Variables**: Real-valued, modeled using Linear Gaussian distributions


### Independence Tests

- **Chi-square test**: For discrete data (structure learning)
- **G-squared test**: Likelihood ratio test for discrete data, with degrees of
  freedom adjusted to the states actually observed in each stratum
- **Pearson correlation**: For continuous data (partial implementation)
- **Fisher's Z-test**: For correlation-based tests

### Inference Algorithms

- **Variable Elimination**: Exact inference for discrete networks by eliminating variables one by one
- **Mixed Variable Elimination**: Exact inference for CLG (Conditional Linear Gaussian) models
  - Enumerates discrete configurations, builds joint Gaussians, and computes weighted results
  - Supports continuous queries with discrete/continuous evidence
  - Supports discrete queries with continuous evidence via likelihood weighting
- Elimination order uses a simple heuristic (can be improved for better performance)

### Structure Learning

- **PC Algorithm**: Constraint-based approach
- Starts with complete graph and removes edges based on conditional independence
- Orients edges using v-structures and propagation rules
- **Hill Climbing**: Score-based greedy search
- **MMHC**: Hybrid, restricting a greedy search to a skeleton learned from local tests

Two properties of score-based search are worth knowing. A score can only tell apart
structures that encode different independencies, so an edge direction is recovered
only as far as the data determines it. And greedy search returns the first network
that no single change improves, which need not be the best one: reaching a collision
of two parents on one child can need two edges reversed at once, and reversing either
alone does not improve the score. The undirected edges are the more dependable part
of the answer.

Observations are transposed into one slice per variable before learning starts.
Structure learning reads every value of a variable thousands of times, and reaching
those values through a map per row makes hashing rather than counting the dominant
cost on a large network. Sufficient statistics are kept only for the parent
configurations that actually occur, so a wide parent set costs memory in proportion
to the sample rather than to the product of the cardinalities.

## Performance Tips

1. **Structure Learning**: On a network of more than about a hundred variables prefer
   `MMHC` over `HillClimbSearch`, and both over `PC`. Capping the number of parents
   with `SetMaxIndegree` bounds the size of the CPDs and the cost of scoring them
2. **Inference**: The order of variable elimination affects performance - smaller elimination cliques are better
3. **Simulation**: Use appropriate random seeds for reproducible results
4. **Memory**: For very large networks, consider streaming data processing

## Testing

Run tests with:

```bash
go test ./...
```

Run examples:

```bash
cd examples
go run main.go example_models.go
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

Areas for contribution:
- Additional inference algorithms (Belief Propagation, Sampling methods)
- Continuous variable support (Linear Gaussian models)
- More structure learning algorithms (Hill Climbing, K2)
- Performance optimizations
- Additional example models
- Documentation improvements

## Roadmap

- [x] Continuous variable support (Linear Gaussian models)
- [x] Mixed discrete/continuous networks
- [x] Exact inference for mixed networks
- [ ] Belief Propagation inference
- [ ] MCMC sampling methods
- [x] Additional structure learning algorithms (Hill Climbing, MMHC)
- [x] Model scoring metrics (BIC, AIC, BDeu, K2)
- [ ] Causal inference (do-calculus, interventions)
- [ ] Model visualization
- [ ] Parallel inference for large networks
- [ ] Approximate inference algorithms

## License

MIT License - see [LICENSE](LICENSE) file for details

## Acknowledgments

This project is inspired by and based on the excellent [pgmpy](https://github.com/pgmpy/pgmpy) library for Python. We thank the pgmpy team for their contributions to probabilistic graphical models.

## References

- Koller, D., & Friedman, N. (2009). Probabilistic Graphical Models: Principles and Techniques. MIT Press.
- Pearl, J. (2009). Causality: Models, Reasoning and Inference. Cambridge University Press.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search. MIT Press.

## Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is a Go implementation of Bayesian Networks for causal inference and probabilistic modeling. For the Python version, see [pgmpy](https://github.com/pgmpy/pgmpy).
