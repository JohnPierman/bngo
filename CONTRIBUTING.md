# Contributing to bngo

Thank you for your interest in contributing to bngo! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Guidelines](#coding-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/bngo.git
   cd bngo
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/JohnPierman/bngo.git
   ```

## Development Setup

### Prerequisites

- Go 1.21 or higher
- Git

### Installing Dependencies

```bash
go mod download
```

### Building

```bash
go build ./...
```

### Running Tests

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Run specific package tests
go test ./graph
go test ./factors
go test ./models
```

### Running Examples

```bash
go run cmd/demo/main.go
```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Go version and OS
- Any relevant code samples or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- Detailed description of the proposed feature
- Any relevant examples or use cases
- Why this enhancement would be useful

### Pull Requests

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes, following the coding guidelines

3. Add or update tests as needed

4. Ensure all tests pass:
   ```bash
   go test ./...
   ```

5. Update documentation as needed

6. Commit your changes with clear commit messages:
   ```bash
   git commit -m "Add feature: description of feature"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/my-new-feature
   ```

8. Open a Pull Request

## Coding Guidelines

### Go Style

- Follow the [Effective Go](https://golang.org/doc/effective_go.html) guidelines
- Use `gofmt` to format your code
- Use `golint` to check for style issues
- Follow Go naming conventions

### Code Organization

- Keep functions focused and single-purpose
- Limit function length to ~50 lines when possible
- Use meaningful variable and function names
- Add comments for exported functions, types, and packages

### Package Documentation

All exported identifiers should have documentation comments:

```go
// NewBayesianNetwork creates a new Bayesian Network with the given edges.
// It returns an error if the edges would create a cycle.
func NewBayesianNetwork(edges [][2]string) (*BayesianNetwork, error) {
    // ...
}
```

### Error Handling

- Return errors rather than panicking
- Provide context in error messages
- Use `fmt.Errorf` with `%w` for error wrapping when appropriate

```go
if err != nil {
    return nil, fmt.Errorf("failed to create factor: %w", err)
}
```

## Testing Guidelines

### Test Organization

- Place tests in the same package as the code being tested
- Use table-driven tests when testing multiple scenarios
- Name test functions descriptively: `TestFunctionName_Scenario`

### Test Coverage

- Aim for >80% test coverage
- Test edge cases and error conditions
- Include benchmarks for performance-critical code

### Example Test

```go
func TestDAGAddEdge(t *testing.T) {
    tests := []struct {
        name    string
        parent  string
        child   string
        wantErr bool
    }{
        {"valid edge", "A", "B", false},
        {"creates cycle", "B", "A", true},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            dag := NewDAG()
            dag.AddEdge("A", "B")
            err := dag.AddEdge(tt.parent, tt.child)
            if (err != nil) != tt.wantErr {
                t.Errorf("AddEdge() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

### Benchmarks

Add benchmarks for performance-critical operations:

```go
func BenchmarkFactorMultiply(b *testing.B) {
    f1, _ := NewDiscreteFactor(...)
    f2, _ := NewDiscreteFactor(...)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        f1.Multiply(f2)
    }
}
```

## Documentation Guidelines

### README

- Keep the README up-to-date
- Include clear examples
- Document all major features

### Code Comments

- Document all exported functions, types, and constants
- Use complete sentences in comments
- Explain "why" not just "what"

### Examples

Add example functions for godoc:

```go
func ExampleBayesianNetwork_Simulate() {
    bn, _ := NewBayesianNetwork([][2]string{{"A", "B"}})
    // ... add CPDs ...
    samples, _ := bn.Simulate(100, 42)
    fmt.Printf("Generated %d samples\n", len(samples))
    // Output: Generated 100 samples
}
```

## Submitting Changes

### Commit Messages

Use clear and meaningful commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests when relevant

Example:
```
Add belief propagation inference algorithm

Implements message passing for exact inference in tree-structured
Bayesian Networks. Includes tests and benchmarks.

Closes #42
```

### Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the IMPLEMENTATION.md with technical details if needed
3. Add tests for new functionality
4. Ensure the test suite passes
5. Request review from maintainers

### Review Process

- All submissions require review
- Maintainers may request changes
- Once approved, a maintainer will merge your PR

## Areas for Contribution

We welcome contributions in these areas:

### High Priority

- Additional inference algorithms (Belief Propagation, Sampling)
- Continuous variable support (Linear Gaussian)
- More structure learning algorithms (Hill Climbing, K2)
- Performance optimizations
- Additional example models

### Medium Priority

- Model validation metrics (BIC, AIC, BDeu)
- Visualization tools
- Additional independence tests
- Better elimination ordering heuristics

### Documentation

- Tutorial notebooks
- More usage examples
- API documentation improvements
- Video tutorials

### Testing

- Increase test coverage
- Add integration tests
- Performance benchmarks
- Stress tests for large networks

## Questions?

Feel free to open an issue for any questions about contributing!

## License

By contributing to bngo, you agree that your contributions will be licensed under the MIT License.

