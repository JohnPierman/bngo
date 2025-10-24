package factors

import (
	"fmt"
	"math"
	"math/rand"
)

// LinearGaussianCPD represents a conditional Gaussian distribution
// For continuous variable X with continuous/discrete parents Y:
// P(X | Y) = N(X; β₀ + Σᵢ βᵢYᵢ, σ²)
// where β₀ is the intercept, βᵢ are coefficients, and σ² is variance
type LinearGaussianCPD struct {
	Variable    string            // The continuous variable
	Parents     []string          // Parent variables
	ParentTypes map[string]string // "continuous" or "discrete"

	// For continuous parents: X = β₀ + Σᵢ βᵢYᵢ + ε, ε ~ N(0, σ²)
	Intercept    float64            // β₀
	Coefficients map[string]float64 // βᵢ for each parent
	Variance     float64            // σ²

	// For discrete parents: different Gaussian for each parent state combination
	// DiscreteStates[parent_config] = (mean, variance)
	DiscreteStates map[string]GaussianParams
	Cardinality    map[string]int // Cardinality of discrete parents
}

// GaussianParams holds mean and variance for a Gaussian
type GaussianParams struct {
	Mean     float64
	Variance float64
}

// NewLinearGaussianCPD creates a new linear Gaussian CPD
// For continuous parents only
func NewLinearGaussianCPD(variable string, parents []string, intercept float64,
	coefficients map[string]float64, variance float64) (*LinearGaussianCPD, error) {

	if variance <= 0 {
		return nil, fmt.Errorf("variance must be positive")
	}

	parentTypes := make(map[string]string)
	for _, p := range parents {
		parentTypes[p] = "continuous"
	}

	return &LinearGaussianCPD{
		Variable:     variable,
		Parents:      parents,
		ParentTypes:  parentTypes,
		Intercept:    intercept,
		Coefficients: coefficients,
		Variance:     variance,
	}, nil
}

// NewDiscreteParentGaussianCPD creates a Gaussian CPD with discrete parents
// Each parent state combination has different mean/variance
func NewDiscreteParentGaussianCPD(variable string, parents []string, cardinality map[string]int,
	states map[string]GaussianParams) (*LinearGaussianCPD, error) {

	// Validate all state combinations are present
	expectedStates := 1
	for _, p := range parents {
		expectedStates *= cardinality[p]
	}

	if len(states) != expectedStates {
		return nil, fmt.Errorf("expected %d state combinations, got %d", expectedStates, len(states))
	}

	parentTypes := make(map[string]string)
	for _, p := range parents {
		parentTypes[p] = "discrete"
	}

	return &LinearGaussianCPD{
		Variable:       variable,
		Parents:        parents,
		ParentTypes:    parentTypes,
		DiscreteStates: states,
		Cardinality:    cardinality,
	}, nil
}

// GetMean returns the conditional mean E[X | parents]
func (cpd *LinearGaussianCPD) GetMean(parentValues map[string]interface{}) (float64, error) {
	// Check if using discrete parents
	hasDiscrete := false
	for _, ptype := range cpd.ParentTypes {
		if ptype == "discrete" {
			hasDiscrete = true
			break
		}
	}

	if hasDiscrete {
		// All parents must be discrete
		stateKey := cpd.getStateKey(parentValues)
		params, ok := cpd.DiscreteStates[stateKey]
		if !ok {
			return 0, fmt.Errorf("no parameters for state %s", stateKey)
		}
		return params.Mean, nil
	}

	// Continuous parents: μ = β₀ + Σᵢ βᵢyᵢ
	mean := cpd.Intercept
	for _, parent := range cpd.Parents {
		val, ok := parentValues[parent]
		if !ok {
			return 0, fmt.Errorf("missing parent value for %s", parent)
		}

		floatVal, ok := val.(float64)
		if !ok {
			return 0, fmt.Errorf("parent %s value must be float64", parent)
		}

		coef := cpd.Coefficients[parent]
		mean += coef * floatVal
	}

	return mean, nil
}

// GetVariance returns the conditional variance Var[X | parents]
func (cpd *LinearGaussianCPD) GetVariance(parentValues map[string]interface{}) (float64, error) {
	// Check if using discrete parents
	hasDiscrete := false
	for _, ptype := range cpd.ParentTypes {
		if ptype == "discrete" {
			hasDiscrete = true
			break
		}
	}

	if hasDiscrete {
		stateKey := cpd.getStateKey(parentValues)
		params, ok := cpd.DiscreteStates[stateKey]
		if !ok {
			return 0, fmt.Errorf("no parameters for state %s", stateKey)
		}
		return params.Variance, nil
	}

	return cpd.Variance, nil
}

// Sample generates a sample from P(X | parents)
func (cpd *LinearGaussianCPD) Sample(parentValues map[string]interface{}, rng *rand.Rand) (float64, error) {
	mean, err := cpd.GetMean(parentValues)
	if err != nil {
		return 0, err
	}

	variance, err := cpd.GetVariance(parentValues)
	if err != nil {
		return 0, err
	}

	// Sample from N(mean, variance)
	stdDev := math.Sqrt(variance)
	return rng.NormFloat64()*stdDev + mean, nil
}

// PDF evaluates the probability density P(x | parents)
func (cpd *LinearGaussianCPD) PDF(x float64, parentValues map[string]interface{}) (float64, error) {
	mean, err := cpd.GetMean(parentValues)
	if err != nil {
		return 0, err
	}

	variance, err := cpd.GetVariance(parentValues)
	if err != nil {
		return 0, err
	}

	// Gaussian PDF: (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
	stdDev := math.Sqrt(variance)
	normConst := 1.0 / (stdDev * math.Sqrt(2*math.Pi))
	diff := x - mean
	exponent := -(diff * diff) / (2 * variance)

	return normConst * math.Exp(exponent), nil
}

// ToFactor converts the CPD to a Gaussian factor
// Only works for continuous parents
func (cpd *LinearGaussianCPD) ToFactor() (*GaussianFactor, error) {
	// Check if has discrete parents
	for _, ptype := range cpd.ParentTypes {
		if ptype == "discrete" {
			return nil, fmt.Errorf("cannot convert CPD with discrete parents to single Gaussian factor")
		}
	}

	if len(cpd.Parents) == 0 {
		// No parents: just a Gaussian over the variable
		mean := map[string]float64{cpd.Variable: cpd.Intercept}
		cov := map[string]map[string]float64{
			cpd.Variable: {cpd.Variable: cpd.Variance},
		}
		return NewGaussianFactor([]string{cpd.Variable}, mean, cov)
	}

	// With continuous parents: build joint Gaussian
	// This requires representing the conditional as a joint
	// For X = β₀ + Σᵢ βᵢYᵢ + ε with ε ~ N(0, σ²)
	// We need the joint P(X, Y₁, ..., Yₙ)
	// This is complex and requires knowing the parent marginals
	return nil, fmt.Errorf("converting linear Gaussian CPD to factor requires parent distributions")
}

// Copy creates a deep copy
func (cpd *LinearGaussianCPD) Copy() *LinearGaussianCPD {
	parentsCopy := make([]string, len(cpd.Parents))
	copy(parentsCopy, cpd.Parents)

	parentTypesCopy := make(map[string]string)
	for k, v := range cpd.ParentTypes {
		parentTypesCopy[k] = v
	}

	coeffCopy := make(map[string]float64)
	for k, v := range cpd.Coefficients {
		coeffCopy[k] = v
	}

	statesCopy := make(map[string]GaussianParams)
	for k, v := range cpd.DiscreteStates {
		statesCopy[k] = v
	}

	cardCopy := make(map[string]int)
	for k, v := range cpd.Cardinality {
		cardCopy[k] = v
	}

	return &LinearGaussianCPD{
		Variable:       cpd.Variable,
		Parents:        parentsCopy,
		ParentTypes:    parentTypesCopy,
		Intercept:      cpd.Intercept,
		Coefficients:   coeffCopy,
		Variance:       cpd.Variance,
		DiscreteStates: statesCopy,
		Cardinality:    cardCopy,
	}
}

// String returns a string representation
func (cpd *LinearGaussianCPD) String() string {
	return fmt.Sprintf("LinearGaussianCPD(%s | %v)", cpd.Variable, cpd.Parents)
}

// getStateKey creates a string key for discrete parent state combination
func (cpd *LinearGaussianCPD) getStateKey(parentValues map[string]interface{}) string {
	key := ""
	for i, parent := range cpd.Parents {
		if i > 0 {
			key += ","
		}
		val, ok := parentValues[parent]
		if !ok {
			return ""
		}
		key += fmt.Sprintf("%v", val)
	}
	return key
}
