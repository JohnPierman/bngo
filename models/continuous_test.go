package models

import (
	"math"
	"testing"

	"github.com/JohnPierman/bngo/factors"
)

func TestContinuousNetwork(t *testing.T) {
	// Create a simple network: X -> Y
	// Where X and Y are continuous
	edges := [][2]string{
		{"X", "Y"},
	}

	bn, err := NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("Failed to create network: %v", err)
	}

	// Add Gaussian CPD for X (no parents)
	// X ~ N(0, 1)
	cpdX, err := factors.NewLinearGaussianCPD("X", []string{}, 0.0, map[string]float64{}, 1.0)
	if err != nil {
		t.Fatalf("Failed to create CPD for X: %v", err)
	}

	err = bn.AddGaussianCPD(cpdX)
	if err != nil {
		t.Fatalf("Failed to add CPD for X: %v", err)
	}

	// Add Gaussian CPD for Y given X
	// Y = 2*X + 1 + ε, ε ~ N(0, 0.5)
	cpdY, err := factors.NewLinearGaussianCPD("Y", []string{"X"}, 1.0, map[string]float64{"X": 2.0}, 0.5)
	if err != nil {
		t.Fatalf("Failed to create CPD for Y: %v", err)
	}

	err = bn.AddGaussianCPD(cpdY)
	if err != nil {
		t.Fatalf("Failed to add CPD for Y: %v", err)
	}

	// Check model
	err = bn.CheckModel()
	if err != nil {
		t.Fatalf("Model check failed: %v", err)
	}

	// Test variable types
	if !bn.IsContinuous("X") {
		t.Errorf("X should be continuous")
	}
	if !bn.IsContinuous("Y") {
		t.Errorf("Y should be continuous")
	}

	// Simulate some samples
	samples, err := bn.SimulateMixed(100, 42)
	if err != nil {
		t.Fatalf("Failed to simulate: %v", err)
	}

	if len(samples) != 100 {
		t.Errorf("Expected 100 samples, got %d", len(samples))
	}

	// Check that samples have expected relationship: Y ≈ 2*X + 1
	// Calculate mean absolute error
	mae := 0.0
	for _, sample := range samples {
		x := sample.Continuous["X"]
		y := sample.Continuous["Y"]
		expected := 2*x + 1
		mae += math.Abs(y - expected)
	}
	mae /= float64(len(samples))

	// MAE should be small (within noise level)
	if mae > 2.0 {
		t.Errorf("Mean absolute error too high: %.4f (expected < 2.0)", mae)
	}
}

func TestMixedNetwork(t *testing.T) {
	// Create a mixed network: D -> X -> Y
	// D is discrete (binary), X and Y are continuous
	edges := [][2]string{
		{"D", "X"},
		{"X", "Y"},
	}

	bn, err := NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("Failed to create network: %v", err)
	}

	// Add discrete CPD for D
	// P(D=0) = 0.6, P(D=1) = 0.4
	cpdD, err := factors.NewTabularCPD("D", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)
	if err != nil {
		t.Fatalf("Failed to create CPD for D: %v", err)
	}

	err = bn.AddCPD(cpdD)
	if err != nil {
		t.Fatalf("Failed to add CPD for D: %v", err)
	}

	// Add Gaussian CPD for X given D
	// X | D=0 ~ N(0, 1)
	// X | D=1 ~ N(5, 1)
	statesX := map[string]factors.GaussianParams{
		"0": {Mean: 0.0, Variance: 1.0},
		"1": {Mean: 5.0, Variance: 1.0},
	}
	cpdX, err := factors.NewDiscreteParentGaussianCPD("X", []string{"D"}, map[string]int{"D": 2}, statesX)
	if err != nil {
		t.Fatalf("Failed to create CPD for X: %v", err)
	}

	err = bn.AddGaussianCPD(cpdX)
	if err != nil {
		t.Fatalf("Failed to add CPD for X: %v", err)
	}

	// Add Gaussian CPD for Y given X
	// Y = X + ε, ε ~ N(0, 0.1)
	cpdY, err := factors.NewLinearGaussianCPD("Y", []string{"X"}, 0.0, map[string]float64{"X": 1.0}, 0.1)
	if err != nil {
		t.Fatalf("Failed to create CPD for Y: %v", err)
	}

	err = bn.AddGaussianCPD(cpdY)
	if err != nil {
		t.Fatalf("Failed to add CPD for Y: %v", err)
	}

	// Check model
	err = bn.CheckModel()
	if err != nil {
		t.Fatalf("Model check failed: %v", err)
	}

	// Test variable types
	if !bn.IsDiscrete("D") {
		t.Errorf("D should be discrete")
	}
	if !bn.IsContinuous("X") {
		t.Errorf("X should be continuous")
	}
	if !bn.IsContinuous("Y") {
		t.Errorf("Y should be continuous")
	}

	// Simulate samples
	samples, err := bn.SimulateMixed(200, 123)
	if err != nil {
		t.Fatalf("Failed to simulate: %v", err)
	}

	if len(samples) != 200 {
		t.Errorf("Expected 200 samples, got %d", len(samples))
	}

	// Check that discrete variable has correct distribution
	countD0 := 0
	for _, sample := range samples {
		if sample.Discrete["D"] == 0 {
			countD0++
		}
	}
	propD0 := float64(countD0) / float64(len(samples))

	// Should be approximately 0.6
	if math.Abs(propD0-0.6) > 0.15 {
		t.Errorf("Proportion of D=0 is %.4f, expected ~0.6", propD0)
	}

	// Check that X has different means for D=0 vs D=1
	var xGivenD0, xGivenD1 []float64
	for _, sample := range samples {
		if sample.Discrete["D"] == 0 {
			xGivenD0 = append(xGivenD0, sample.Continuous["X"])
		} else {
			xGivenD1 = append(xGivenD1, sample.Continuous["X"])
		}
	}

	meanX0 := mean(xGivenD0)
	meanX1 := mean(xGivenD1)

	// Should be approximately 0 and 5
	if math.Abs(meanX0-0.0) > 1.0 {
		t.Errorf("Mean of X|D=0 is %.4f, expected ~0.0", meanX0)
	}
	if math.Abs(meanX1-5.0) > 1.0 {
		t.Errorf("Mean of X|D=1 is %.4f, expected ~5.0", meanX1)
	}
}

func TestParameterLearning(t *testing.T) {
	// Create a simple linear network: X -> Y
	edges := [][2]string{
		{"X", "Y"},
	}

	bn, err := NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("Failed to create network: %v", err)
	}

	// Generate synthetic data: Y = 3*X + 2 + noise
	nSamples := 500
	samples := make([]Sample, nSamples)
	for i := 0; i < nSamples; i++ {
		x := float64(i)/100.0 - 2.5        // X ranges from -2.5 to 2.5
		y := 3*x + 2 + 0.1*float64(i%10-5) // Small noise

		samples[i] = Sample{
			Discrete:   make(map[string]int),
			Continuous: map[string]float64{"X": x, "Y": y},
		}
	}

	// Learn parameters
	err = bn.FitMixed(samples)
	if err != nil {
		t.Fatalf("Failed to fit model: %v", err)
	}

	// Check learned parameters
	cpdY, err := bn.GetGaussianCPD("Y")
	if err != nil {
		t.Fatalf("Failed to get CPD for Y: %v", err)
	}

	// Intercept should be ~2.0
	if math.Abs(cpdY.Intercept-2.0) > 0.5 {
		t.Errorf("Learned intercept is %.4f, expected ~2.0", cpdY.Intercept)
	}

	// Coefficient for X should be ~3.0
	if math.Abs(cpdY.Coefficients["X"]-3.0) > 0.5 {
		t.Errorf("Learned coefficient is %.4f, expected ~3.0", cpdY.Coefficients["X"])
	}
}

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}
