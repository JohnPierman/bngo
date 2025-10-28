package inference

import (
	"math"
	"testing"

	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/models"
)

func TestMixedDiscreteInference(t *testing.T) {
	// Create a discrete Bayesian Network: A -> B
	edges := [][2]string{
		{"A", "B"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("Failed to create network: %v", err)
	}

	// Add discrete CPDs
	cpdA, err := factors.NewTabularCPD("A", 2,
		[][]float64{{0.7, 0.3}},
		[]string{},
		map[string]int{},
	)
	if err != nil {
		t.Fatalf("Failed to create CPD: %v", err)
	}

	err = bn.AddCPD(cpdA)
	if err != nil {
		t.Fatalf("Failed to add CPD: %v", err)
	}

	cpdB, err := factors.NewTabularCPD("B", 2,
		[][]float64{
			{0.8, 0.2}, // A=0
			{0.3, 0.7}, // A=1
		},
		[]string{"A"},
		map[string]int{"A": 2},
	)
	if err != nil {
		t.Fatalf("Failed to create CPD: %v", err)
	}

	err = bn.AddCPD(cpdB)
	if err != nil {
		t.Fatalf("Failed to add CPD: %v", err)
	}

	// Create mixed inference engine
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("Failed to create mixed inference engine: %v", err)
	}

	// Query P(A | B=1)
	evidence := MixedEvidence{
		Discrete: map[string]int{"B": 1},
	}

	result, err := mve.Query([]string{"A"}, []string{}, evidence)
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}

	// Verify result
	if len(result.Values) != 2 {
		t.Errorf("Expected 2 values, got %d", len(result.Values))
	}

	// Check normalization
	sum := 0.0
	for _, v := range result.Values {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Result not normalized, sum=%.6f", sum)
	}
}

func TestMixedContinuousInference(t *testing.T) {
	// Create a continuous network: X -> Y
	edges := [][2]string{
		{"X", "Y"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("Failed to create network: %v", err)
	}

	// Add Gaussian CPDs
	cpdX, err := factors.NewLinearGaussianCPD("X", []string{}, 0.0, map[string]float64{}, 1.0)
	if err != nil {
		t.Fatalf("Failed to create CPD: %v", err)
	}

	err = bn.AddGaussianCPD(cpdX)
	if err != nil {
		t.Fatalf("Failed to add CPD: %v", err)
	}

	cpdY, err := factors.NewLinearGaussianCPD("Y", []string{"X"}, 0.0, map[string]float64{"X": 2.0}, 0.5)
	if err != nil {
		t.Fatalf("Failed to create CPD: %v", err)
	}

	err = bn.AddGaussianCPD(cpdY)
	if err != nil {
		t.Fatalf("Failed to add CPD: %v", err)
	}

	// Create mixed inference engine
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("Failed to create mixed inference engine: %v", err)
	}

	// Query P(Y | X=1.0)
	evidence := MixedEvidence{
		Continuous: map[string]float64{"X": 1.0},
	}

	result, err := mve.Query([]string{}, []string{"Y"}, evidence)
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}

	// Verify result has mean and covariance
	if result.Mean == nil || result.Covariance == nil {
		t.Errorf("Result missing mean or covariance")
	}

	// For Y = 2*X, given X=1, mean should be approximately 2
	if mean, ok := result.Mean["Y"]; ok {
		if math.Abs(mean-2.0) > 1.0 {
			t.Errorf("Expected mean ≈ 2.0, got %.4f", mean)
		}
	}
}

func TestMixedDiscreteQueryInMixedModel(t *testing.T) {
	// Create a mixed network: D -> X
	// D is discrete, X is continuous
	edges := [][2]string{
		{"D", "X"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("Failed to create network: %v", err)
	}

	// Add discrete CPD for D
	cpdD, err := factors.NewTabularCPD("D", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)
	if err != nil {
		t.Fatalf("Failed to create CPD: %v", err)
	}

	err = bn.AddCPD(cpdD)
	if err != nil {
		t.Fatalf("Failed to add CPD: %v", err)
	}

	// Add Gaussian CPD for X given D
	statesX := map[string]factors.GaussianParams{
		"0": {Mean: 0.0, Variance: 1.0},
		"1": {Mean: 5.0, Variance: 1.0},
	}
	cpdX, err := factors.NewDiscreteParentGaussianCPD("X", []string{"D"}, map[string]int{"D": 2}, statesX)
	if err != nil {
		t.Fatalf("Failed to create CPD: %v", err)
	}

	err = bn.AddGaussianCPD(cpdX)
	if err != nil {
		t.Fatalf("Failed to add CPD: %v", err)
	}

	// Create mixed inference engine
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("Failed to create mixed inference engine: %v", err)
	}

	// Query P(D) (no evidence)
	evidence := MixedEvidence{
		Discrete:   map[string]int{},
		Continuous: map[string]float64{},
	}

	result, err := mve.Query([]string{"D"}, []string{}, evidence)
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}

	// Verify result
	if len(result.Values) != 2 {
		t.Errorf("Expected 2 values, got %d", len(result.Values))
	}

	// Should be approximately [0.6, 0.4]
	if math.Abs(result.Values[0]-0.6) > 0.1 {
		t.Errorf("Expected P(D=0) ≈ 0.6, got %.4f", result.Values[0])
	}
	if math.Abs(result.Values[1]-0.4) > 0.1 {
		t.Errorf("Expected P(D=1) ≈ 0.4, got %.4f", result.Values[1])
	}
}

func TestMixedErrorHandling(t *testing.T) {
	edges := [][2]string{
		{"A", "B"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("Failed to create network: %v", err)
	}

	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5, 0.5}}, []string{}, map[string]int{})
	_ = bn.AddCPD(cpdA)

	cpdB, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.5, 0.5}, {0.5, 0.5}}, []string{"A"}, map[string]int{"A": 2})
	_ = bn.AddCPD(cpdB)

	mve, _ := NewMixedVariableElimination(bn)

	// Try to query continuous variable in discrete model
	evidence := MixedEvidence{Discrete: map[string]int{}}
	_, err = mve.Query([]string{"A"}, []string{"X"}, evidence)
	if err == nil {
		t.Error("Expected error when querying continuous variable in discrete model")
	}
}
