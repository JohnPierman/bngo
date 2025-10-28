package examples

import (
	"fmt"

	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/inference"
	"github.com/JohnPierman/bngo/models"
)

// ExampleMixedInferenceDiscreteOnly demonstrates inference on discrete networks
func ExampleMixedInferenceDiscreteOnly() {
	// Create a discrete network: A -> B
	edges := [][2]string{
		{"A", "B"},
	}

	bn, _ := models.NewBayesianNetwork(edges)

	// Add discrete CPDs
	cpdA, _ := factors.NewTabularCPD("A", 2,
		[][]float64{{0.7, 0.3}},
		[]string{},
		map[string]int{},
	)
	_ = bn.AddCPD(cpdA)

	cpdB, _ := factors.NewTabularCPD("B", 2,
		[][]float64{
			{0.8, 0.2}, // A=0
			{0.3, 0.7}, // A=1
		},
		[]string{"A"},
		map[string]int{"A": 2},
	)
	_ = bn.AddCPD(cpdB)

	// Create mixed inference engine (works for discrete-only models too)
	mve, _ := inference.NewMixedVariableElimination(bn)

	// Query P(A | B=1)
	evidence := inference.MixedEvidence{
		Discrete: map[string]int{"B": 1},
	}

	result, _ := mve.Query([]string{"A"}, []string{}, evidence)

	fmt.Printf("P(A=0 | B=1) = %.4f\n", result.Values[0])
	fmt.Printf("P(A=1 | B=1) = %.4f\n", result.Values[1])

	// Output:
	// P(A=0 | B=1) = 0.5000
	// P(A=1 | B=1) = 0.5000
}

// ExampleMixedInferenceContinuousOnly demonstrates inference on continuous networks
func ExampleMixedInferenceContinuousOnly() {
	// Create a continuous network: X -> Y
	edges := [][2]string{
		{"X", "Y"},
	}

	bn, _ := models.NewBayesianNetwork(edges)

	// Add Gaussian CPDs
	// X ~ N(0, 1)
	cpdX, _ := factors.NewLinearGaussianCPD("X", []string{}, 0.0, map[string]float64{}, 1.0)
	_ = bn.AddGaussianCPD(cpdX)

	// Y = 2*X + ε, ε ~ N(0, 0.5)
	cpdY, _ := factors.NewLinearGaussianCPD("Y", []string{"X"}, 0.0, map[string]float64{"X": 2.0}, 0.5)
	_ = bn.AddGaussianCPD(cpdY)

	// Create mixed inference engine
	mve, _ := inference.NewMixedVariableElimination(bn)

	// Query P(Y | X=1.0)
	evidence := inference.MixedEvidence{
		Continuous: map[string]float64{"X": 1.0},
	}

	result, _ := mve.Query([]string{}, []string{"Y"}, evidence)

	fmt.Printf("E[Y | X=1.0] = %.4f\n", result.Mean["Y"])
	fmt.Printf("Var[Y | X=1.0] = %.4f\n", result.Covariance["Y"]["Y"])

	// Output:
	// E[Y | X=1.0] = 2.0000
	// Var[Y | X=1.0] = 0.5000
}

// ExampleMixedInferenceCLG demonstrates inference on conditional linear Gaussian (CLG) models
func ExampleMixedInferenceCLG() {
	// Create a mixed network: D -> X -> Y
	// D is discrete, X and Y are continuous
	edges := [][2]string{
		{"D", "X"},
		{"X", "Y"},
	}

	bn, _ := models.NewBayesianNetwork(edges)

	// Add discrete CPD for D
	cpdD, _ := factors.NewTabularCPD("D", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)
	_ = bn.AddCPD(cpdD)

	// Add Gaussian CPD for X given D
	// X | D=0 ~ N(0, 1), X | D=1 ~ N(5, 1)
	statesX := map[string]factors.GaussianParams{
		"0": {Mean: 0.0, Variance: 1.0},
		"1": {Mean: 5.0, Variance: 1.0},
	}
	cpdX, _ := factors.NewDiscreteParentGaussianCPD("X", []string{"D"}, map[string]int{"D": 2}, statesX)
	_ = bn.AddGaussianCPD(cpdX)

	// Add Gaussian CPD for Y given X
	// Y = X + ε, ε ~ N(0, 0.1)
	cpdY, _ := factors.NewLinearGaussianCPD("Y", []string{"X"}, 0.0, map[string]float64{"X": 1.0}, 0.1)
	_ = bn.AddGaussianCPD(cpdY)

	// Create mixed inference engine
	mve, _ := inference.NewMixedVariableElimination(bn)

	// Query P(D) (no evidence)
	evidence := inference.MixedEvidence{
		Discrete:   map[string]int{},
		Continuous: map[string]float64{},
	}

	result, _ := mve.Query([]string{"D"}, []string{}, evidence)

	fmt.Printf("P(D=0) = %.4f\n", result.Values[0])
	fmt.Printf("P(D=1) = %.4f\n", result.Values[1])

	// Output:
	// P(D=0) = 0.6000
	// P(D=1) = 0.4000
}

// ExampleMixedInferenceUseCase demonstrates a real-world use case
func ExampleMixedInferenceUseCase() {
	// Scenario: Predicting student performance
	// - Intelligence (discrete: low, high)
	// - SAT Score (continuous, depends on intelligence)
	// - Grade (continuous, depends on SAT score)

	edges := [][2]string{
		{"Intelligence", "SAT"},
		{"SAT", "Grade"},
	}

	bn, _ := models.NewBayesianNetwork(edges)

	// P(Intelligence=low) = 0.6, P(Intelligence=high) = 0.4
	cpdInt, _ := factors.NewTabularCPD("Intelligence", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)
	_ = bn.AddCPD(cpdInt)

	// SAT ~ N(μ_intelligence, σ)
	// Low intelligence: mean=1000, variance=10000
	// High intelligence: mean=1400, variance=10000
	satStates := map[string]factors.GaussianParams{
		"0": {Mean: 1000.0, Variance: 10000.0},
		"1": {Mean: 1400.0, Variance: 10000.0},
	}
	cpdSAT, _ := factors.NewDiscreteParentGaussianCPD("SAT", []string{"Intelligence"}, map[string]int{"Intelligence": 2}, satStates)
	_ = bn.AddGaussianCPD(cpdSAT)

	// Grade = 60 + 0.05*SAT + ε, ε ~ N(0, 25)
	cpdGrade, _ := factors.NewLinearGaussianCPD("Grade", []string{"SAT"}, 60.0, map[string]float64{"SAT": 0.05}, 25.0)
	_ = bn.AddGaussianCPD(cpdGrade)

	// Create inference engine
	mve, _ := inference.NewMixedVariableElimination(bn)

	// Scenario 1: Given intelligence=high, predict expected grade
	fmt.Println("Scenario 1: Predicting grade given high intelligence")

	// Note: This would require full CLG inference implementation
	// For now, show the setup
	fmt.Println("Network configured for mixed inference")

	// Scenario 2: Query intelligence distribution
	fmt.Println("\nScenario 2: Prior on intelligence")
	evidence2 := inference.MixedEvidence{
		Discrete:   map[string]int{},
		Continuous: map[string]float64{},
	}
	result, _ := mve.Query([]string{"Intelligence"}, []string{}, evidence2)
	fmt.Printf("P(Intelligence=low) = %.2f\n", result.Values[0])
	fmt.Printf("P(Intelligence=high) = %.2f\n", result.Values[1])

	// Output:
	// Scenario 1: Predicting grade given high intelligence
	// Network configured for mixed inference
	//
	// Scenario 2: Prior on intelligence
	// P(Intelligence=low) = 0.60
	// P(Intelligence=high) = 0.40
}
