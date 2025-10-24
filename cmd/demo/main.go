// Example usage of the bngo package
package main

import (
	"fmt"

	"github.com/JohnPierman/bngo/estimators"
	"github.com/JohnPierman/bngo/examples"
	"github.com/JohnPierman/bngo/inference"
	"github.com/JohnPierman/bngo/models"
	"github.com/JohnPierman/bngo/utils"
)

func main() {
	fmt.Println("=== bngo: Bayesian Networks in Go ===")

	// Example 1: Create and use the Student model
	fmt.Println("Example 1: Student Bayesian Network")
	studentExample()
	fmt.Println()

	// Example 2: Create and use the Alarm model
	fmt.Println("Example 2: Alarm Bayesian Network")
	alarmExample()
	fmt.Println()

	// Example 3: Structure learning with PC algorithm
	fmt.Println("Example 3: Structure Learning with PC Algorithm")
	structureLearningExample()
	fmt.Println()
}

func studentExample() {
	// Load the Student model
	bn, err := examples.GetStudentModel()
	if err != nil {
		fmt.Printf("Error creating model: %v\n", err)
		return
	}

	fmt.Printf("Model nodes: %v\n", bn.Nodes())
	fmt.Printf("Model edges: %v\n", bn.Edges())

	// Simulate data
	samples, err := bn.Simulate(100, 42)
	if err != nil {
		fmt.Printf("Error simulating: %v\n", err)
		return
	}
	fmt.Printf("Simulated %d samples\n", len(samples))
	fmt.Printf("First sample: %v\n", samples[0])

	// Perform inference
	ve, err := inference.NewVariableElimination(bn)
	if err != nil {
		fmt.Printf("Error creating inference: %v\n", err)
		return
	}

	// Query: P(Letter | Intelligence=1)
	evidence := map[string]int{"Intelligence": 1}
	result, err := ve.Query([]string{"Letter"}, evidence)
	if err != nil {
		fmt.Printf("Error in query: %v\n", err)
		return
	}

	fmt.Printf("\nP(Letter | Intelligence=high):\n")
	fmt.Printf("  P(Letter=weak) = %.4f\n", result.Values[0])
	fmt.Printf("  P(Letter=strong) = %.4f\n", result.Values[1])
}

func alarmExample() {
	// Load the Alarm model
	bn, err := examples.GetAlarmModel()
	if err != nil {
		fmt.Printf("Error creating model: %v\n", err)
		return
	}

	fmt.Printf("Model nodes: %v\n", bn.Nodes())
	fmt.Printf("Model edges: %v\n", bn.Edges())

	// Simulate data
	samples, err := bn.Simulate(1000, 123)
	if err != nil {
		fmt.Printf("Error simulating: %v\n", err)
		return
	}
	fmt.Printf("Simulated %d samples\n", len(samples))

	// Learn a new network from simulated data
	newBN, err := models.NewBayesianNetwork(bn.Edges())
	if err != nil {
		fmt.Printf("Error creating new network: %v\n", err)
		return
	}

	err = newBN.Fit(samples)
	if err != nil {
		fmt.Printf("Error fitting: %v\n", err)
		return
	}

	fmt.Println("Successfully learned parameters from data")

	// Compare a CPD
	originalCPD, _ := bn.GetCPD("Alarm")
	learnedCPD, _ := newBN.GetCPD("Alarm")

	fmt.Printf("\nOriginal P(Alarm | Burglary=no, Earthquake=no): %.4f\n",
		originalCPD.Values[0][1])
	fmt.Printf("Learned P(Alarm | Burglary=no, Earthquake=no): %.4f\n",
		learnedCPD.Values[0][1])

	// Prediction
	testSamples := samples[:10]
	for i := range testSamples {
		delete(testSamples[i], "Alarm")
	}

	predictions, err := newBN.Predict(testSamples)
	if err != nil {
		fmt.Printf("Error predicting: %v\n", err)
		return
	}

	fmt.Printf("\nPredicted Alarm for first test sample: %d\n", predictions["Alarm"][0])
}

func structureLearningExample() {
	// Create a simple network and generate data
	bn, err := examples.GetSprinklerModel()
	if err != nil {
		fmt.Printf("Error creating model: %v\n", err)
		return
	}

	// Simulate data
	samples, err := bn.Simulate(500, 456)
	if err != nil {
		fmt.Printf("Error simulating: %v\n", err)
		return
	}

	fmt.Printf("Original structure: %v\n", bn.Edges())

	// Learn structure using PC algorithm
	pc := estimators.NewPC(samples)
	pc.SetAlpha(0.05)

	learnedDAG, err := pc.Estimate()
	if err != nil {
		fmt.Printf("Error learning structure: %v\n", err)
		return
	}

	fmt.Printf("Learned structure: %v\n", learnedDAG.Edges())
	fmt.Printf("Learned nodes: %v\n", learnedDAG.Nodes())

	// Create a new BN with learned structure
	learnedBN, err := models.NewBayesianNetwork(learnedDAG.Edges())
	if err != nil {
		fmt.Printf("Error creating learned BN: %v\n", err)
		return
	}

	// Fit parameters
	err = learnedBN.Fit(samples)
	if err != nil {
		fmt.Printf("Error fitting parameters: %v\n", err)
		return
	}

	fmt.Println("Successfully learned both structure and parameters!")

	// Save to CSV
	df := utils.DataFrameFromSamples(samples, bn.Nodes())
	err = df.SaveCSV("sprinkler_data.csv")
	if err != nil {
		fmt.Printf("Error saving CSV: %v\n", err)
	} else {
		fmt.Println("Saved data to sprinkler_data.csv")
	}
}
