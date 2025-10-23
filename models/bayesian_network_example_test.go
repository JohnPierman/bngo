package models_test

import (
	"fmt"
	"log"
	
	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/inference"
	"github.com/JohnPierman/bngo/models"
)

// ExampleBayesianNetwork_Simulate demonstrates data simulation
func ExampleBayesianNetwork_Simulate() {
	// Create a simple network
	edges := [][2]string{{"A", "B"}}
	bn, _ := models.NewBayesianNetwork(edges)
	
	// Add CPDs
	cpdA, _ := factors.NewTabularCPD("A", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)
	bn.AddCPD(cpdA)
	
	cpdB, _ := factors.NewTabularCPD("B", 2,
		[][]float64{
			{0.8, 0.2},
			{0.3, 0.7},
		},
		[]string{"A"},
		map[string]int{"A": 2},
	)
	bn.AddCPD(cpdB)
	
	// Simulate data
	samples, _ := bn.Simulate(5, 42)
	fmt.Printf("Generated %d samples\n", len(samples))
	
	// Output:
	// Generated 5 samples
}

// ExampleBayesianNetwork_Fit demonstrates parameter learning
func ExampleBayesianNetwork_Fit() {
	// Create training data
	data := []map[string]int{
		{"A": 0, "B": 0},
		{"A": 0, "B": 1},
		{"A": 1, "B": 1},
		{"A": 1, "B": 1},
	}
	
	// Create network structure
	edges := [][2]string{{"A", "B"}}
	bn, _ := models.NewBayesianNetwork(edges)
	
	// Learn parameters
	err := bn.Fit(data)
	if err != nil {
		log.Fatal(err)
	}
	
	fmt.Println("Parameters learned successfully")
	
	// Output:
	// Parameters learned successfully
}

// ExampleBayesianNetwork_Predict demonstrates prediction
func ExampleBayesianNetwork_Predict() {
	// Create and fit a network
	edges := [][2]string{{"A", "B"}}
	bn, _ := models.NewBayesianNetwork(edges)
	
	cpdA, _ := factors.NewTabularCPD("A", 2,
		[][]float64{{0.7, 0.3}},
		[]string{},
		map[string]int{},
	)
	bn.AddCPD(cpdA)
	
	cpdB, _ := factors.NewTabularCPD("B", 2,
		[][]float64{
			{0.9, 0.1},
			{0.2, 0.8},
		},
		[]string{"A"},
		map[string]int{"A": 2},
	)
	bn.AddCPD(cpdB)
	
	// Predict missing values
	observations := []map[string]int{
		{"A": 1},
		{"A": 0},
	}
	
	predictions, _ := bn.Predict(observations)
	fmt.Printf("Predicted %d values for B\n", len(predictions["B"]))
	
	// Output:
	// Predicted 2 values for B
}

// ExampleNewBayesianNetwork demonstrates network creation
func ExampleNewBayesianNetwork() {
	edges := [][2]string{
		{"A", "C"},
		{"B", "C"},
	}
	
	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		log.Fatal(err)
	}
	
	fmt.Printf("Created network with %d nodes\n", len(bn.Nodes()))
	
	// Output:
	// Created network with 3 nodes
}

// Example_inference demonstrates using Variable Elimination for inference
func Example_inference() {
	// Create a simple network
	edges := [][2]string{{"A", "B"}, {"B", "C"}}
	bn, _ := models.NewBayesianNetwork(edges)
	
	// Add CPDs
	cpdA, _ := factors.NewTabularCPD("A", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)
	bn.AddCPD(cpdA)
	
	cpdB, _ := factors.NewTabularCPD("B", 2,
		[][]float64{
			{0.8, 0.2},
			{0.3, 0.7},
		},
		[]string{"A"},
		map[string]int{"A": 2},
	)
	bn.AddCPD(cpdB)
	
	cpdC, _ := factors.NewTabularCPD("C", 2,
		[][]float64{
			{0.9, 0.1},
			{0.4, 0.6},
		},
		[]string{"B"},
		map[string]int{"B": 2},
	)
	bn.AddCPD(cpdC)
	
	// Perform inference
	ve, _ := inference.NewVariableElimination(bn)
	result, _ := ve.Query([]string{"C"}, map[string]int{"A": 1})
	
	fmt.Printf("P(C | A=1) computed with %d values\n", len(result.Values))
	
	// Output:
	// P(C | A=1) computed with 2 values
}

