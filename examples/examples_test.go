package examples

import (
	"fmt"
	"log"
)

// ExampleGetStudentModel demonstrates how to use the Student Bayesian Network
func ExampleGetStudentModel() {
	// Load the pre-built Student model
	bn, err := GetStudentModel()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Nodes: %d\n", len(bn.Nodes()))
	fmt.Printf("Edges: %d\n", len(bn.Edges()))

	// Simulate some data
	samples, _ := bn.Simulate(10, 42)
	fmt.Printf("Generated %d samples\n", len(samples))

	// Output:
	// Nodes: 5
	// Edges: 4
	// Generated 10 samples
}

// ExampleGetAlarmModel demonstrates the Alarm network
func ExampleGetAlarmModel() {
	bn, err := GetAlarmModel()
	if err != nil {
		log.Fatal(err)
	}

	// Check the structure
	fmt.Printf("Network has %d variables\n", len(bn.Nodes()))

	// Get a CPD
	alarmCPD, _ := bn.GetCPD("Alarm")
	fmt.Printf("Alarm has %d states\n", alarmCPD.VariableCard)

	// Output:
	// Network has 5 variables
	// Alarm has 2 states
}

// ExampleGetCancerModel demonstrates the Cancer diagnosis network
func ExampleGetCancerModel() {
	bn, err := GetCancerModel()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Cancer network nodes: %d\n", len(bn.Nodes()))

	// Output:
	// Cancer network nodes: 5
}

// ExampleGetSprinklerModel demonstrates the classic Sprinkler network
func ExampleGetSprinklerModel() {
	bn, err := GetSprinklerModel()
	if err != nil {
		log.Fatal(err)
	}

	// Simulate data
	samples, _ := bn.Simulate(100, 123)
	fmt.Printf("Simulated %d samples from Sprinkler network\n", len(samples))

	// Output:
	// Simulated 100 samples from Sprinkler network
}
