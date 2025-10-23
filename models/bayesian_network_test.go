package models

import (
	"testing"

	"github.com/JohnPierman/bngo/factors"
)

func TestBayesianNetworkCreation(t *testing.T) {
	edges := [][2]string{
		{"A", "C"},
		{"B", "C"},
	}

	bn, err := NewBayesianNetwork(edges)
	if err != nil {
		t.Errorf("Failed to create network: %v", err)
	}

	if len(bn.Nodes()) != 3 {
		t.Errorf("Expected 3 nodes, got %d", len(bn.Nodes()))
	}
}

func TestBayesianNetworkCPD(t *testing.T) {
	edges := [][2]string{
		{"A", "B"},
	}

	bn, _ := NewBayesianNetwork(edges)

	// Add CPD for A
	cpdA, _ := factors.NewTabularCPD("A", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)

	err := bn.AddCPD(cpdA)
	if err != nil {
		t.Errorf("Failed to add CPD: %v", err)
	}

	// Add CPD for B
	cpdB, _ := factors.NewTabularCPD("B", 2,
		[][]float64{
			{0.8, 0.2}, // A=0
			{0.3, 0.7}, // A=1
		},
		[]string{"A"},
		map[string]int{"A": 2},
	)

	err = bn.AddCPD(cpdB)
	if err != nil {
		t.Errorf("Failed to add CPD: %v", err)
	}

	// Check model
	err = bn.CheckModel()
	if err != nil {
		t.Errorf("Model check failed: %v", err)
	}
}

func TestBayesianNetworkSimulation(t *testing.T) {
	edges := [][2]string{
		{"A", "B"},
	}

	bn, _ := NewBayesianNetwork(edges)

	cpdA, _ := factors.NewTabularCPD("A", 2,
		[][]float64{{0.5, 0.5}},
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

	samples, err := bn.Simulate(100, 42)
	if err != nil {
		t.Errorf("Simulation failed: %v", err)
	}

	if len(samples) != 100 {
		t.Errorf("Expected 100 samples, got %d", len(samples))
	}

	// Check that each sample has both variables
	for i, sample := range samples {
		if _, ok := sample["A"]; !ok {
			t.Errorf("Sample %d missing variable A", i)
		}
		if _, ok := sample["B"]; !ok {
			t.Errorf("Sample %d missing variable B", i)
		}
	}
}

func TestBayesianNetworkFit(t *testing.T) {
	// Create synthetic data
	samples := []map[string]int{
		{"A": 0, "B": 0},
		{"A": 0, "B": 1},
		{"A": 1, "B": 1},
		{"A": 1, "B": 1},
	}

	edges := [][2]string{
		{"A", "B"},
	}

	bn, _ := NewBayesianNetwork(edges)
	err := bn.Fit(samples)
	if err != nil {
		t.Errorf("Fitting failed: %v", err)
	}

	// Check that CPDs were learned
	if len(bn.CPDs) != 2 {
		t.Errorf("Expected 2 CPDs, got %d", len(bn.CPDs))
	}

	cpdA, err := bn.GetCPD("A")
	if err != nil {
		t.Errorf("Failed to get CPD for A: %v", err)
	}

	if cpdA.VariableCard != 2 {
		t.Errorf("Expected cardinality 2 for A, got %d", cpdA.VariableCard)
	}
}
