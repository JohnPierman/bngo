package models

import (
	"testing"

	"github.com/JohnPierman/bngo/factors"
)

func BenchmarkBayesianNetworkSimulate(b *testing.B) {
	edges := [][2]string{
		{"A", "C"},
		{"B", "C"},
		{"C", "D"},
	}

	bn, _ := NewBayesianNetwork(edges)

	// Add CPDs
	cpdA, _ := factors.NewTabularCPD("A", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)
	_ = bn.AddCPD(cpdA)

	cpdB, _ := factors.NewTabularCPD("B", 2,
		[][]float64{{0.7, 0.3}},
		[]string{},
		map[string]int{},
	)
	_ = bn.AddCPD(cpdB)

	cpdC, _ := factors.NewTabularCPD("C", 2,
		[][]float64{
			{0.9, 0.1},
			{0.5, 0.5},
			{0.6, 0.4},
			{0.2, 0.8},
		},
		[]string{"A", "B"},
		map[string]int{"A": 2, "B": 2},
	)
	_ = bn.AddCPD(cpdC)

	cpdD, _ := factors.NewTabularCPD("D", 2,
		[][]float64{
			{0.8, 0.2},
			{0.3, 0.7},
		},
		[]string{"C"},
		map[string]int{"C": 2},
	)
	bn.AddCPD(cpdD)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bn.Simulate(100, 42)
	}
}

func BenchmarkBayesianNetworkFit(b *testing.B) {
	// Create test data
	data := make([]map[string]int, 1000)
	for i := range data {
		data[i] = map[string]int{
			"A": i % 2,
			"B": (i / 2) % 2,
		}
	}

	edges := [][2]string{{"A", "B"}}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bn, _ := NewBayesianNetwork(edges)
		bn.Fit(data)
	}
}

func BenchmarkBayesianNetworkPredict(b *testing.B) {
	edges := [][2]string{{"A", "B"}}
	bn, _ := NewBayesianNetwork(edges)

	cpdA, _ := factors.NewTabularCPD("A", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)
	_ = bn.AddCPD(cpdA)

	cpdB, _ := factors.NewTabularCPD("B", 2,
		[][]float64{
			{0.8, 0.2},
			{0.3, 0.7},
		},
		[]string{"A"},
		map[string]int{"A": 2},
	)
	_ = bn.AddCPD(cpdB)

	testData := make([]map[string]int, 100)
	for i := range testData {
		testData[i] = map[string]int{"A": i % 2}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bn.Predict(testData)
	}
}
