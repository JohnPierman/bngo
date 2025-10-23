package factors

import (
	"testing"
)

func BenchmarkFactorMultiply(b *testing.B) {
	factor1, _ := NewDiscreteFactor(
		[]string{"A", "B"},
		map[string]int{"A": 2, "B": 2},
		[]float64{0.3, 0.7, 0.4, 0.6},
	)
	
	factor2, _ := NewDiscreteFactor(
		[]string{"B", "C"},
		map[string]int{"B": 2, "C": 2},
		[]float64{0.5, 0.5, 0.2, 0.8},
	)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		factor1.Multiply(factor2)
	}
}

func BenchmarkFactorMarginalize(b *testing.B) {
	factor, _ := NewDiscreteFactor(
		[]string{"A", "B", "C"},
		map[string]int{"A": 2, "B": 2, "C": 2},
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
	)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		factor.Marginalize([]string{"B"})
	}
}

func BenchmarkFactorReduce(b *testing.B) {
	factor, _ := NewDiscreteFactor(
		[]string{"A", "B", "C"},
		map[string]int{"A": 2, "B": 2, "C": 2},
		[]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
	)
	
	evidence := map[string]int{"A": 1}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		factor.Reduce(evidence)
	}
}

func BenchmarkFactorNormalize(b *testing.B) {
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		factor, _ := NewDiscreteFactor(
			[]string{"A"},
			map[string]int{"A": 2},
			[]float64{1.0, 3.0},
		)
		b.StartTimer()
		
		factor.Normalize()
	}
}

func BenchmarkCPDToFactor(b *testing.B) {
	cpd, _ := NewTabularCPD("B", 2,
		[][]float64{
			{0.8, 0.2},
			{0.3, 0.7},
		},
		[]string{"A"},
		map[string]int{"A": 2},
	)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cpd.ToFactor()
	}
}

func BenchmarkFactorMultiply_Large(b *testing.B) {
	// Large factors with 3 variables each
	factor1, _ := NewDiscreteFactor(
		[]string{"A", "B", "C"},
		map[string]int{"A": 3, "B": 3, "C": 3},
		make([]float64, 27), // 3^3 values
	)
	
	factor2, _ := NewDiscreteFactor(
		[]string{"C", "D", "E"},
		map[string]int{"C": 3, "D": 3, "E": 3},
		make([]float64, 27),
	)
	
	// Fill with dummy values
	for i := range factor1.Values {
		factor1.Values[i] = 0.1
	}
	for i := range factor2.Values {
		factor2.Values[i] = 0.1
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		factor1.Multiply(factor2)
	}
}

