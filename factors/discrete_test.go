package factors

import (
	"math"
	"testing"
)

func TestFactorCreation(t *testing.T) {
	factor, err := NewDiscreteFactor(
		[]string{"A", "B"},
		map[string]int{"A": 2, "B": 2},
		[]float64{0.3, 0.7, 0.4, 0.6},
	)

	if err != nil {
		t.Errorf("Failed to create factor: %v", err)
	}

	if len(factor.Variables) != 2 {
		t.Errorf("Expected 2 variables, got %d", len(factor.Variables))
	}

	if len(factor.Values) != 4 {
		t.Errorf("Expected 4 values, got %d", len(factor.Values))
	}
}

func TestFactorMultiply(t *testing.T) {
	factor1, _ := NewDiscreteFactor(
		[]string{"A"},
		map[string]int{"A": 2},
		[]float64{0.3, 0.7},
	)

	factor2, _ := NewDiscreteFactor(
		[]string{"A"},
		map[string]int{"A": 2},
		[]float64{0.5, 0.5},
	)

	result, err := factor1.Multiply(factor2)
	if err != nil {
		t.Errorf("Multiplication failed: %v", err)
	}

	expected := []float64{0.15, 0.35}
	for i, v := range result.Values {
		if math.Abs(v-expected[i]) > 0.0001 {
			t.Errorf("Value %d: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestFactorMarginalize(t *testing.T) {
	factor, _ := NewDiscreteFactor(
		[]string{"A", "B"},
		map[string]int{"A": 2, "B": 2},
		[]float64{0.1, 0.2, 0.3, 0.4},
	)

	result, err := factor.Marginalize([]string{"B"})
	if err != nil {
		t.Errorf("Marginalization failed: %v", err)
	}

	if len(result.Variables) != 1 {
		t.Errorf("Expected 1 variable after marginalization, got %d", len(result.Variables))
	}

	expected := []float64{0.3, 0.7} // Sum over B
	for i, v := range result.Values {
		if math.Abs(v-expected[i]) > 0.0001 {
			t.Errorf("Value %d: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestFactorReduce(t *testing.T) {
	factor, _ := NewDiscreteFactor(
		[]string{"A", "B"},
		map[string]int{"A": 2, "B": 2},
		[]float64{0.1, 0.2, 0.3, 0.4},
	)

	evidence := map[string]int{"A": 1}
	result, err := factor.Reduce(evidence)
	if err != nil {
		t.Errorf("Reduction failed: %v", err)
	}

	if len(result.Variables) != 1 {
		t.Errorf("Expected 1 variable after reduction, got %d", len(result.Variables))
	}

	expected := []float64{0.3, 0.4}
	for i, v := range result.Values {
		if math.Abs(v-expected[i]) > 0.0001 {
			t.Errorf("Value %d: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestFactorNormalize(t *testing.T) {
	factor, _ := NewDiscreteFactor(
		[]string{"A"},
		map[string]int{"A": 2},
		[]float64{1.0, 3.0},
	)

	err := factor.Normalize()
	if err != nil {
		t.Errorf("Normalization failed: %v", err)
	}

	sum := 0.0
	for _, v := range factor.Values {
		sum += v
	}

	if math.Abs(sum-1.0) > 0.0001 {
		t.Errorf("Expected sum=1.0, got %f", sum)
	}
}
