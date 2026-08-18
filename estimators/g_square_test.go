package estimators

import (
	"math"
	"testing"
)

func TestGSquareTest_FindsTheIndependenciesOfAChain(t *testing.T) {
	data := chainData(4000, 21)
	cardinality := map[string]int{"A": 2, "B": 2, "C": 2}

	tests := []struct {
		name        string
		x, y        string
		z           []string
		independent bool
	}{
		{"neighbours are dependent", "A", "B", nil, false},
		{"ends of the chain are dependent", "A", "C", nil, false},
		{"ends are independent given the middle", "A", "C", []string{"B"}, true},
		{"neighbours stay dependent given the third", "A", "B", []string{"C"}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			statistic, pValue := GSquareTest(data, tt.x, tt.y, tt.z, cardinality)

			if tt.independent {
				if pValue <= 0.05 {
					t.Errorf("p-value = %.4g (statistic %.3f), want above 0.05", pValue, statistic)
				}
				return
			}
			if pValue > 0.05 {
				t.Errorf("p-value = %.4g (statistic %.3f), want at most 0.05", pValue, statistic)
			}
		})
	}
}

func TestGSquareTest_FindsTheIndependenciesOfACollider(t *testing.T) {
	data := colliderData(4000, 22)
	cardinality := map[string]int{"A": 2, "B": 2, "C": 2}

	// The parents of a collider are independent until the collider is observed,
	// which is the pattern that distinguishes a collider from a chain.
	_, marginal := GSquareTest(data, "A", "B", nil, cardinality)
	if marginal <= 0.05 {
		t.Errorf("P(A independent of B) = %.4g, want above 0.05", marginal)
	}

	_, conditional := GSquareTest(data, "A", "B", []string{"C"}, cardinality)
	if conditional > 0.05 {
		t.Errorf("P(A independent of B given C) = %.4g, want at most 0.05", conditional)
	}
}

func TestGSquareTest_ReportsIndependenceForUnrelatedVariables(t *testing.T) {
	data := independentData(4000, 23)
	cardinality := map[string]int{"A": 2, "B": 2, "C": 2}

	for _, pair := range [][2]string{{"A", "B"}, {"A", "C"}, {"B", "C"}} {
		_, pValue := GSquareTest(data, pair[0], pair[1], nil, cardinality)
		if pValue <= 0.05 {
			t.Errorf("p-value for %s and %s = %.4g, want above 0.05", pair[0], pair[1], pValue)
		}
	}
}

func TestGSquareTest_HasNoOpinionWithoutData(t *testing.T) {
	cardinality := map[string]int{"A": 2, "B": 2}

	statistic, pValue := GSquareTest(nil, "A", "B", nil, cardinality)
	if statistic != 0 || pValue != 1.0 {
		t.Errorf("GSquareTest() = (%v, %v) with no data, want (0, 1)", statistic, pValue)
	}
}

func TestGSquareTest_HasNoOpinionOnAConstantVariable(t *testing.T) {
	// B never varies, so there is nothing for a test to reject.
	data := []map[string]int{
		{"A": 0, "B": 0},
		{"A": 1, "B": 0},
		{"A": 0, "B": 0},
		{"A": 1, "B": 0},
	}

	_, pValue := GSquareTest(data, "A", "B", nil, map[string]int{"A": 2, "B": 1})
	if pValue != 1.0 {
		t.Errorf("p-value = %v for a constant variable, want 1", pValue)
	}
}

func TestGSquareTest_InfersAMissingCardinality(t *testing.T) {
	data := chainData(2000, 24)

	// The cardinality of B is not given, so it is read off the data instead.
	_, inferred := GSquareTest(data, "A", "C", []string{"B"}, map[string]int{"A": 2, "C": 2})
	_, declared := GSquareTest(data, "A", "C", []string{"B"},
		map[string]int{"A": 2, "B": 2, "C": 2})

	if math.Abs(inferred-declared) > 1e-9 {
		t.Errorf("p-value with an inferred cardinality = %v, want %v as when it is declared",
			inferred, declared)
	}
}

func TestGSquareTest_HasNoOpinionOnAConditioningVariableAbsentFromTheData(t *testing.T) {
	data := chainData(200, 24)

	// Z appears nowhere in the data, so there is nothing to condition on.
	_, pValue := GSquareTest(data, "A", "C", []string{"Z"},
		map[string]int{"A": 2, "B": 2, "C": 2})
	if pValue != 1.0 {
		t.Errorf("p-value = %v conditioning on a variable absent from the data, want 1", pValue)
	}
}

func TestGSquareTest_IsSymmetric(t *testing.T) {
	data := chainData(2000, 25)
	cardinality := map[string]int{"A": 2, "B": 2, "C": 2}

	forward, forwardP := GSquareTest(data, "A", "C", []string{"B"}, cardinality)
	backward, backwardP := GSquareTest(data, "C", "A", []string{"B"}, cardinality)

	// The statistic is a sum over strata, so swapping the arguments changes the
	// order the terms are added in and the totals agree only to rounding.
	if math.Abs(forward-backward) > 1e-9 || math.Abs(forwardP-backwardP) > 1e-9 {
		t.Errorf("GSquareTest() = (%v, %v) one way and (%v, %v) the other",
			forward, forwardP, backward, backwardP)
	}
}

func TestGSquareTest_SkipsRowsWithUnobservedValues(t *testing.T) {
	complete := []map[string]int{
		{"A": 0, "B": 0},
		{"A": 1, "B": 1},
		{"A": 0, "B": 0},
		{"A": 1, "B": 1},
	}
	withGaps := append([]map[string]int{}, complete...)
	withGaps = append(withGaps, map[string]int{"A": 0}, map[string]int{"B": 1})

	cardinality := map[string]int{"A": 2, "B": 2}

	wantStatistic, wantP := GSquareTest(complete, "A", "B", nil, cardinality)
	gotStatistic, gotP := GSquareTest(withGaps, "A", "B", nil, cardinality)

	if gotStatistic != wantStatistic || gotP != wantP {
		t.Errorf("GSquareTest() = (%v, %v) with gaps, want (%v, %v) as without them",
			gotStatistic, gotP, wantStatistic, wantP)
	}
}

func TestGSquareTest_UsesTheSparsePathForHighCardinalityData(t *testing.T) {
	// Three conditioning variables of 60 states each need 216000 strata, far past
	// the point where a dense table is worth allocating, so this exercises the
	// sparse counting path. Only a handful of strata are ever occupied.
	const states = 60

	cardinality := map[string]int{"X": 2, "Y": 2, "Z1": states, "Z2": states, "Z3": states}
	data := make([]map[string]int, 0, 400)

	for i := 0; i < 400; i++ {
		// The conditioning pattern repeats every 60 rows, so X is taken from the
		// cycle count rather than from i. That way X still varies inside each
		// stratum, and the test has some degrees of freedom to work with.
		value := (i / 60) % 2
		data = append(data, map[string]int{
			"X": value, "Y": value,
			"Z1": i % 3, "Z2": i % 4, "Z3": i % 5,
		})
	}

	// X and Y are equal in every row, so they are strongly dependent whatever the
	// conditioning set.
	_, pValue := GSquareTest(data, "X", "Y", []string{"Z1", "Z2", "Z3"}, cardinality)
	if pValue > 0.05 {
		t.Errorf("p-value = %.4g for two identical variables, want at most 0.05", pValue)
	}
}

func TestGSquareTest_TreatsAStateOutsideTheDeclaredRangeAsUnobserved(t *testing.T) {
	data := []map[string]int{
		{"A": 0, "B": 0},
		{"A": 1, "B": 1},
		{"A": 0, "B": 0},
		{"A": 1, "B": 1},
		{"A": 5, "B": 5}, // outside the two declared states
	}

	cardinality := map[string]int{"A": 2, "B": 2}

	withOutlier, pOutlier := GSquareTest(data, "A", "B", nil, cardinality)
	withoutOutlier, pClean := GSquareTest(data[:4], "A", "B", nil, cardinality)

	if withOutlier != withoutOutlier || pOutlier != pClean {
		t.Errorf("GSquareTest() = (%v, %v) with an out of range row, want (%v, %v)",
			withOutlier, pOutlier, withoutOutlier, pClean)
	}
}
