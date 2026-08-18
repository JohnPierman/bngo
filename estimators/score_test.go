package estimators

import (
	"math"
	"testing"

	"github.com/JohnPierman/bngo/graph"
)

// singleVariableData returns 7 rows of X=0 and 3 rows of X=1, small enough that
// every score can be worked out by hand.
func singleVariableData() []map[string]int {
	data := make([]map[string]int, 0, 10)
	for i := 0; i < 7; i++ {
		data = append(data, map[string]int{"X": 0})
	}
	for i := 0; i < 3; i++ {
		data = append(data, map[string]int{"X": 1})
	}
	return data
}

func TestScores_MatchValuesWorkedOutByHand(t *testing.T) {
	data := singleVariableData()

	bdeu, err := NewBDeu(data, nil, 10)
	if err != nil {
		t.Fatalf("NewBDeu() error = %v", err)
	}

	tests := []struct {
		name  string
		score *Score
		want  float64
	}{
		// Log likelihood of the sample is 7*ln(0.7) + 3*ln(0.3) = -6.108643.
		// BIC subtracts 0.5*ln(10) for the one free parameter.
		{"BIC", NewBIC(data, nil), -7.259936},
		// AIC subtracts 1 for the one free parameter.
		{"AIC", NewAIC(data, nil), -7.108643},
		// K2 is lnG(2) - lnG(12) + lnG(8) + lnG(4).
		{"K2", NewK2(data, nil), -7.185387},
		// BDeu with prior weight 10 spreads 5 over each of the two cells:
		// lnG(10) - lnG(20) + lnG(12) - lnG(5) + lnG(8) - lnG(5).
		{"BDeu", bdeu, -6.866695},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.score.Name(); got != tt.name {
				t.Errorf("Name() = %q, want %q", got, tt.name)
			}

			got, err := tt.score.LocalScore("X", nil)
			if err != nil {
				t.Fatalf("LocalScore() error = %v", err)
			}
			if math.Abs(got-tt.want) > 1e-6 {
				t.Errorf("LocalScore() = %.6f, want %.6f", got, tt.want)
			}
		})
	}
}

func TestScores_AreEqualForMarkovEquivalentStructures(t *testing.T) {
	data := chainData(4000, 1)

	// These three networks encode exactly the same independencies, so a score
	// equivalent criterion must not prefer any of them over the others.
	equivalent := [][][2]string{
		{{"A", "B"}, {"B", "C"}},
		{{"B", "A"}, {"B", "C"}},
		{{"C", "B"}, {"B", "A"}},
	}

	bdeu, err := NewBDeu(data, nil, 10)
	if err != nil {
		t.Fatalf("NewBDeu() error = %v", err)
	}

	for _, score := range []*Score{NewBIC(data, nil), bdeu} {
		t.Run(score.Name(), func(t *testing.T) {
			reference := 0.0
			for i, edges := range equivalent {
				dag, err := graph.NewDAGFromEdges(edges)
				if err != nil {
					t.Fatalf("NewDAGFromEdges() error = %v", err)
				}

				total, err := ScoreDAG(dag, score)
				if err != nil {
					t.Fatalf("ScoreDAG() error = %v", err)
				}

				if i == 0 {
					reference = total
					continue
				}
				if math.Abs(total-reference) > 1e-6 {
					t.Errorf("score of %v = %.6f, want %.6f like the equivalent structure",
						edges, total, reference)
				}
			}
		})
	}
}

func TestScoreDAG_PrefersTheGeneratingStructure(t *testing.T) {
	data := chainData(4000, 2)
	score := NewBIC(data, nil)

	truth, err := graph.NewDAGFromEdges([][2]string{{"A", "B"}, {"B", "C"}})
	if err != nil {
		t.Fatalf("NewDAGFromEdges() error = %v", err)
	}

	empty := graph.NewDAG()
	for _, node := range []string{"A", "B", "C"} {
		empty.AddNode(node)
	}

	// A -> C is redundant once B is a parent of C, so the penalty should tell.
	overfitted, err := graph.NewDAGFromEdges([][2]string{{"A", "B"}, {"B", "C"}, {"A", "C"}})
	if err != nil {
		t.Fatalf("NewDAGFromEdges() error = %v", err)
	}

	truthScore, err := ScoreDAG(truth, score)
	if err != nil {
		t.Fatalf("ScoreDAG() error = %v", err)
	}
	emptyScore, err := ScoreDAG(empty, score)
	if err != nil {
		t.Fatalf("ScoreDAG() error = %v", err)
	}
	overfittedScore, err := ScoreDAG(overfitted, score)
	if err != nil {
		t.Fatalf("ScoreDAG() error = %v", err)
	}

	if truthScore <= emptyScore {
		t.Errorf("BIC of the true chain = %.3f, want it above the empty graph %.3f",
			truthScore, emptyScore)
	}
	if truthScore <= overfittedScore {
		t.Errorf("BIC of the true chain = %.3f, want it above the overfitted graph %.3f",
			truthScore, overfittedScore)
	}
}

func TestScore_CachesLocalScores(t *testing.T) {
	score := NewBIC(chainData(200, 3), nil)

	first, err := score.LocalScore("C", []string{"B", "A"})
	if err != nil {
		t.Fatalf("LocalScore() error = %v", err)
	}

	// The parent set is given in a different order the second time, so this also
	// checks that the cache key does not depend on the order.
	second, err := score.LocalScore("C", []string{"A", "B"})
	if err != nil {
		t.Fatalf("LocalScore() error = %v", err)
	}

	if first != second {
		t.Errorf("LocalScore() = %v then %v for the same family", first, second)
	}
}

func TestScore_DoesNotModifyTheParentSlice(t *testing.T) {
	score := NewBIC(chainData(100, 4), nil)
	parents := []string{"B", "A"}

	if _, err := score.LocalScore("C", parents); err != nil {
		t.Fatalf("LocalScore() error = %v", err)
	}

	if parents[0] != "B" || parents[1] != "A" {
		t.Errorf("LocalScore() reordered its argument: %v", parents)
	}
}

func TestNewBDeu_RejectsANonPositivePriorWeight(t *testing.T) {
	for _, weight := range []float64{0, -1} {
		if _, err := NewBDeu(chainData(10, 5), nil, weight); err == nil {
			t.Errorf("NewBDeu(ess=%v) error = nil, want error", weight)
		}
	}
}

func TestScore_DeclaredCardinalityOverridesTheData(t *testing.T) {
	data := singleVariableData()

	fromData := NewBIC(data, nil)
	declared := NewBIC(data, map[string]int{"X": 4})

	if got := fromData.Cardinality()["X"]; got != 2 {
		t.Errorf("cardinality read off the data = %d, want 2", got)
	}
	if got := declared.Cardinality()["X"]; got != 4 {
		t.Errorf("declared cardinality = %d, want 4", got)
	}

	// Four states means three free parameters instead of one, so the penalty grows
	// and the score falls.
	loose, err := fromData.LocalScore("X", nil)
	if err != nil {
		t.Fatalf("LocalScore() error = %v", err)
	}
	strict, err := declared.LocalScore("X", nil)
	if err != nil {
		t.Fatalf("LocalScore() error = %v", err)
	}

	if strict >= loose {
		t.Errorf("BIC with 4 declared states = %.4f, want it below the 2 state score %.4f",
			strict, loose)
	}
}

func TestScore_Accessors(t *testing.T) {
	score := NewBIC(chainData(50, 6), nil)

	if got := score.SampleSize(); got != 50 {
		t.Errorf("SampleSize() = %d, want 50", got)
	}

	variables := score.Variables()
	if len(variables) != 3 || variables[0] != "A" || variables[2] != "C" {
		t.Errorf("Variables() = %v, want sorted A B C", variables)
	}

	variables[0] = "mutated"
	if score.Variables()[0] != "A" {
		t.Error("Variables() returns the internal slice")
	}

	cardinality := score.Cardinality()
	cardinality["A"] = 99
	if score.Cardinality()["A"] != 2 {
		t.Error("Cardinality() returns the internal map")
	}
}

func TestScore_LocalScoreRejectsUnknownVariables(t *testing.T) {
	score := NewBIC(chainData(50, 7), nil)

	if _, err := score.LocalScore("Nope", nil); err == nil {
		t.Error("LocalScore() error = nil for an unknown variable, want error")
	}
	if _, err := score.LocalScore("A", []string{"Nope"}); err == nil {
		t.Error("LocalScore() error = nil for an unknown parent, want error")
	}
}

func TestScore_EmptyData(t *testing.T) {
	score := NewBIC(nil, nil)

	if got := score.SampleSize(); got != 0 {
		t.Errorf("SampleSize() = %d, want 0", got)
	}
	if got := len(score.Variables()); got != 0 {
		t.Errorf("Variables() has %d entries, want 0", got)
	}
	if _, err := score.LocalScore("A", nil); err == nil {
		t.Error("LocalScore() error = nil with no data, want error")
	}
}

func TestScoreDAG_RejectsMissingArguments(t *testing.T) {
	score := NewBIC(chainData(50, 8), nil)
	dag, err := graph.NewDAGFromEdges([][2]string{{"A", "B"}})
	if err != nil {
		t.Fatalf("NewDAGFromEdges() error = %v", err)
	}

	if _, err := ScoreDAG(nil, score); err == nil {
		t.Error("ScoreDAG(nil graph) error = nil, want error")
	}
	if _, err := ScoreDAG(dag, nil); err == nil {
		t.Error("ScoreDAG(nil score) error = nil, want error")
	}
}

func TestScoreDAG_ReportsAScoringFailure(t *testing.T) {
	score := NewBIC(chainData(50, 9), nil)

	// D never appears in the data, so it has no states to score.
	dag, err := graph.NewDAGFromEdges([][2]string{{"A", "D"}})
	if err != nil {
		t.Fatalf("NewDAGFromEdges() error = %v", err)
	}

	if _, err := ScoreDAG(dag, score); err == nil {
		t.Error("ScoreDAG() error = nil for a variable absent from the data, want error")
	}
}

func TestNewFamilyIndexer_RejectsAnUnindexableParentSet(t *testing.T) {
	// Three parents of two billion states each overflow a 64 bit index.
	huge := map[string]int{"A": 1 << 21, "B": 1 << 21, "C": 1 << 21}

	data := newColumnData([]map[string]int{{"A": 0, "B": 0, "C": 0}}, huge)

	if _, err := data.indexer([]string{"A", "B", "C"}); err == nil {
		t.Error("indexer() error = nil for an unindexable parent set, want error")
	}
}

func TestColumnData_SkipsIncompleteRows(t *testing.T) {
	data := []map[string]int{
		{"A": 0, "B": 0},
		{"A": 1, "B": 1},
		{"A": 1}, // B unobserved, so this row cannot inform P(B|A)
	}

	counts, configs, err := newColumnData(data, nil).count("B", []string{"A"})
	if err != nil {
		t.Fatalf("count() error = %v", err)
	}

	if configs != 2 {
		t.Errorf("count() reported %v parent configurations, want 2", configs)
	}

	total := 0.0
	for _, subtotal := range counts.totals {
		total += subtotal
	}
	if total != 2 {
		t.Errorf("count() used %v rows, want the 2 complete ones", total)
	}
}
