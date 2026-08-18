package inference

import (
	"math"
	"strings"
	"testing"

	"github.com/JohnPierman/bngo/categorical"
	"github.com/JohnPierman/bngo/examples"
	"github.com/JohnPierman/bngo/factors"
)

func weatherElimination(t *testing.T) *VariableElimination {
	t.Helper()

	bn, err := examples.GetWeatherModel()
	if err != nil {
		t.Fatalf("GetWeatherModel() error = %v", err)
	}

	ve, err := NewVariableElimination(bn)
	if err != nil {
		t.Fatalf("NewVariableElimination() error = %v", err)
	}

	return ve
}

func TestVariableElimination_QueryLabeled_MatchesTheNumericQuery(t *testing.T) {
	ve := weatherElimination(t)

	labelled, err := ve.QueryLabeled([]string{"WetGrass"}, map[string]string{"Weather": "rainy"})
	if err != nil {
		t.Fatalf("QueryLabeled() error = %v", err)
	}

	// Weather=rainy is state 1 of [cloudy rainy sunny].
	numeric, err := ve.Query([]string{"WetGrass"}, map[string]int{"Weather": 1})
	if err != nil {
		t.Fatalf("Query() error = %v", err)
	}

	if len(labelled.Assignments) != len(numeric.Values) {
		t.Fatalf("QueryLabeled() returned %d assignments, want %d",
			len(labelled.Assignments), len(numeric.Values))
	}

	for i, assignment := range labelled.Assignments {
		if math.Abs(assignment.Probability-numeric.Values[i]) > 1e-12 {
			t.Errorf("assignment %d probability = %v, want %v", i, assignment.Probability, numeric.Values[i])
		}
	}

	yes, err := labelled.Probability(map[string]string{"WetGrass": "yes"})
	if err != nil {
		t.Fatalf("Probability() error = %v", err)
	}
	no, err := labelled.Probability(map[string]string{"WetGrass": "no"})
	if err != nil {
		t.Fatalf("Probability() error = %v", err)
	}
	if math.Abs(yes+no-1.0) > 1e-9 {
		t.Errorf("labelled probabilities sum to %v, want 1", yes+no)
	}
	if yes <= no {
		t.Errorf("P(WetGrass=yes | rainy) = %.4f, expected it to beat %.4f", yes, no)
	}
}

func TestVariableElimination_QueryLabeled_MostLikely(t *testing.T) {
	ve := weatherElimination(t)

	labelled, err := ve.QueryLabeled([]string{"Weather"}, map[string]string{"WetGrass": "yes", "Sprinkler": "off"})
	if err != nil {
		t.Fatalf("QueryLabeled() error = %v", err)
	}

	best, err := labelled.MostLikely()
	if err != nil {
		t.Fatalf("MostLikely() error = %v", err)
	}
	if best.Labels["Weather"] != "rainy" {
		t.Errorf("MostLikely() Weather = %q, want %q", best.Labels["Weather"], "rainy")
	}
}

func TestVariableElimination_QueryLabeled_JointOverTwoVariables(t *testing.T) {
	ve := weatherElimination(t)

	labelled, err := ve.QueryLabeled([]string{"Sprinkler", "WetGrass"}, nil)
	if err != nil {
		t.Fatalf("QueryLabeled() error = %v", err)
	}

	if len(labelled.Assignments) != 4 {
		t.Fatalf("QueryLabeled() returned %d assignments, want 4", len(labelled.Assignments))
	}

	total := 0.0
	seen := make(map[string]bool)
	for _, assignment := range labelled.Assignments {
		total += assignment.Probability
		seen[assignment.Labels["Sprinkler"]+"/"+assignment.Labels["WetGrass"]] = true
	}

	if math.Abs(total-1.0) > 1e-9 {
		t.Errorf("joint probabilities sum to %v, want 1", total)
	}
	for _, combination := range []string{"off/no", "off/yes", "on/no", "on/yes"} {
		if !seen[combination] {
			t.Errorf("joint distribution is missing %s", combination)
		}
	}
}

func TestVariableElimination_QueryLabeled_RejectsUnknownEvidence(t *testing.T) {
	ve := weatherElimination(t)

	if _, err := ve.QueryLabeled([]string{"WetGrass"}, map[string]string{"Weather": "snowy"}); err == nil {
		t.Error("QueryLabeled() error = nil for an unknown label, want error")
	}
	if _, err := ve.QueryLabeled([]string{"WetGrass"}, map[string]string{"Nope": "x"}); err == nil {
		t.Error("QueryLabeled() error = nil for an unknown variable, want error")
	}
}

func TestVariableElimination_MAPLabeled(t *testing.T) {
	ve := weatherElimination(t)

	labels, err := ve.MAPLabeled([]string{"WetGrass"}, map[string]string{"Weather": "rainy", "Sprinkler": "on"})
	if err != nil {
		t.Fatalf("MAPLabeled() error = %v", err)
	}

	if labels["WetGrass"] != "yes" {
		t.Errorf("MAPLabeled() WetGrass = %q, want %q", labels["WetGrass"], "yes")
	}

	if _, err := ve.MAPLabeled([]string{"WetGrass"}, map[string]string{"Weather": "snowy"}); err == nil {
		t.Error("MAPLabeled() error = nil for an unknown label, want error")
	}
}

func TestLabelDistribution_Errors(t *testing.T) {
	codebook := categorical.NewCodebook()
	if err := codebook.Declare("A", []string{"no", "yes"}); err != nil {
		t.Fatalf("Declare() error = %v", err)
	}

	factor, err := factors.NewDiscreteFactor([]string{"A"}, map[string]int{"A": 2}, []float64{0.4, 0.6})
	if err != nil {
		t.Fatalf("NewDiscreteFactor() error = %v", err)
	}

	if _, err := LabelDistribution(nil, codebook); err == nil {
		t.Error("LabelDistribution(nil factor) error = nil, want error")
	}
	if _, err := LabelDistribution(factor, nil); err == nil {
		t.Error("LabelDistribution(nil codebook) error = nil, want error")
	}

	undeclared, err := factors.NewDiscreteFactor([]string{"B"}, map[string]int{"B": 2}, []float64{0.4, 0.6})
	if err != nil {
		t.Fatalf("NewDiscreteFactor() error = %v", err)
	}
	_, err = LabelDistribution(undeclared, codebook)
	if err == nil {
		t.Fatal("LabelDistribution() error = nil for an undeclared variable, want error")
	}
	if !strings.Contains(err.Error(), "B") {
		t.Errorf("error %q does not name the undeclared variable", err)
	}
}

func TestLabelDistribution_RejectsAStateOutsideTheDeclaredRange(t *testing.T) {
	codebook := categorical.NewCodebook()
	if err := codebook.Declare("A", []string{"no", "yes"}); err != nil {
		t.Fatalf("Declare() error = %v", err)
	}

	// The factor assumes three states while the codebook declares two.
	factor, err := factors.NewDiscreteFactor([]string{"A"}, map[string]int{"A": 3}, []float64{0.2, 0.3, 0.5})
	if err != nil {
		t.Fatalf("NewDiscreteFactor() error = %v", err)
	}

	if _, err := LabelDistribution(factor, codebook); err == nil {
		t.Error("LabelDistribution() error = nil for a state outside the declared range, want error")
	}
}

func TestLabeledDistribution_ProbabilityRejectsPartialAssignments(t *testing.T) {
	ve := weatherElimination(t)

	labelled, err := ve.QueryLabeled([]string{"Sprinkler", "WetGrass"}, nil)
	if err != nil {
		t.Fatalf("QueryLabeled() error = %v", err)
	}

	if _, err := labelled.Probability(map[string]string{"Sprinkler": "on"}); err == nil {
		t.Error("Probability() error = nil for a partial assignment, want error")
	}
	if _, err := labelled.Probability(map[string]string{"Sprinkler": "on", "WetGrass": "damp"}); err == nil {
		t.Error("Probability() error = nil for an unknown label, want error")
	}
}

func TestLabeledDistribution_MostLikelyOnAnEmptyDistribution(t *testing.T) {
	empty := &LabeledDistribution{}

	if _, err := empty.MostLikely(); err == nil {
		t.Error("MostLikely() error = nil on an empty distribution, want error")
	}
}

func TestLabeledDistribution_String(t *testing.T) {
	ve := weatherElimination(t)

	labelled, err := ve.QueryLabeled([]string{"WetGrass"}, map[string]string{"Weather": "rainy"})
	if err != nil {
		t.Fatalf("QueryLabeled() error = %v", err)
	}

	rendered := labelled.String()
	for _, want := range []string{"P(WetGrass)", "WetGrass=no", "WetGrass=yes"} {
		if !strings.Contains(rendered, want) {
			t.Errorf("String() = %q, want it to contain %q", rendered, want)
		}
	}
}
