package models

import (
	"math"
	"strings"
	"testing"

	"github.com/JohnPierman/bngo/categorical"
	"github.com/JohnPierman/bngo/factors"
)

// weatherNetwork builds a labelled network whose Weather field is categorical and
// whose Sprinkler and WetGrass fields are binary.
func weatherNetwork(t *testing.T) *BayesianNetwork {
	t.Helper()

	bn, err := NewBayesianNetwork([][2]string{
		{"Weather", "Sprinkler"},
		{"Weather", "WetGrass"},
		{"Sprinkler", "WetGrass"},
	})
	if err != nil {
		t.Fatalf("NewBayesianNetwork() error = %v", err)
	}

	declare(t, bn, "Weather", []string{"cloudy", "rainy", "sunny"})
	declare(t, bn, "Sprinkler", []string{"off", "on"})
	declare(t, bn, "WetGrass", []string{"no", "yes"})

	addCPD(t, bn, "Weather", nil, [][]float64{{0.3, 0.2, 0.5}})
	addCPD(t, bn, "Sprinkler", []string{"Weather"}, [][]float64{
		{0.60, 0.40},
		{0.95, 0.05},
		{0.30, 0.70},
	})
	addCPD(t, bn, "WetGrass", []string{"Sprinkler", "Weather"}, [][]float64{
		{0.90, 0.10},
		{0.20, 0.80},
		{0.95, 0.05},
		{0.10, 0.90},
		{0.01, 0.99},
		{0.10, 0.90},
	})

	if err := bn.CheckModel(); err != nil {
		t.Fatalf("CheckModel() error = %v", err)
	}

	return bn
}

func declare(t *testing.T, bn *BayesianNetwork, variable string, labels []string) {
	t.Helper()
	if err := bn.DeclareStates(variable, labels); err != nil {
		t.Fatalf("DeclareStates(%s) error = %v", variable, err)
	}
}

func addCPD(t *testing.T, bn *BayesianNetwork, variable string, evidence []string, values [][]float64) {
	t.Helper()
	if err := bn.AddCategoricalCPD(variable, evidence, values); err != nil {
		t.Fatalf("AddCategoricalCPD(%s) error = %v", variable, err)
	}
}

func TestBayesianNetwork_AddCategoricalCPD_TakesCardinalityFromStates(t *testing.T) {
	bn := weatherNetwork(t)

	cpd, err := bn.GetCPD("WetGrass")
	if err != nil {
		t.Fatalf("GetCPD() error = %v", err)
	}

	if cpd.VariableCard != 2 {
		t.Errorf("VariableCard = %d, want 2", cpd.VariableCard)
	}
	if cpd.EvidenceCard["Weather"] != 3 {
		t.Errorf("EvidenceCard[Weather] = %d, want 3", cpd.EvidenceCard["Weather"])
	}
	if cpd.EvidenceCard["Sprinkler"] != 2 {
		t.Errorf("EvidenceCard[Sprinkler] = %d, want 2", cpd.EvidenceCard["Sprinkler"])
	}
}

func TestBayesianNetwork_AddCategoricalCPD_UndeclaredStates(t *testing.T) {
	bn, err := NewBayesianNetwork([][2]string{{"A", "B"}})
	if err != nil {
		t.Fatalf("NewBayesianNetwork() error = %v", err)
	}

	if err := bn.AddCategoricalCPD("A", nil, [][]float64{{0.5, 0.5}}); err == nil {
		t.Error("AddCategoricalCPD() error = nil for an undeclared variable, want error")
	}

	declare(t, bn, "B", []string{"no", "yes"})
	if err := bn.AddCategoricalCPD("B", []string{"A"}, [][]float64{{0.5, 0.5}}); err == nil {
		t.Error("AddCategoricalCPD() error = nil for undeclared evidence, want error")
	}
}

func TestBayesianNetwork_AddCategoricalCPD_RejectsMalformedValues(t *testing.T) {
	bn, err := NewBayesianNetwork([][2]string{{"A", "B"}})
	if err != nil {
		t.Fatalf("NewBayesianNetwork() error = %v", err)
	}
	declare(t, bn, "A", []string{"no", "yes"})

	if err := bn.AddCategoricalCPD("A", nil, [][]float64{{0.5, 0.6}}); err == nil {
		t.Error("AddCategoricalCPD() error = nil for rows that do not sum to one, want error")
	}
}

func TestBayesianNetwork_SimulateCategorical_ReturnsKnownLabels(t *testing.T) {
	bn := weatherNetwork(t)

	samples, err := bn.SimulateCategorical(200, 7)
	if err != nil {
		t.Fatalf("SimulateCategorical() error = %v", err)
	}
	if len(samples) != 200 {
		t.Fatalf("SimulateCategorical() returned %d samples, want 200", len(samples))
	}

	for i, sample := range samples {
		for _, variable := range []string{"Weather", "Sprinkler", "WetGrass"} {
			states, _ := bn.StateNames(variable)
			if !states.HasLabel(sample[variable]) {
				t.Fatalf("sample %d has %s=%q, which is not a declared state", i, variable, sample[variable])
			}
		}
	}
}

func TestBayesianNetwork_SimulateCategorical_MatchesTheMarginal(t *testing.T) {
	bn := weatherNetwork(t)

	samples, err := bn.SimulateCategorical(20000, 42)
	if err != nil {
		t.Fatalf("SimulateCategorical() error = %v", err)
	}

	counts := make(map[string]int)
	for _, sample := range samples {
		counts[sample["Weather"]]++
	}

	want := map[string]float64{"cloudy": 0.3, "rainy": 0.2, "sunny": 0.5}
	for label, wanted := range want {
		got := float64(counts[label]) / float64(len(samples))
		if math.Abs(got-wanted) > 0.02 {
			t.Errorf("P(Weather=%s) = %.3f, want %.3f +/- 0.02", label, got, wanted)
		}
	}
}

func TestBayesianNetwork_FitCategorical_RecoversParameters(t *testing.T) {
	truth := weatherNetwork(t)

	samples, err := truth.SimulateCategorical(40000, 11)
	if err != nil {
		t.Fatalf("SimulateCategorical() error = %v", err)
	}

	learned := weatherNetwork(t)
	if err := learned.FitCategorical(samples); err != nil {
		t.Fatalf("FitCategorical() error = %v", err)
	}

	for _, variable := range []string{"Weather", "Sprinkler", "WetGrass"} {
		want, err := truth.GetCPD(variable)
		if err != nil {
			t.Fatalf("GetCPD() error = %v", err)
		}
		got, err := learned.GetCPD(variable)
		if err != nil {
			t.Fatalf("GetCPD() error = %v", err)
		}

		if got.VariableCard != want.VariableCard {
			t.Fatalf("%s VariableCard = %d, want %d", variable, got.VariableCard, want.VariableCard)
		}
		for row := range want.Values {
			for col := range want.Values[row] {
				if math.Abs(got.Values[row][col]-want.Values[row][col]) > 0.03 {
					t.Errorf("%s CPD[%d][%d] = %.3f, want %.3f +/- 0.03",
						variable, row, col, got.Values[row][col], want.Values[row][col])
				}
			}
		}
	}
}

func TestBayesianNetwork_FitCategorical_KeepsADeclaredStateTheDataNeverShows(t *testing.T) {
	bn, err := NewBayesianNetwork([][2]string{{"Weather", "Umbrella"}})
	if err != nil {
		t.Fatalf("NewBayesianNetwork() error = %v", err)
	}
	declare(t, bn, "Weather", []string{"cloudy", "rainy", "sunny"})
	declare(t, bn, "Umbrella", []string{"no", "yes"})

	// The sample never contains "sunny", which is exactly the case where reading
	// the cardinality off the data would drop a state.
	rows := []map[string]string{
		{"Weather": "cloudy", "Umbrella": "no"},
		{"Weather": "rainy", "Umbrella": "yes"},
		{"Weather": "rainy", "Umbrella": "yes"},
		{"Weather": "cloudy", "Umbrella": "no"},
	}

	if err := bn.FitCategorical(rows); err != nil {
		t.Fatalf("FitCategorical() error = %v", err)
	}

	weather, err := bn.GetCPD("Weather")
	if err != nil {
		t.Fatalf("GetCPD() error = %v", err)
	}
	if weather.VariableCard != 3 {
		t.Fatalf("Weather VariableCard = %d, want 3 declared states", weather.VariableCard)
	}
	if weather.Values[0][2] <= 0 {
		t.Errorf("P(Weather=sunny) = %v, want positive mass from smoothing", weather.Values[0][2])
	}

	umbrella, err := bn.GetCPD("Umbrella")
	if err != nil {
		t.Fatalf("GetCPD() error = %v", err)
	}
	if len(umbrella.Values) != 3 {
		t.Errorf("Umbrella CPD has %d rows, want one per declared weather state", len(umbrella.Values))
	}
}

func TestBayesianNetwork_FitCategorical_InfersUndeclaredVariables(t *testing.T) {
	bn, err := NewBayesianNetwork([][2]string{{"Weather", "Umbrella"}})
	if err != nil {
		t.Fatalf("NewBayesianNetwork() error = %v", err)
	}

	rows := []map[string]string{
		{"Weather": "rainy", "Umbrella": "yes"},
		{"Weather": "sunny", "Umbrella": "no"},
		{"Weather": "rainy", "Umbrella": "yes"},
	}

	if err := bn.FitCategorical(rows); err != nil {
		t.Fatalf("FitCategorical() error = %v", err)
	}

	umbrella, ok := bn.StateNames("Umbrella")
	if !ok {
		t.Fatal("StateNames(Umbrella) not found after fitting")
	}
	if got, want := strings.Join(umbrella.Labels(), ","), "no,yes"; got != want {
		t.Errorf("Umbrella labels = %q, want %q", got, want)
	}
}

func TestBayesianNetwork_FitCategorical_RejectsAnUndeclaredLabel(t *testing.T) {
	bn, err := NewBayesianNetwork([][2]string{{"Weather", "Umbrella"}})
	if err != nil {
		t.Fatalf("NewBayesianNetwork() error = %v", err)
	}
	declare(t, bn, "Weather", []string{"cloudy", "rainy"})
	declare(t, bn, "Umbrella", []string{"no", "yes"})

	err = bn.FitCategorical([]map[string]string{{"Weather": "snowy", "Umbrella": "yes"}})
	if err == nil {
		t.Fatal("FitCategorical() error = nil for a label outside the declared states, want error")
	}
	if !strings.Contains(err.Error(), "snowy") {
		t.Errorf("error %q does not name the offending label", err)
	}
}

func TestBayesianNetwork_FitCategorical_NeedsDataForEveryNode(t *testing.T) {
	bn, err := NewBayesianNetwork([][2]string{{"Weather", "Umbrella"}})
	if err != nil {
		t.Fatalf("NewBayesianNetwork() error = %v", err)
	}

	err = bn.FitCategorical([]map[string]string{{"Weather": "rainy"}})
	if err == nil {
		t.Fatal("FitCategorical() error = nil with no data for Umbrella, want error")
	}
	if !strings.Contains(err.Error(), "Umbrella") {
		t.Errorf("error %q does not name the variable without data", err)
	}
}

func TestBayesianNetwork_FitCategorical_TreatsMissingValuesAsUnobserved(t *testing.T) {
	bn, err := NewBayesianNetwork([][2]string{{"Weather", "Umbrella"}})
	if err != nil {
		t.Fatalf("NewBayesianNetwork() error = %v", err)
	}
	declare(t, bn, "Weather", []string{"cloudy", "rainy"})
	declare(t, bn, "Umbrella", []string{"no", "yes"})

	rows := []map[string]string{
		{"Weather": "rainy", "Umbrella": "yes"},
		{"Weather": "cloudy", "Umbrella": categorical.MissingLabel},
		{"Weather": "cloudy"},
	}

	if err := bn.FitCategorical(rows); err != nil {
		t.Fatalf("FitCategorical() error = %v", err)
	}

	// Only the first row observes Umbrella, so the cloudy row of its CPD carries
	// smoothing alone and must stay a valid distribution.
	umbrella, err := bn.GetCPD("Umbrella")
	if err != nil {
		t.Fatalf("GetCPD() error = %v", err)
	}
	for row, probabilities := range umbrella.Values {
		sum := 0.0
		for _, p := range probabilities {
			sum += p
		}
		if math.Abs(sum-1.0) > 1e-9 {
			t.Errorf("Umbrella CPD row %d sums to %v, want 1", row, sum)
		}
	}
}

func TestBayesianNetwork_PredictCategorical_ReturnsLabels(t *testing.T) {
	bn := weatherNetwork(t)

	predictions, err := bn.PredictCategorical([]map[string]string{
		{"Weather": "rainy", "Sprinkler": "off"},
		{"Weather": "sunny", "Sprinkler": "off"},
	})
	if err != nil {
		t.Fatalf("PredictCategorical() error = %v", err)
	}

	wetGrass, ok := predictions["WetGrass"]
	if !ok {
		t.Fatal("PredictCategorical() did not predict WetGrass")
	}
	if len(wetGrass) != 2 {
		t.Fatalf("PredictCategorical() returned %d predictions, want 2", len(wetGrass))
	}
	if wetGrass[0] != "yes" {
		t.Errorf("WetGrass when rainy = %q, want %q", wetGrass[0], "yes")
	}
	if wetGrass[1] != "no" {
		t.Errorf("WetGrass when sunny and the sprinkler is off = %q, want %q", wetGrass[1], "no")
	}
}

func TestBayesianNetwork_PredictCategorical_RejectsUnknownLabels(t *testing.T) {
	bn := weatherNetwork(t)

	if _, err := bn.PredictCategorical([]map[string]string{{"Weather": "snowy"}}); err == nil {
		t.Error("PredictCategorical() error = nil for an unknown label, want error")
	}
}

func TestBayesianNetwork_CategoricalMethodsNeedDeclaredStates(t *testing.T) {
	bn, err := NewBayesianNetwork([][2]string{{"A", "B"}})
	if err != nil {
		t.Fatalf("NewBayesianNetwork() error = %v", err)
	}

	cpdA, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5, 0.5}}, nil, nil)
	if err := bn.AddCPD(cpdA); err != nil {
		t.Fatalf("AddCPD() error = %v", err)
	}
	cpdB, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.5, 0.5}, {0.5, 0.5}},
		[]string{"A"}, map[string]int{"A": 2})
	if err := bn.AddCPD(cpdB); err != nil {
		t.Fatalf("AddCPD() error = %v", err)
	}

	if _, err := bn.SimulateCategorical(10, 1); err == nil {
		t.Error("SimulateCategorical() error = nil without declared states, want error")
	}
	if _, err := bn.PredictCategorical([]map[string]string{{}}); err == nil {
		t.Error("PredictCategorical() error = nil without declared states, want error")
	}
}

func TestBayesianNetwork_ValidateCodebook(t *testing.T) {
	bn := weatherNetwork(t)
	if err := bn.ValidateCodebook(); err != nil {
		t.Fatalf("ValidateCodebook() error = %v for a consistent network", err)
	}

	// Redeclaring Weather with fewer states contradicts the CPDs already added.
	declare(t, bn, "Weather", []string{"cloudy", "rainy"})

	err := bn.ValidateCodebook()
	if err == nil {
		t.Fatal("ValidateCodebook() error = nil after a contradicting declaration, want error")
	}
	if !strings.Contains(err.Error(), "Weather") {
		t.Errorf("error %q does not name the inconsistent variable", err)
	}
}

func TestBayesianNetwork_SetCodebookCopiesTheArgument(t *testing.T) {
	bn, err := NewBayesianNetwork([][2]string{{"A", "B"}})
	if err != nil {
		t.Fatalf("NewBayesianNetwork() error = %v", err)
	}

	external := categorical.NewCodebook()
	if err := external.Declare("A", []string{"no", "yes"}); err != nil {
		t.Fatalf("Declare() error = %v", err)
	}
	if err := bn.SetCodebook(external); err != nil {
		t.Fatalf("SetCodebook() error = %v", err)
	}

	if err := external.Declare("Sneaky", []string{"x", "y"}); err != nil {
		t.Fatalf("Declare() error = %v", err)
	}
	if bn.Codebook().Has("Sneaky") {
		t.Error("SetCodebook() kept a reference to the codebook of the caller")
	}
	if !bn.Codebook().Has("A") {
		t.Error("SetCodebook() lost the declared states")
	}

	if err := bn.SetCodebook(nil); err == nil {
		t.Error("SetCodebook(nil) error = nil, want error")
	}
}

func TestBayesianNetwork_CopyCopiesTheCodebook(t *testing.T) {
	bn := weatherNetwork(t)
	copied := bn.Copy()

	if !copied.Codebook().Has("Weather") {
		t.Fatal("Copy() lost the codebook")
	}

	if err := copied.DeclareStates("Extra", []string{"a", "b"}); err != nil {
		t.Fatalf("DeclareStates() error = %v", err)
	}
	if bn.Codebook().Has("Extra") {
		t.Error("Copy() shares its codebook with the original")
	}
}
