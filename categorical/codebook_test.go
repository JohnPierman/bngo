package categorical

import (
	"strings"
	"testing"
)

func labelledRows() []map[string]string {
	return []map[string]string{
		{"Weather": "sunny", "Umbrella": "no"},
		{"Weather": "rainy", "Umbrella": "yes"},
		{"Weather": "cloudy", "Umbrella": "no"},
		{"Weather": "rainy", "Umbrella": "yes"},
	}
}

func TestInferCodebook_OrdersEachVariable(t *testing.T) {
	codebook, err := InferCodebook(labelledRows())
	if err != nil {
		t.Fatalf("InferCodebook() error = %v", err)
	}

	weather, ok := codebook.Get("Weather")
	if !ok {
		t.Fatal("Get(Weather) not found")
	}
	if got, want := strings.Join(weather.Labels(), ","), "cloudy,rainy,sunny"; got != want {
		t.Errorf("Weather labels = %q, want %q", got, want)
	}

	umbrella, ok := codebook.Get("Umbrella")
	if !ok {
		t.Fatal("Get(Umbrella) not found")
	}
	if got, want := strings.Join(umbrella.Labels(), ","), "no,yes"; got != want {
		t.Errorf("Umbrella labels = %q, want %q", got, want)
	}
}

func TestInferCodebook_IgnoresMissingValues(t *testing.T) {
	codebook, err := InferCodebook([]map[string]string{
		{"Flag": "yes"},
		{"Flag": MissingLabel},
		{"Flag": "no"},
	})
	if err != nil {
		t.Fatalf("InferCodebook() error = %v", err)
	}

	if got, _ := codebook.Cardinality("Flag"); got != 2 {
		t.Errorf("Cardinality(Flag) = %d, want 2", got)
	}
}

func TestCodebook_DeclareFixesTheOrder(t *testing.T) {
	codebook := NewCodebook()
	if err := codebook.Declare("Grade", []string{"A", "B", "C"}); err != nil {
		t.Fatalf("Declare() error = %v", err)
	}

	index, err := codebook.EncodeValue("Grade", "C")
	if err != nil {
		t.Fatalf("EncodeValue() error = %v", err)
	}
	if index != 2 {
		t.Errorf("EncodeValue(Grade, C) = %d, want 2", index)
	}
}

func TestCodebook_DeclareRejectsInvalidLabels(t *testing.T) {
	codebook := NewCodebook()
	if err := codebook.Declare("Grade", []string{"A", "A"}); err == nil {
		t.Error("Declare() error = nil for duplicate labels, want error")
	}
}

func TestCodebook_SetInvalid(t *testing.T) {
	codebook := NewCodebook()
	states, _ := NewStateNames([]string{"a", "b"})

	if err := codebook.Set("", states); err == nil {
		t.Error("Set() error = nil for an empty variable name, want error")
	}
	if err := codebook.Set("X", nil); err == nil {
		t.Error("Set() error = nil for nil states, want error")
	}
}

func TestCodebook_HasAndVariablesAreSorted(t *testing.T) {
	codebook, _ := InferCodebook(labelledRows())

	if !codebook.Has("Weather") {
		t.Error("Has(Weather) = false")
	}
	if codebook.Has("Nope") {
		t.Error("Has(Nope) = true")
	}

	if got, want := strings.Join(codebook.Variables(), ","), "Umbrella,Weather"; got != want {
		t.Errorf("Variables() = %q, want %q", got, want)
	}
}

func TestCodebook_Cardinalities(t *testing.T) {
	codebook, _ := InferCodebook(labelledRows())

	cardinalities := codebook.Cardinalities()
	if cardinalities["Weather"] != 3 || cardinalities["Umbrella"] != 2 {
		t.Errorf("Cardinalities() = %v, want Weather=3 Umbrella=2", cardinalities)
	}

	if _, ok := codebook.Cardinality("Nope"); ok {
		t.Error("Cardinality(Nope) reported ok for an unknown variable")
	}
}

func TestCodebook_EncodeRowsThenDecodeRowsRoundTrips(t *testing.T) {
	rows := labelledRows()
	codebook, _ := InferCodebook(rows)

	encoded, err := codebook.EncodeRows(rows)
	if err != nil {
		t.Fatalf("EncodeRows() error = %v", err)
	}

	decoded, err := codebook.DecodeRows(encoded)
	if err != nil {
		t.Fatalf("DecodeRows() error = %v", err)
	}

	for i, want := range rows {
		for variable, label := range want {
			if decoded[i][variable] != label {
				t.Errorf("row %d %s = %q, want %q", i, variable, decoded[i][variable], label)
			}
		}
	}
}

func TestCodebook_EncodeRowKeepsMissingValuesMissing(t *testing.T) {
	codebook, _ := InferCodebook(labelledRows())

	encoded, err := codebook.EncodeRow(map[string]string{"Weather": "sunny", "Umbrella": MissingLabel})
	if err != nil {
		t.Fatalf("EncodeRow() error = %v", err)
	}

	if _, present := encoded["Umbrella"]; present {
		t.Error("EncodeRow() invented a state for a missing value")
	}
	if len(encoded) != 1 {
		t.Errorf("EncodeRow() = %v, want only Weather", encoded)
	}
}

func TestCodebook_EncodeRowDoesNotModifyItsArgument(t *testing.T) {
	codebook, _ := InferCodebook(labelledRows())
	row := map[string]string{"Weather": "sunny"}

	if _, err := codebook.EncodeRow(row); err != nil {
		t.Fatalf("EncodeRow() error = %v", err)
	}
	if len(row) != 1 || row["Weather"] != "sunny" {
		t.Errorf("EncodeRow() modified its argument: %v", row)
	}
}

func TestCodebook_EncodeRejectsUnknownInput(t *testing.T) {
	codebook, _ := InferCodebook(labelledRows())

	if _, err := codebook.EncodeRow(map[string]string{"Nope": "x"}); err == nil {
		t.Error("EncodeRow() error = nil for an unknown variable, want error")
	}
	if _, err := codebook.EncodeRow(map[string]string{"Weather": "snowy"}); err == nil {
		t.Error("EncodeRow() error = nil for an unknown label, want error")
	}
}

func TestCodebook_EncodeRowsReportsTheFailingRow(t *testing.T) {
	codebook, _ := InferCodebook(labelledRows())

	_, err := codebook.EncodeRows([]map[string]string{
		{"Weather": "sunny"},
		{"Weather": "snowy"},
	})
	if err == nil {
		t.Fatal("EncodeRows() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "row 1") {
		t.Errorf("error %q does not name the failing row", err)
	}
}

func TestCodebook_DecodeRejectsOutOfRangeState(t *testing.T) {
	codebook, _ := InferCodebook(labelledRows())

	if _, err := codebook.DecodeRow(map[string]int{"Umbrella": 7}); err == nil {
		t.Error("DecodeRow() error = nil for an out of range state, want error")
	}
	if _, err := codebook.DecodeRow(map[string]int{"Nope": 0}); err == nil {
		t.Error("DecodeRow() error = nil for an unknown variable, want error")
	}
	if _, err := codebook.DecodeRows([]map[string]int{{"Nope": 0}}); err == nil {
		t.Error("DecodeRows() error = nil for an unknown variable, want error")
	}
}

func TestCodebook_DecodeValueAndEncodeValueErrors(t *testing.T) {
	codebook, _ := InferCodebook(labelledRows())

	if _, err := codebook.EncodeValue("Nope", "x"); err == nil {
		t.Error("EncodeValue() error = nil for an unknown variable, want error")
	}
	if _, err := codebook.DecodeValue("Nope", 0); err == nil {
		t.Error("DecodeValue() error = nil for an unknown variable, want error")
	}
}

func TestCodebook_CopyIsIndependent(t *testing.T) {
	original, _ := InferCodebook(labelledRows())
	copied := original.Copy()

	if err := copied.Declare("Extra", []string{"a", "b"}); err != nil {
		t.Fatalf("Declare() error = %v", err)
	}

	if original.Has("Extra") {
		t.Error("Copy() shares state with the original")
	}
	if !copied.Has("Weather") {
		t.Error("Copy() lost an existing variable")
	}
}
