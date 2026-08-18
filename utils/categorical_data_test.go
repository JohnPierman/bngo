package utils

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/JohnPierman/bngo/categorical"
)

func writeCSV(t *testing.T, name, content string) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	return path
}

func TestLoadCategoricalCSV_ReadsLabels(t *testing.T) {
	path := writeCSV(t, "weather.csv", "Weather,Umbrella\nrainy,yes\nsunny,no\ncloudy,no\n")

	frame, err := LoadCategoricalCSV(path)
	if err != nil {
		t.Fatalf("LoadCategoricalCSV() error = %v", err)
	}

	if got, want := strings.Join(frame.Columns, ","), "Weather,Umbrella"; got != want {
		t.Errorf("Columns = %q, want %q", got, want)
	}
	if frame.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", frame.Len())
	}
	if frame.Rows[0]["Weather"] != "rainy" || frame.Rows[0]["Umbrella"] != "yes" {
		t.Errorf("first row = %v", frame.Rows[0])
	}
	if got, want := strings.Join(frame.Column("Weather"), ","), "rainy,sunny,cloudy"; got != want {
		t.Errorf("Column(Weather) = %q, want %q", got, want)
	}
}

func TestLoadCategoricalCSV_EmptyFieldIsMissing(t *testing.T) {
	path := writeCSV(t, "gaps.csv", "A,B\nx,\n,y\n")

	frame, err := LoadCategoricalCSV(path)
	if err != nil {
		t.Fatalf("LoadCategoricalCSV() error = %v", err)
	}

	if _, present := frame.Rows[0]["B"]; present {
		t.Error("an empty field became a state instead of a missing value")
	}
	if _, present := frame.Rows[1]["A"]; present {
		t.Error("an empty field became a state instead of a missing value")
	}
	if frame.Rows[0]["A"] != "x" {
		t.Errorf("row 0 A = %q, want %q", frame.Rows[0]["A"], "x")
	}
}

func TestLoadCategoricalCSV_KeepsNAByDefault(t *testing.T) {
	// NA is a real label in some data sets, so it must survive unless the caller
	// says otherwise.
	path := writeCSV(t, "na.csv", "Country\nNA\nCA\n")

	frame, err := LoadCategoricalCSV(path)
	if err != nil {
		t.Fatalf("LoadCategoricalCSV() error = %v", err)
	}
	if frame.Rows[0]["Country"] != "NA" {
		t.Errorf("row 0 Country = %q, want %q", frame.Rows[0]["Country"], "NA")
	}

	options := DefaultCSVOptions()
	options.MissingMarkers = []string{"", "NA"}
	frame, err = LoadCategoricalCSVWithOptions(path, options)
	if err != nil {
		t.Fatalf("LoadCategoricalCSVWithOptions() error = %v", err)
	}
	if _, present := frame.Rows[0]["Country"]; present {
		t.Error("NA was not treated as missing once it was configured as a marker")
	}
}

func TestLoadCategoricalCSVWithOptions_Delimiter(t *testing.T) {
	path := writeCSV(t, "semi.csv", "A;B\nx;y\n")

	options := DefaultCSVOptions()
	options.Delimiter = ';'

	frame, err := LoadCategoricalCSVWithOptions(path, options)
	if err != nil {
		t.Fatalf("LoadCategoricalCSVWithOptions() error = %v", err)
	}
	if frame.Rows[0]["B"] != "y" {
		t.Errorf("row 0 B = %q, want %q", frame.Rows[0]["B"], "y")
	}
}

func TestLoadCategoricalCSV_Errors(t *testing.T) {
	if _, err := LoadCategoricalCSV(filepath.Join(t.TempDir(), "absent.csv")); err == nil {
		t.Error("LoadCategoricalCSV() error = nil for a missing file, want error")
	}

	empty := writeCSV(t, "empty.csv", "")
	if _, err := LoadCategoricalCSV(empty); err == nil {
		t.Error("LoadCategoricalCSV() error = nil for a file with no header, want error")
	}

	ragged := writeCSV(t, "ragged.csv", "A,B\nx,y\nz\n")
	if _, err := LoadCategoricalCSV(ragged); err == nil {
		t.Error("LoadCategoricalCSV() error = nil for a row with too few fields, want error")
	}
}

func TestLoadCategoricalCSV_ReportsTheFailingLine(t *testing.T) {
	ragged := writeCSV(t, "ragged.csv", "A,B\nx,y\nz\n")

	_, err := LoadCategoricalCSV(ragged)
	if err == nil {
		t.Fatal("LoadCategoricalCSV() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "line 3") {
		t.Errorf("error %q does not name the failing line", err)
	}
}

func TestCategoricalFrame_EncodeInfersACodebook(t *testing.T) {
	path := writeCSV(t, "weather.csv", "Weather,Umbrella\nrainy,yes\nsunny,no\ncloudy,no\n")

	frame, err := LoadCategoricalCSV(path)
	if err != nil {
		t.Fatalf("LoadCategoricalCSV() error = %v", err)
	}

	encoded, codebook, err := frame.Encode()
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}

	if len(encoded) != 3 {
		t.Fatalf("Encode() returned %d rows, want 3", len(encoded))
	}

	// Umbrella is binary, so "no" must be state 0 and "yes" state 1.
	umbrella, ok := codebook.Get("Umbrella")
	if !ok {
		t.Fatal("codebook is missing Umbrella")
	}
	if got, want := strings.Join(umbrella.Labels(), ","), "no,yes"; got != want {
		t.Errorf("Umbrella labels = %q, want %q", got, want)
	}
	if encoded[0]["Umbrella"] != 1 {
		t.Errorf("encoded yes as %d, want 1", encoded[0]["Umbrella"])
	}
}

func TestCategoricalFrame_EncodeWithAnExistingCodebook(t *testing.T) {
	frame := NewCategoricalFrame([]string{"Grade"})
	frame.AddRow(map[string]string{"Grade": "C"})

	codebook := categorical.NewCodebook()
	if err := codebook.Declare("Grade", []string{"A", "B", "C"}); err != nil {
		t.Fatalf("Declare() error = %v", err)
	}

	encoded, err := frame.EncodeWith(codebook)
	if err != nil {
		t.Fatalf("EncodeWith() error = %v", err)
	}
	if encoded[0]["Grade"] != 2 {
		t.Errorf("encoded C as %d, want 2", encoded[0]["Grade"])
	}

	if _, err := frame.EncodeWith(nil); err == nil {
		t.Error("EncodeWith(nil) error = nil, want error")
	}

	frame.AddRow(map[string]string{"Grade": "F"})
	if _, err := frame.EncodeWith(codebook); err == nil {
		t.Error("EncodeWith() error = nil for a label outside the codebook, want error")
	}
}

func TestNewCategoricalFrameFromRows_RoundTrips(t *testing.T) {
	original := NewCategoricalFrame([]string{"Weather", "Umbrella"})
	original.AddRow(map[string]string{"Weather": "rainy", "Umbrella": "yes"})
	original.AddRow(map[string]string{"Weather": "sunny", "Umbrella": "no"})

	encoded, codebook, err := original.Encode()
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}

	decoded, err := NewCategoricalFrameFromRows(encoded, codebook, original.Columns)
	if err != nil {
		t.Fatalf("NewCategoricalFrameFromRows() error = %v", err)
	}

	if decoded.Len() != original.Len() {
		t.Fatalf("round trip produced %d rows, want %d", decoded.Len(), original.Len())
	}
	for i, row := range original.Rows {
		for variable, label := range row {
			if decoded.Rows[i][variable] != label {
				t.Errorf("row %d %s = %q, want %q", i, variable, decoded.Rows[i][variable], label)
			}
		}
	}

	if _, err := NewCategoricalFrameFromRows(encoded, nil, nil); err == nil {
		t.Error("NewCategoricalFrameFromRows(nil codebook) error = nil, want error")
	}

	defaulted, err := NewCategoricalFrameFromRows(encoded, codebook, nil)
	if err != nil {
		t.Fatalf("NewCategoricalFrameFromRows() error = %v", err)
	}
	if got, want := strings.Join(defaulted.Columns, ","), "Umbrella,Weather"; got != want {
		t.Errorf("default columns = %q, want %q", got, want)
	}

	if _, err := NewCategoricalFrameFromRows([]map[string]int{{"Weather": 9}}, codebook, nil); err == nil {
		t.Error("NewCategoricalFrameFromRows() error = nil for an out of range state, want error")
	}
}

func TestCategoricalFrame_SaveCSVRoundTrips(t *testing.T) {
	frame := NewCategoricalFrame([]string{"Weather", "Umbrella"})
	frame.AddRow(map[string]string{"Weather": "rainy", "Umbrella": "yes"})
	frame.AddRow(map[string]string{"Weather": "sunny"})

	path := filepath.Join(t.TempDir(), "out.csv")
	if err := frame.SaveCSV(path); err != nil {
		t.Fatalf("SaveCSV() error = %v", err)
	}

	reloaded, err := LoadCategoricalCSV(path)
	if err != nil {
		t.Fatalf("LoadCategoricalCSV() error = %v", err)
	}

	if reloaded.Len() != 2 {
		t.Fatalf("reloaded %d rows, want 2", reloaded.Len())
	}
	if reloaded.Rows[0]["Umbrella"] != "yes" {
		t.Errorf("row 0 Umbrella = %q, want %q", reloaded.Rows[0]["Umbrella"], "yes")
	}
	if _, present := reloaded.Rows[1]["Umbrella"]; present {
		t.Error("a value that was never observed came back as a state")
	}
}

func TestCategoricalFrame_SaveCSVReportsAnUnwritablePath(t *testing.T) {
	frame := NewCategoricalFrame([]string{"A"})

	if err := frame.SaveCSV(filepath.Join(t.TempDir(), "missing-dir", "out.csv")); err == nil {
		t.Error("SaveCSV() error = nil for an unwritable path, want error")
	}
}

func TestNewCategoricalFrame_CopiesItsColumns(t *testing.T) {
	columns := []string{"A", "B"}
	frame := NewCategoricalFrame(columns)
	columns[0] = "mutated"

	if frame.Columns[0] != "A" {
		t.Errorf("NewCategoricalFrame() kept a reference to the argument: %v", frame.Columns)
	}
}
