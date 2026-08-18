package categorical

import (
	"strings"
	"testing"
)

func TestNewStateNames_PreservesGivenOrder(t *testing.T) {
	states, err := NewStateNames([]string{"high", "low", "medium"})
	if err != nil {
		t.Fatalf("NewStateNames() error = %v", err)
	}

	if got := states.Cardinality(); got != 3 {
		t.Errorf("Cardinality() = %d, want 3", got)
	}

	for index, want := range []string{"high", "low", "medium"} {
		got, err := states.Label(index)
		if err != nil {
			t.Fatalf("Label(%d) error = %v", index, err)
		}
		if got != want {
			t.Errorf("Label(%d) = %q, want %q", index, got, want)
		}
	}
}

func TestNewStateNames_Invalid(t *testing.T) {
	tests := []struct {
		name   string
		labels []string
	}{
		{"no labels", []string{}},
		{"nil labels", nil},
		{"empty label", []string{"yes", ""}},
		{"duplicate label", []string{"yes", "no", "yes"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewStateNames(tt.labels); err == nil {
				t.Errorf("NewStateNames(%v) error = nil, want error", tt.labels)
			}
		})
	}
}

func TestStateNames_LabelsIsACopy(t *testing.T) {
	states, err := NewStateNames([]string{"a", "b"})
	if err != nil {
		t.Fatalf("NewStateNames() error = %v", err)
	}

	labels := states.Labels()
	labels[0] = "mutated"

	if got, _ := states.Label(0); got != "a" {
		t.Errorf("Label(0) = %q after mutating the returned slice, want %q", got, "a")
	}
}

func TestStateNames_IndexRoundTrip(t *testing.T) {
	states, err := NewStateNames([]string{"sunny", "rainy", "cloudy"})
	if err != nil {
		t.Fatalf("NewStateNames() error = %v", err)
	}

	for _, label := range states.Labels() {
		index, err := states.Index(label)
		if err != nil {
			t.Fatalf("Index(%q) error = %v", label, err)
		}
		back, err := states.Label(index)
		if err != nil {
			t.Fatalf("Label(%d) error = %v", index, err)
		}
		if back != label {
			t.Errorf("round trip of %q gave %q", label, back)
		}
	}
}

func TestStateNames_IndexUnknownLabelListsExpected(t *testing.T) {
	states, _ := NewStateNames([]string{"sunny", "rainy"})

	_, err := states.Index("snowy")
	if err == nil {
		t.Fatal("Index() error = nil, want error")
	}
	if !strings.Contains(err.Error(), "sunny") {
		t.Errorf("error %q does not report the expected labels", err)
	}
}

func TestStateNames_LabelOutOfRange(t *testing.T) {
	states, _ := NewStateNames([]string{"sunny", "rainy"})

	for _, index := range []int{-1, 2} {
		if _, err := states.Label(index); err == nil {
			t.Errorf("Label(%d) error = nil, want error", index)
		}
	}
}

func TestStateNames_IsBinary(t *testing.T) {
	binary, _ := NewStateNames([]string{"no", "yes"})
	if !binary.IsBinary() {
		t.Error("IsBinary() = false for a two state field")
	}

	ternary, _ := NewStateNames([]string{"low", "mid", "high"})
	if ternary.IsBinary() {
		t.Error("IsBinary() = true for a three state field")
	}
}

func TestStateNames_Equal(t *testing.T) {
	base, _ := NewStateNames([]string{"no", "yes"})
	same, _ := NewStateNames([]string{"no", "yes"})
	reordered, _ := NewStateNames([]string{"yes", "no"})
	longer, _ := NewStateNames([]string{"no", "yes", "maybe"})

	if !base.Equal(same) {
		t.Error("Equal() = false for identical states")
	}
	if base.Equal(reordered) {
		t.Error("Equal() = true for states in a different order")
	}
	if base.Equal(longer) {
		t.Error("Equal() = true for states of different length")
	}
	if base.Equal(nil) {
		t.Error("Equal(nil) = true")
	}
}

func TestStateNames_String(t *testing.T) {
	states, _ := NewStateNames([]string{"no", "yes"})
	if got, want := states.String(), "[no yes]"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
}

func TestCanonicalOrder(t *testing.T) {
	tests := []struct {
		name   string
		labels []string
		want   []string
	}{
		{"binary no yes", []string{"yes", "no"}, []string{"no", "yes"}},
		{"binary already ordered", []string{"no", "yes"}, []string{"no", "yes"}},
		{"binary true false", []string{"true", "false"}, []string{"false", "true"}},
		{"binary mixed case", []string{"YES", "No"}, []string{"No", "YES"}},
		{"binary single letters", []string{"y", "n"}, []string{"n", "y"}},
		{"binary on off", []string{"on", "off"}, []string{"off", "on"}},
		{"binary present absent", []string{"present", "absent"}, []string{"absent", "present"}},
		{"binary digits", []string{"1", "0"}, []string{"0", "1"}},
		{"numeric beats lexicographic", []string{"10", "2", "1"}, []string{"1", "2", "10"}},
		{"negative numbers", []string{"5", "-1", "0"}, []string{"-1", "0", "5"}},
		{"decimals", []string{"1.5", "1.25", "10.5"}, []string{"1.25", "1.5", "10.5"}},
		{"lexicographic fallback", []string{"medium", "high", "low"}, []string{"high", "low", "medium"}},
		{"two non boolean labels stay lexicographic", []string{"dog", "cat"}, []string{"cat", "dog"}},
		{"two negatives are not a pair", []string{"no", "n"}, []string{"n", "no"}},
		{"mixed numeric and text falls back", []string{"2", "many"}, []string{"2", "many"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CanonicalOrder(tt.labels)
			if len(got) != len(tt.want) {
				t.Fatalf("CanonicalOrder() = %v, want %v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("CanonicalOrder() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestCanonicalOrder_DoesNotModifyInput(t *testing.T) {
	labels := []string{"yes", "no"}
	CanonicalOrder(labels)

	if labels[0] != "yes" || labels[1] != "no" {
		t.Errorf("CanonicalOrder() modified its argument: %v", labels)
	}
}

func TestInferStateNames_DropsDuplicates(t *testing.T) {
	states, err := InferStateNames([]string{"yes", "no", "yes", "no", "yes"})
	if err != nil {
		t.Fatalf("InferStateNames() error = %v", err)
	}

	if got := states.Cardinality(); got != 2 {
		t.Fatalf("Cardinality() = %d, want 2", got)
	}
	if got, _ := states.Label(0); got != "no" {
		t.Errorf("Label(0) = %q, want %q", got, "no")
	}
}

func TestInferStateNames_RejectsMissingLabel(t *testing.T) {
	if _, err := InferStateNames([]string{"yes", ""}); err == nil {
		t.Error("InferStateNames() error = nil for an empty label, want error")
	}
}

func TestStateNames_HasLabel(t *testing.T) {
	states, _ := NewStateNames([]string{"sunny", "rainy"})

	if !states.HasLabel("sunny") {
		t.Error("HasLabel(sunny) = false")
	}
	if states.HasLabel("snowy") {
		t.Error("HasLabel(snowy) = true")
	}
}
