// Package categorical maps the labels of categorical and binary fields onto the
// integer state indices used by the discrete factor algebra.
//
// The rest of bngo represents a discrete variable with cardinality r as the
// integer states 0..r-1. Real data is rarely encoded that way: fields arrive as
// "sunny"/"rainy", "yes"/"no" or "true"/"false". StateNames is the value object
// that fixes an order for those labels, and Codebook groups one StateNames per
// variable so whole rows can be encoded before learning and decoded after
// inference.
package categorical

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// StateNames is an ordered, immutable set of labels for one categorical field.
// The position of a label in the order is the integer state the rest of the
// library uses for it.
type StateNames struct {
	labels  []string
	indexOf map[string]int
}

// NewStateNames builds StateNames from labels in the exact order given, so the
// caller controls which label maps to which state index. It fails on an empty
// set, on an empty label and on duplicates, because each of those makes the
// label-to-index mapping ambiguous.
func NewStateNames(labels []string) (*StateNames, error) {
	if len(labels) == 0 {
		return nil, fmt.Errorf("state names: need at least one label")
	}

	indexOf := make(map[string]int, len(labels))
	ordered := make([]string, len(labels))

	for i, label := range labels {
		if label == "" {
			return nil, fmt.Errorf("state names: label at position %d is empty", i)
		}
		if previous, exists := indexOf[label]; exists {
			return nil, fmt.Errorf("state names: label %q repeats at positions %d and %d", label, previous, i)
		}
		indexOf[label] = i
		ordered[i] = label
	}

	return &StateNames{labels: ordered, indexOf: indexOf}, nil
}

// InferStateNames derives StateNames from observed labels, discarding
// duplicates and ordering what remains by the rules documented on
// CanonicalOrder. Empty labels are rejected: an empty cell means "missing", not
// a state of its own.
func InferStateNames(observed []string) (*StateNames, error) {
	unique := make([]string, 0, len(observed))
	seen := make(map[string]bool, len(observed))

	for _, label := range observed {
		if label == "" {
			return nil, fmt.Errorf("state names: cannot infer states from an empty label")
		}
		if seen[label] {
			continue
		}
		seen[label] = true
		unique = append(unique, label)
	}

	return NewStateNames(CanonicalOrder(unique))
}

// Cardinality returns the number of states.
func (s *StateNames) Cardinality() int {
	return len(s.labels)
}

// Labels returns the labels in state order. The result is a copy, so callers
// cannot disturb the mapping.
func (s *StateNames) Labels() []string {
	labels := make([]string, len(s.labels))
	copy(labels, s.labels)
	return labels
}

// Label returns the label for a state index.
func (s *StateNames) Label(index int) (string, error) {
	if index < 0 || index >= len(s.labels) {
		return "", fmt.Errorf("state names: state %d out of range [0,%d)", index, len(s.labels))
	}
	return s.labels[index], nil
}

// Index returns the state index for a label.
func (s *StateNames) Index(label string) (int, error) {
	index, ok := s.indexOf[label]
	if !ok {
		return 0, fmt.Errorf("state names: unknown label %q, expected one of %v", label, s.labels)
	}
	return index, nil
}

// HasLabel reports whether the label is one of the states.
func (s *StateNames) HasLabel(label string) bool {
	_, ok := s.indexOf[label]
	return ok
}

// IsBinary reports whether the field has exactly two states.
func (s *StateNames) IsBinary() bool {
	return len(s.labels) == 2
}

// Equal reports whether both hold the same labels in the same order.
func (s *StateNames) Equal(other *StateNames) bool {
	if other == nil || len(s.labels) != len(other.labels) {
		return false
	}
	for i, label := range s.labels {
		if other.labels[i] != label {
			return false
		}
	}
	return true
}

// String renders the states as an ordered list.
func (s *StateNames) String() string {
	return "[" + strings.Join(s.labels, " ") + "]"
}

// negativeLabels and positiveLabels are the literals bngo recognises as the two
// poles of a binary field. They exist so that a field written as "yes"/"no"
// lands on state 1 for "yes" instead of following an accident of alphabetical
// order, which would flip the meaning of a coefficient or a CPD row.
var negativeLabels = map[string]bool{
	"false": true, "f": true, "no": true, "n": true, "0": true,
	"off": true, "absent": true, "negative": true, "neg": true,
}

var positiveLabels = map[string]bool{
	"true": true, "t": true, "yes": true, "y": true, "1": true,
	"on": true, "present": true, "positive": true, "pos": true,
}

// CanonicalOrder returns labels in the deterministic order bngo uses when the
// caller has not fixed one, and never modifies its argument. Determinism
// matters: the same data must encode to the same integers on every run, or a
// learned CPD cannot be compared against an earlier one.
//
// The rules, in priority order:
//
//  1. A recognised binary pair ("no"/"yes", "false"/"true", "0"/"1", ...) is
//     ordered negative first, so the positive state is always 1.
//  2. Labels that all parse as numbers are ordered numerically, so "2" sorts
//     before "10" rather than after it.
//  3. Anything else is ordered lexicographically.
func CanonicalOrder(labels []string) []string {
	ordered := make([]string, len(labels))
	copy(ordered, labels)

	if pair, ok := orderBinaryPair(ordered); ok {
		return pair
	}

	if numbers, ok := parseAllAsNumbers(ordered); ok {
		sort.SliceStable(ordered, func(i, j int) bool {
			if numbers[ordered[i]] != numbers[ordered[j]] {
				return numbers[ordered[i]] < numbers[ordered[j]]
			}
			return ordered[i] < ordered[j]
		})
		return ordered
	}

	sort.Strings(ordered)
	return ordered
}

// orderBinaryPair returns the two labels ordered negative-then-positive when
// they form a recognised binary pair.
func orderBinaryPair(labels []string) ([]string, bool) {
	if len(labels) != 2 {
		return nil, false
	}

	first := strings.ToLower(labels[0])
	second := strings.ToLower(labels[1])

	if negativeLabels[first] && positiveLabels[second] {
		return []string{labels[0], labels[1]}, true
	}
	if positiveLabels[first] && negativeLabels[second] {
		return []string{labels[1], labels[0]}, true
	}

	return nil, false
}

// parseAllAsNumbers returns each label's numeric value, and false as soon as one
// label is not a number.
func parseAllAsNumbers(labels []string) (map[string]float64, bool) {
	numbers := make(map[string]float64, len(labels))
	for _, label := range labels {
		value, err := strconv.ParseFloat(label, 64)
		if err != nil {
			return nil, false
		}
		numbers[label] = value
	}
	return numbers, true
}
