// Package factors provides factor and CPD implementations
package factors

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// DiscreteFactor represents a discrete factor (potential function)
type DiscreteFactor struct {
	Variables   []string
	Cardinality map[string]int
	Values      []float64
}

// NewDiscreteFactor creates a new discrete factor
func NewDiscreteFactor(variables []string, cardinality map[string]int, values []float64) (*DiscreteFactor, error) {
	// Calculate expected size
	size := 1
	for _, v := range variables {
		size *= cardinality[v]
	}

	if len(values) != size {
		return nil, fmt.Errorf("values length %d does not match expected size %d", len(values), size)
	}

	return &DiscreteFactor{
		Variables:   variables,
		Cardinality: cardinality,
		Values:      values,
	}, nil
}

// Copy creates a deep copy of the factor
func (f *DiscreteFactor) Copy() *DiscreteFactor {
	cardCopy := make(map[string]int)
	for k, v := range f.Cardinality {
		cardCopy[k] = v
	}

	valuesCopy := make([]float64, len(f.Values))
	copy(valuesCopy, f.Values)

	varsCopy := make([]string, len(f.Variables))
	copy(varsCopy, f.Variables)

	return &DiscreteFactor{
		Variables:   varsCopy,
		Cardinality: cardCopy,
		Values:      valuesCopy,
	}
}

// Multiply multiplies this factor with another factor
func (f *DiscreteFactor) Multiply(other *DiscreteFactor) (*DiscreteFactor, error) {
	// Find union of variables
	varSet := make(map[string]bool)
	for _, v := range f.Variables {
		varSet[v] = true
	}
	for _, v := range other.Variables {
		varSet[v] = true
	}

	newVars := make([]string, 0, len(varSet))
	for v := range varSet {
		newVars = append(newVars, v)
	}
	sort.Strings(newVars)

	// Merge cardinality
	newCard := make(map[string]int)
	for k, v := range f.Cardinality {
		newCard[k] = v
	}
	for k, v := range other.Cardinality {
		if existing, ok := newCard[k]; ok && existing != v {
			return nil, fmt.Errorf("cardinality mismatch for variable %s", k)
		}
		newCard[k] = v
	}

	// Calculate new values
	size := 1
	for _, v := range newVars {
		size *= newCard[v]
	}
	newValues := make([]float64, size)

	// For each assignment of the new variables
	assignment := make(map[string]int)
	f.multiplyHelper(0, newVars, assignment, newCard, other, newValues)

	return NewDiscreteFactor(newVars, newCard, newValues)
}

func (f *DiscreteFactor) multiplyHelper(depth int, vars []string, assignment map[string]int,
	cardinality map[string]int, other *DiscreteFactor, result []float64) {
	if depth == len(vars) {
		idx := f.assignmentToIndex(vars, assignment, cardinality)
		idx1 := f.projectAssignmentToIndex(assignment)
		idx2 := other.projectAssignmentToIndex(assignment)
		result[idx] = f.Values[idx1] * other.Values[idx2]
		return
	}

	v := vars[depth]
	for i := 0; i < cardinality[v]; i++ {
		assignment[v] = i
		f.multiplyHelper(depth+1, vars, assignment, cardinality, other, result)
	}
}

func (f *DiscreteFactor) projectAssignmentToIndex(assignment map[string]int) int {
	idx := 0
	stride := 1
	for i := len(f.Variables) - 1; i >= 0; i-- {
		v := f.Variables[i]
		idx += assignment[v] * stride
		stride *= f.Cardinality[v]
	}
	return idx
}

func (f *DiscreteFactor) assignmentToIndex(vars []string, assignment map[string]int,
	cardinality map[string]int) int {
	idx := 0
	stride := 1
	for i := len(vars) - 1; i >= 0; i-- {
		v := vars[i]
		idx += assignment[v] * stride
		stride *= cardinality[v]
	}
	return idx
}

// Marginalize sums out variables from the factor
func (f *DiscreteFactor) Marginalize(variables []string) (*DiscreteFactor, error) {
	// Find remaining variables
	toRemove := make(map[string]bool)
	for _, v := range variables {
		toRemove[v] = true
	}

	newVars := make([]string, 0)
	for _, v := range f.Variables {
		if !toRemove[v] {
			newVars = append(newVars, v)
		}
	}

	if len(newVars) == 0 {
		// Sum all values
		sum := 0.0
		for _, v := range f.Values {
			sum += v
		}
		return NewDiscreteFactor([]string{}, map[string]int{}, []float64{sum})
	}

	// Create new cardinality
	newCard := make(map[string]int)
	for _, v := range newVars {
		newCard[v] = f.Cardinality[v]
	}

	// Calculate new size
	size := 1
	for _, v := range newVars {
		size *= newCard[v]
	}
	newValues := make([]float64, size)

	// Sum over all assignments
	assignment := make(map[string]int)
	f.marginalizeHelper(0, f.Variables, assignment, newVars, newCard, newValues)

	return NewDiscreteFactor(newVars, newCard, newValues)
}

func (f *DiscreteFactor) marginalizeHelper(depth int, vars []string, assignment map[string]int,
	newVars []string, newCard map[string]int, result []float64) {
	if depth == len(vars) {
		oldIdx := f.projectAssignmentToIndex(assignment)
		newIdx := f.assignmentToIndex(newVars, assignment, newCard)
		result[newIdx] += f.Values[oldIdx]
		return
	}

	v := vars[depth]
	for i := 0; i < f.Cardinality[v]; i++ {
		assignment[v] = i
		f.marginalizeHelper(depth+1, vars, assignment, newVars, newCard, result)
	}
}

// Reduce reduces the factor by fixing certain variables to specific values
func (f *DiscreteFactor) Reduce(evidence map[string]int) (*DiscreteFactor, error) {
	// Check if evidence variables are in the factor
	for v := range evidence {
		found := false
		for _, fv := range f.Variables {
			if fv == v {
				found = true
				break
			}
		}
		if !found {
			continue // Skip evidence not in this factor
		}
	}

	// Find remaining variables
	newVars := make([]string, 0)
	for _, v := range f.Variables {
		if _, ok := evidence[v]; !ok {
			newVars = append(newVars, v)
		}
	}

	if len(newVars) == 0 {
		// All variables are evidence
		idx := f.projectAssignmentToIndex(evidence)
		return NewDiscreteFactor([]string{}, map[string]int{}, []float64{f.Values[idx]})
	}

	// Create new cardinality
	newCard := make(map[string]int)
	for _, v := range newVars {
		newCard[v] = f.Cardinality[v]
	}

	// Calculate new size
	size := 1
	for _, v := range newVars {
		size *= newCard[v]
	}
	newValues := make([]float64, size)

	// Extract values matching evidence
	assignment := make(map[string]int)
	for k, v := range evidence {
		assignment[k] = v
	}
	f.reduceHelper(0, newVars, assignment, newCard, newValues)

	return NewDiscreteFactor(newVars, newCard, newValues)
}

func (f *DiscreteFactor) reduceHelper(depth int, vars []string, assignment map[string]int,
	newCard map[string]int, result []float64) {
	if depth == len(vars) {
		oldIdx := f.projectAssignmentToIndex(assignment)
		newIdx := f.assignmentToIndex(vars, assignment, newCard)
		result[newIdx] = f.Values[oldIdx]
		return
	}

	v := vars[depth]
	for i := 0; i < newCard[v]; i++ {
		assignment[v] = i
		f.reduceHelper(depth+1, vars, assignment, newCard, result)
	}
}

// Normalize normalizes the factor so it sums to 1
func (f *DiscreteFactor) Normalize() error {
	sum := 0.0
	for _, v := range f.Values {
		sum += v
	}

	if sum == 0 {
		return fmt.Errorf("cannot normalize factor with sum 0")
	}

	for i := range f.Values {
		f.Values[i] /= sum
	}

	return nil
}

// String returns a string representation of the factor
func (f *DiscreteFactor) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Factor(%s)\n", strings.Join(f.Variables, ", ")))

	assignment := make(map[string]int)
	f.stringHelper(0, assignment, &sb)

	return sb.String()
}

func (f *DiscreteFactor) stringHelper(depth int, assignment map[string]int, sb *strings.Builder) {
	if depth == len(f.Variables) {
		idx := f.projectAssignmentToIndex(assignment)
		sb.WriteString(fmt.Sprintf("  "))
		for _, v := range f.Variables {
			sb.WriteString(fmt.Sprintf("%s=%d ", v, assignment[v]))
		}
		sb.WriteString(fmt.Sprintf("-> %.4f\n", f.Values[idx]))
		return
	}

	v := f.Variables[depth]
	for i := 0; i < f.Cardinality[v]; i++ {
		assignment[v] = i
		f.stringHelper(depth+1, assignment, sb)
	}
}

// MaxMarginalize returns the maximum value over the marginalized variables
func (f *DiscreteFactor) MaxMarginalize(variables []string) (*DiscreteFactor, error) {
	// Find remaining variables
	toRemove := make(map[string]bool)
	for _, v := range variables {
		toRemove[v] = true
	}

	newVars := make([]string, 0)
	for _, v := range f.Variables {
		if !toRemove[v] {
			newVars = append(newVars, v)
		}
	}

	if len(newVars) == 0 {
		// Find max value
		maxVal := math.Inf(-1)
		for _, v := range f.Values {
			if v > maxVal {
				maxVal = v
			}
		}
		return NewDiscreteFactor([]string{}, map[string]int{}, []float64{maxVal})
	}

	// Create new cardinality
	newCard := make(map[string]int)
	for _, v := range newVars {
		newCard[v] = f.Cardinality[v]
	}

	// Calculate new size
	size := 1
	for _, v := range newVars {
		size *= newCard[v]
	}
	newValues := make([]float64, size)
	for i := range newValues {
		newValues[i] = math.Inf(-1)
	}

	// Take max over all assignments
	assignment := make(map[string]int)
	f.maxMarginalizeHelper(0, f.Variables, assignment, newVars, newCard, newValues)

	return NewDiscreteFactor(newVars, newCard, newValues)
}

func (f *DiscreteFactor) maxMarginalizeHelper(depth int, vars []string, assignment map[string]int,
	newVars []string, newCard map[string]int, result []float64) {
	if depth == len(vars) {
		oldIdx := f.projectAssignmentToIndex(assignment)
		newIdx := f.assignmentToIndex(newVars, assignment, newCard)
		if f.Values[oldIdx] > result[newIdx] {
			result[newIdx] = f.Values[oldIdx]
		}
		return
	}

	v := vars[depth]
	for i := 0; i < f.Cardinality[v]; i++ {
		assignment[v] = i
		f.maxMarginalizeHelper(depth+1, vars, assignment, newVars, newCard, result)
	}
}
