package factors

import (
	"fmt"
	"sort"
)

// TabularCPD represents a Conditional Probability Distribution in tabular form
type TabularCPD struct {
	Variable     string
	VariableCard int
	Evidence     []string
	EvidenceCard map[string]int
	Values       [][]float64 // [evidence_combination][variable_state]
}

// NewTabularCPD creates a new tabular CPD
func NewTabularCPD(variable string, variableCard int, values [][]float64,
	evidence []string, evidenceCard map[string]int) (*TabularCPD, error) {

	// Calculate expected number of rows
	expectedRows := 1
	for _, e := range evidence {
		expectedRows *= evidenceCard[e]
	}

	if len(values) != expectedRows {
		return nil, fmt.Errorf("values has %d rows, expected %d", len(values), expectedRows)
	}

	// Check each row
	for i, row := range values {
		if len(row) != variableCard {
			return nil, fmt.Errorf("row %d has %d columns, expected %d", i, len(row), variableCard)
		}

		// Check if probabilities sum to 1
		sum := 0.0
		for _, p := range row {
			sum += p
		}
		if sum < 0.999 || sum > 1.001 {
			return nil, fmt.Errorf("row %d probabilities sum to %f, expected 1.0", i, sum)
		}
	}

	return &TabularCPD{
		Variable:     variable,
		VariableCard: variableCard,
		Evidence:     evidence,
		EvidenceCard: evidenceCard,
		Values:       values,
	}, nil
}

// ToFactor converts the CPD to a factor
func (cpd *TabularCPD) ToFactor() (*DiscreteFactor, error) {
	// Variables are [evidence..., variable] in sorted order
	allVars := make([]string, 0, len(cpd.Evidence)+1)
	allVars = append(allVars, cpd.Evidence...)
	allVars = append(allVars, cpd.Variable)
	sort.Strings(allVars)

	// Cardinality
	card := make(map[string]int)
	for k, v := range cpd.EvidenceCard {
		card[k] = v
	}
	card[cpd.Variable] = cpd.VariableCard

	// Calculate size
	size := 1
	for _, v := range allVars {
		size *= card[v]
	}

	values := make([]float64, size)

	// Fill values
	assignment := make(map[string]int)
	cpd.toFactorHelper(0, allVars, assignment, card, values)

	return NewDiscreteFactor(allVars, card, values)
}

func (cpd *TabularCPD) toFactorHelper(depth int, vars []string, assignment map[string]int,
	card map[string]int, result []float64) {
	if depth == len(vars) {
		// Calculate row index from evidence assignment
		rowIdx := 0
		stride := 1
		for i := len(cpd.Evidence) - 1; i >= 0; i-- {
			e := cpd.Evidence[i]
			rowIdx += assignment[e] * stride
			stride *= cpd.EvidenceCard[e]
		}

		// Get column index from variable assignment
		colIdx := assignment[cpd.Variable]

		// Calculate result index
		resIdx := 0
		stride = 1
		for i := len(vars) - 1; i >= 0; i-- {
			v := vars[i]
			resIdx += assignment[v] * stride
			stride *= card[v]
		}

		result[resIdx] = cpd.Values[rowIdx][colIdx]
		return
	}

	v := vars[depth]
	for i := 0; i < card[v]; i++ {
		assignment[v] = i
		cpd.toFactorHelper(depth+1, vars, assignment, card, result)
	}
}

// GetValue returns P(variable=varState | evidence)
func (cpd *TabularCPD) GetValue(varState int, evidenceValues map[string]int) (float64, error) {
	if varState < 0 || varState >= cpd.VariableCard {
		return 0, fmt.Errorf("invalid variable state %d", varState)
	}

	// Calculate row index
	rowIdx := 0
	stride := 1
	for i := len(cpd.Evidence) - 1; i >= 0; i-- {
		e := cpd.Evidence[i]
		val, ok := evidenceValues[e]
		if !ok {
			return 0, fmt.Errorf("missing evidence value for %s", e)
		}
		rowIdx += val * stride
		stride *= cpd.EvidenceCard[e]
	}

	return cpd.Values[rowIdx][varState], nil
}

// String returns a string representation of the CPD
func (cpd *TabularCPD) String() string {
	return fmt.Sprintf("CPD(%s | %v)", cpd.Variable, cpd.Evidence)
}

// Copy creates a deep copy of the CPD
func (cpd *TabularCPD) Copy() *TabularCPD {
	evidenceCopy := make([]string, len(cpd.Evidence))
	copy(evidenceCopy, cpd.Evidence)

	evidenceCardCopy := make(map[string]int)
	for k, v := range cpd.EvidenceCard {
		evidenceCardCopy[k] = v
	}

	valuesCopy := make([][]float64, len(cpd.Values))
	for i, row := range cpd.Values {
		valuesCopy[i] = make([]float64, len(row))
		copy(valuesCopy[i], row)
	}

	return &TabularCPD{
		Variable:     cpd.Variable,
		VariableCard: cpd.VariableCard,
		Evidence:     evidenceCopy,
		EvidenceCard: evidenceCardCopy,
		Values:       valuesCopy,
	}
}
