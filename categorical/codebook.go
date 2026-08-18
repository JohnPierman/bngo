package categorical

import (
	"fmt"
	"sort"
)

// MissingLabel is the label that means "this field was not observed". Encoding
// leaves the variable out of the row instead of inventing a state for it, which
// is what the estimators and parameter learners already expect from an
// incomplete row.
const MissingLabel = ""

// Codebook holds the StateNames of every categorical variable in a data set and
// translates whole rows between labels and state indices.
type Codebook struct {
	states map[string]*StateNames
}

// NewCodebook creates an empty codebook.
func NewCodebook() *Codebook {
	return &Codebook{states: make(map[string]*StateNames)}
}

// InferCodebook builds a codebook from labelled rows, inferring each variable's
// states from the labels observed for it. Missing values are ignored rather than
// treated as a state of their own.
func InferCodebook(rows []map[string]string) (*Codebook, error) {
	observed := make(map[string][]string)
	seen := make(map[string]map[string]bool)

	for _, row := range rows {
		for variable, label := range row {
			if label == MissingLabel {
				continue
			}
			if seen[variable] == nil {
				seen[variable] = make(map[string]bool)
			}
			if seen[variable][label] {
				continue
			}
			seen[variable][label] = true
			observed[variable] = append(observed[variable], label)
		}
	}

	codebook := NewCodebook()
	for variable, labels := range observed {
		states, err := InferStateNames(labels)
		if err != nil {
			return nil, fmt.Errorf("inferring states for %q: %w", variable, err)
		}
		codebook.states[variable] = states
	}

	return codebook, nil
}

// Set registers the states of a variable, replacing any previous entry.
func (c *Codebook) Set(variable string, states *StateNames) error {
	if variable == "" {
		return fmt.Errorf("codebook: variable name is empty")
	}
	if states == nil {
		return fmt.Errorf("codebook: no states given for %q", variable)
	}
	c.states[variable] = states
	return nil
}

// Declare registers a variable whose labels are given in the order the caller
// wants them numbered.
func (c *Codebook) Declare(variable string, labels []string) error {
	states, err := NewStateNames(labels)
	if err != nil {
		return fmt.Errorf("declaring %q: %w", variable, err)
	}
	return c.Set(variable, states)
}

// Get returns the states of a variable.
func (c *Codebook) Get(variable string) (*StateNames, bool) {
	states, ok := c.states[variable]
	return states, ok
}

// Has reports whether the variable is known to the codebook.
func (c *Codebook) Has(variable string) bool {
	_, ok := c.states[variable]
	return ok
}

// Variables returns the known variables in sorted order.
func (c *Codebook) Variables() []string {
	variables := make([]string, 0, len(c.states))
	for variable := range c.states {
		variables = append(variables, variable)
	}
	sort.Strings(variables)
	return variables
}

// Cardinality returns the number of states of a variable.
func (c *Codebook) Cardinality(variable string) (int, bool) {
	states, ok := c.states[variable]
	if !ok {
		return 0, false
	}
	return states.Cardinality(), true
}

// Cardinalities returns the cardinality of every known variable. It is the
// bridge to the structure learners and to factor construction, which work in
// integer states and only need to know how many there are.
func (c *Codebook) Cardinalities() map[string]int {
	cardinalities := make(map[string]int, len(c.states))
	for variable, states := range c.states {
		cardinalities[variable] = states.Cardinality()
	}
	return cardinalities
}

// EncodeValue turns one label into its state index.
func (c *Codebook) EncodeValue(variable, label string) (int, error) {
	states, ok := c.states[variable]
	if !ok {
		return 0, fmt.Errorf("codebook: unknown variable %q", variable)
	}
	index, err := states.Index(label)
	if err != nil {
		return 0, fmt.Errorf("encoding %q: %w", variable, err)
	}
	return index, nil
}

// DecodeValue turns one state index back into its label.
func (c *Codebook) DecodeValue(variable string, index int) (string, error) {
	states, ok := c.states[variable]
	if !ok {
		return "", fmt.Errorf("codebook: unknown variable %q", variable)
	}
	label, err := states.Label(index)
	if err != nil {
		return "", fmt.Errorf("decoding %q: %w", variable, err)
	}
	return label, nil
}

// EncodeRow turns a labelled row into the integer row the model works with.
// Missing values are left out of the result, so a partially observed row stays
// partially observed. The argument is not modified.
func (c *Codebook) EncodeRow(row map[string]string) (map[string]int, error) {
	encoded := make(map[string]int, len(row))

	for variable, label := range row {
		if label == MissingLabel {
			continue
		}
		index, err := c.EncodeValue(variable, label)
		if err != nil {
			return nil, err
		}
		encoded[variable] = index
	}

	return encoded, nil
}

// DecodeRow turns an integer row back into labels. The argument is not modified.
func (c *Codebook) DecodeRow(row map[string]int) (map[string]string, error) {
	decoded := make(map[string]string, len(row))

	for variable, index := range row {
		label, err := c.DecodeValue(variable, index)
		if err != nil {
			return nil, err
		}
		decoded[variable] = label
	}

	return decoded, nil
}

// EncodeRows encodes every row, reporting which row failed if one does.
func (c *Codebook) EncodeRows(rows []map[string]string) ([]map[string]int, error) {
	encoded := make([]map[string]int, len(rows))

	for i, row := range rows {
		row, err := c.EncodeRow(row)
		if err != nil {
			return nil, fmt.Errorf("row %d: %w", i, err)
		}
		encoded[i] = row
	}

	return encoded, nil
}

// DecodeRows decodes every row, reporting which row failed if one does.
func (c *Codebook) DecodeRows(rows []map[string]int) ([]map[string]string, error) {
	decoded := make([]map[string]string, len(rows))

	for i, row := range rows {
		row, err := c.DecodeRow(row)
		if err != nil {
			return nil, fmt.Errorf("row %d: %w", i, err)
		}
		decoded[i] = row
	}

	return decoded, nil
}

// Copy returns an independent codebook. StateNames are immutable, so they are
// shared rather than duplicated.
func (c *Codebook) Copy() *Codebook {
	copied := &Codebook{states: make(map[string]*StateNames, len(c.states))}
	for variable, states := range c.states {
		copied.states[variable] = states
	}
	return copied
}
