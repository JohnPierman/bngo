package models

import (
	"fmt"

	"github.com/JohnPierman/bngo/categorical"
	"github.com/JohnPierman/bngo/factors"
)

// Codebook returns the network's codebook, creating an empty one on first use.
// The returned codebook is the live registry, so declaring states on it declares
// them on the network.
func (bn *BayesianNetwork) Codebook() *categorical.Codebook {
	if bn.states == nil {
		bn.states = categorical.NewCodebook()
	}
	return bn.states
}

// SetCodebook replaces the network's codebook with a copy of the one given, so
// later changes to the caller's codebook do not reach into the network.
func (bn *BayesianNetwork) SetCodebook(codebook *categorical.Codebook) error {
	if codebook == nil {
		return fmt.Errorf("set codebook: no codebook given")
	}
	bn.states = codebook.Copy()
	return nil
}

// DeclareStates records the labels of a categorical variable in the order the
// caller wants them numbered. Declaring states up front is what keeps a rare
// label out of the sample from being dropped when the CPDs are fitted.
func (bn *BayesianNetwork) DeclareStates(variable string, labels []string) error {
	return bn.Codebook().Declare(variable, labels)
}

// StateNames returns the declared states of a variable.
func (bn *BayesianNetwork) StateNames(variable string) (*categorical.StateNames, bool) {
	return bn.Codebook().Get(variable)
}

// copyStates returns an independent copy of the codebook, tolerating a network
// that never had one.
func (bn *BayesianNetwork) copyStates() *categorical.Codebook {
	if bn.states == nil {
		return categorical.NewCodebook()
	}
	return bn.states.Copy()
}

// ValidateCodebook checks that the declared states agree with the CPDs already
// in the network. It reports the first disagreement it finds, and reports nothing
// for variables whose states were never declared.
func (bn *BayesianNetwork) ValidateCodebook() error {
	codebook := bn.Codebook()

	for _, cpd := range bn.GetCPDs() {
		if err := validateCardinality(codebook, cpd.Variable, cpd.VariableCard); err != nil {
			return err
		}
		for evidence, card := range cpd.EvidenceCard {
			if err := validateCardinality(codebook, evidence, card); err != nil {
				return err
			}
		}
	}

	return nil
}

// validateCardinality compares a declared number of states against the number a
// CPD assumes.
func validateCardinality(codebook *categorical.Codebook, variable string, card int) error {
	declared, ok := codebook.Cardinality(variable)
	if !ok || declared == card {
		return nil
	}
	return fmt.Errorf("variable %s has %d declared states but its CPD assumes %d", variable, declared, card)
}

// AddCategoricalCPD adds a discrete CPD whose cardinalities come from the
// declared states, so the caller supplies probabilities and nothing else. Rows
// of values run over the evidence combinations in the same order as
// NewTabularCPD expects, with the last evidence variable varying fastest.
func (bn *BayesianNetwork) AddCategoricalCPD(variable string, evidence []string, values [][]float64) error {
	codebook := bn.Codebook()

	variableCard, ok := codebook.Cardinality(variable)
	if !ok {
		return fmt.Errorf("add categorical CPD: no states declared for %s", variable)
	}

	evidenceCard := make(map[string]int, len(evidence))
	for _, parent := range evidence {
		card, ok := codebook.Cardinality(parent)
		if !ok {
			return fmt.Errorf("add categorical CPD for %s: no states declared for evidence %s", variable, parent)
		}
		evidenceCard[parent] = card
	}

	cpd, err := factors.NewTabularCPD(variable, variableCard, values, evidence, evidenceCard)
	if err != nil {
		return fmt.Errorf("add categorical CPD for %s: %w", variable, err)
	}

	return bn.AddCPD(cpd)
}

// FitCategorical learns every CPD from label valued rows. States already
// declared keep their order and their full set of labels; states of any other
// variable present in the data are inferred from it. A row that leaves a
// variable out, or gives it the empty label, counts as an unobserved value for
// that variable rather than as a state.
func (bn *BayesianNetwork) FitCategorical(rows []map[string]string) error {
	if err := bn.completeCodebookFrom(rows); err != nil {
		return err
	}

	codebook := bn.Codebook()
	for _, node := range bn.Nodes() {
		if !codebook.Has(node) {
			return fmt.Errorf("fit categorical: no observations and no declared states for %s", node)
		}
	}

	encoded, err := codebook.EncodeRows(rows)
	if err != nil {
		return fmt.Errorf("fit categorical: %w", err)
	}

	return bn.fitEncoded(encoded, codebook.Cardinalities())
}

// completeCodebookFrom infers states for the variables in the data that have not
// been declared, leaving declared variables untouched.
func (bn *BayesianNetwork) completeCodebookFrom(rows []map[string]string) error {
	inferred, err := categorical.InferCodebook(rows)
	if err != nil {
		return fmt.Errorf("fit categorical: %w", err)
	}

	codebook := bn.Codebook()
	for _, variable := range inferred.Variables() {
		if codebook.Has(variable) {
			continue
		}
		states, _ := inferred.Get(variable)
		if err := codebook.Set(variable, states); err != nil {
			return fmt.Errorf("fit categorical: %w", err)
		}
	}

	return nil
}

// SimulateCategorical generates samples and returns them as labels.
func (bn *BayesianNetwork) SimulateCategorical(nSamples int, seed int64) ([]map[string]string, error) {
	if err := bn.requireDeclaredStates("simulate categorical"); err != nil {
		return nil, err
	}

	samples, err := bn.Simulate(nSamples, seed)
	if err != nil {
		return nil, err
	}

	decoded, err := bn.Codebook().DecodeRows(samples)
	if err != nil {
		return nil, fmt.Errorf("simulate categorical: %w", err)
	}

	return decoded, nil
}

// PredictCategorical fills in the variables missing from label valued
// observations, returning the most likely label of each. The result holds one
// slice per predicted variable, aligned with the observations given.
func (bn *BayesianNetwork) PredictCategorical(observations []map[string]string) (map[string][]string, error) {
	if err := bn.requireDeclaredStates("predict categorical"); err != nil {
		return nil, err
	}

	codebook := bn.Codebook()

	encoded, err := codebook.EncodeRows(observations)
	if err != nil {
		return nil, fmt.Errorf("predict categorical: %w", err)
	}

	predictions, err := bn.Predict(encoded)
	if err != nil {
		return nil, err
	}

	labelled := make(map[string][]string, len(predictions))
	for variable, states := range predictions {
		labels := make([]string, len(states))
		for i, state := range states {
			label, err := codebook.DecodeValue(variable, state)
			if err != nil {
				return nil, fmt.Errorf("predict categorical: %w", err)
			}
			labels[i] = label
		}
		labelled[variable] = labels
	}

	return labelled, nil
}

// requireDeclaredStates checks that every discrete node has states, which is
// what any translation between labels and integer states needs.
func (bn *BayesianNetwork) requireDeclaredStates(operation string) error {
	codebook := bn.Codebook()

	for _, node := range bn.Nodes() {
		if bn.IsContinuous(node) {
			continue
		}
		if !codebook.Has(node) {
			return fmt.Errorf("%s: no states declared for %s", operation, node)
		}
	}

	return bn.ValidateCodebook()
}
