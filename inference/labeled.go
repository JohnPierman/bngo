package inference

import (
	"fmt"
	"strings"

	"github.com/JohnPierman/bngo/categorical"
	"github.com/JohnPierman/bngo/factors"
)

// LabeledAssignment is one joint assignment of labels together with its
// probability.
type LabeledAssignment struct {
	Labels      map[string]string
	Probability float64
}

// LabeledDistribution is the result of a query stated and answered in the
// vocabulary of the data, rather than in the integer states the factor algebra
// works in.
type LabeledDistribution struct {
	Variables   []string
	Assignments []LabeledAssignment
}

// MostLikely returns the assignment with the highest probability. Ties resolve
// to the first assignment in factor order, so the result is deterministic.
func (d *LabeledDistribution) MostLikely() (LabeledAssignment, error) {
	if len(d.Assignments) == 0 {
		return LabeledAssignment{}, fmt.Errorf("labeled distribution: no assignments")
	}

	best := 0
	for i, assignment := range d.Assignments {
		if assignment.Probability > d.Assignments[best].Probability {
			best = i
		}
	}

	return d.Assignments[best], nil
}

// Probability returns the probability of one assignment, which must fix every
// variable of the distribution.
func (d *LabeledDistribution) Probability(labels map[string]string) (float64, error) {
	if len(labels) != len(d.Variables) {
		return 0, fmt.Errorf("labeled distribution: got %d labels for %d variables %v",
			len(labels), len(d.Variables), d.Variables)
	}

	for _, assignment := range d.Assignments {
		if matchesLabels(assignment.Labels, labels) {
			return assignment.Probability, nil
		}
	}

	return 0, fmt.Errorf("labeled distribution: no assignment matches %v", labels)
}

// matchesLabels reports whether every label in want equals the one in got.
func matchesLabels(got, want map[string]string) bool {
	for variable, label := range want {
		if got[variable] != label {
			return false
		}
	}
	return true
}

// String renders the distribution one assignment per line.
func (d *LabeledDistribution) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "P(%s)\n", strings.Join(d.Variables, ", "))

	for _, assignment := range d.Assignments {
		sb.WriteString("  ")
		for _, variable := range d.Variables {
			fmt.Fprintf(&sb, "%s=%s ", variable, assignment.Labels[variable])
		}
		fmt.Fprintf(&sb, "-> %.4f\n", assignment.Probability)
	}

	return sb.String()
}

// LabelDistribution turns a discrete factor into a labelled distribution using
// the states in the codebook. Every variable of the factor must be declared,
// since a partly labelled result would be more confusing than none.
func LabelDistribution(factor *factors.DiscreteFactor, codebook *categorical.Codebook) (*LabeledDistribution, error) {
	if factor == nil {
		return nil, fmt.Errorf("label distribution: no factor given")
	}
	if codebook == nil {
		return nil, fmt.Errorf("label distribution: no codebook given")
	}

	for _, variable := range factor.Variables {
		if !codebook.Has(variable) {
			return nil, fmt.Errorf("label distribution: no states declared for %s", variable)
		}
	}

	variables := make([]string, len(factor.Variables))
	copy(variables, factor.Variables)

	assignments := make([]LabeledAssignment, 0, len(factor.Values))
	for index, probability := range factor.Values {
		labels, err := labelsForIndex(factor, codebook, index)
		if err != nil {
			return nil, err
		}
		assignments = append(assignments, LabeledAssignment{Labels: labels, Probability: probability})
	}

	return &LabeledDistribution{Variables: variables, Assignments: assignments}, nil
}

// labelsForIndex decodes one position of a factor's value slice back into
// labels. The factor stores values with its last variable varying fastest.
func labelsForIndex(factor *factors.DiscreteFactor, codebook *categorical.Codebook,
	index int) (map[string]string, error) {

	labels := make(map[string]string, len(factor.Variables))
	remainder := index

	for i := len(factor.Variables) - 1; i >= 0; i-- {
		variable := factor.Variables[i]
		cardinality := factor.Cardinality[variable]
		if cardinality <= 0 {
			return nil, fmt.Errorf("label distribution: variable %s has cardinality %d", variable, cardinality)
		}

		label, err := codebook.DecodeValue(variable, remainder%cardinality)
		if err != nil {
			return nil, fmt.Errorf("label distribution: %w", err)
		}
		labels[variable] = label
		remainder /= cardinality
	}

	return labels, nil
}

// QueryLabeled computes P(variables | evidence) with both the query and the
// evidence given as labels, and the result reported as labels.
func (ve *VariableElimination) QueryLabeled(variables []string, evidence map[string]string) (*LabeledDistribution, error) {
	codebook := ve.Model.Codebook()

	encodedEvidence, err := codebook.EncodeRow(evidence)
	if err != nil {
		return nil, fmt.Errorf("query labeled: %w", err)
	}

	factor, err := ve.Query(variables, encodedEvidence)
	if err != nil {
		return nil, err
	}

	return LabelDistribution(factor, codebook)
}

// MAPLabeled returns the most likely label of each query variable given
// labelled evidence.
func (ve *VariableElimination) MAPLabeled(variables []string, evidence map[string]string) (map[string]string, error) {
	codebook := ve.Model.Codebook()

	encodedEvidence, err := codebook.EncodeRow(evidence)
	if err != nil {
		return nil, fmt.Errorf("map labeled: %w", err)
	}

	states, err := ve.MAP(variables, encodedEvidence)
	if err != nil {
		return nil, err
	}

	labels, err := codebook.DecodeRow(states)
	if err != nil {
		return nil, fmt.Errorf("map labeled: %w", err)
	}

	return labels, nil
}
