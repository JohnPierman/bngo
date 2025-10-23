// Package inference provides probabilistic inference algorithms
package inference

import (
	"fmt"
	"sort"

	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/models"
)

// VariableElimination performs exact inference using variable elimination
type VariableElimination struct {
	Model *models.BayesianNetwork
}

// NewVariableElimination creates a new variable elimination inference engine
func NewVariableElimination(model *models.BayesianNetwork) (*VariableElimination, error) {
	if err := model.CheckModel(); err != nil {
		return nil, err
	}
	return &VariableElimination{Model: model}, nil
}

// Query computes P(variables | evidence)
func (ve *VariableElimination) Query(variables []string, evidence map[string]int) (*factors.DiscreteFactor, error) {
	// Convert all CPDs to factors
	factorList := make([]*factors.DiscreteFactor, 0)
	for _, cpd := range ve.Model.GetCPDs() {
		factor, err := cpd.ToFactor()
		if err != nil {
			return nil, err
		}
		factorList = append(factorList, factor)
	}

	// Reduce factors by evidence
	reducedFactors := make([]*factors.DiscreteFactor, 0)
	for _, factor := range factorList {
		reduced, err := factor.Reduce(evidence)
		if err != nil {
			return nil, err
		}
		reducedFactors = append(reducedFactors, reduced)
	}

	// Find variables to eliminate
	allVars := make(map[string]bool)
	for _, node := range ve.Model.Nodes() {
		allVars[node] = true
	}

	queryVars := make(map[string]bool)
	for _, v := range variables {
		queryVars[v] = true
		delete(allVars, v)
	}

	for v := range evidence {
		delete(allVars, v)
	}

	toEliminate := make([]string, 0, len(allVars))
	for v := range allVars {
		toEliminate = append(toEliminate, v)
	}
	sort.Strings(toEliminate)

	// Eliminate variables one by one
	currentFactors := reducedFactors
	for _, v := range toEliminate {
		currentFactors = ve.eliminateVariable(v, currentFactors)
	}

	// Multiply remaining factors
	if len(currentFactors) == 0 {
		return nil, fmt.Errorf("no factors remaining after elimination")
	}

	result := currentFactors[0]
	for i := 1; i < len(currentFactors); i++ {
		newResult, err := result.Multiply(currentFactors[i])
		if err != nil {
			return nil, err
		}
		result = newResult
	}

	// Normalize
	if err := result.Normalize(); err != nil {
		return nil, err
	}

	return result, nil
}

func (ve *VariableElimination) eliminateVariable(variable string, factorList []*factors.DiscreteFactor) []*factors.DiscreteFactor {
	// Find factors containing the variable
	relevant := make([]*factors.DiscreteFactor, 0)
	irrelevant := make([]*factors.DiscreteFactor, 0)

	for _, factor := range factorList {
		contains := false
		for _, v := range factor.Variables {
			if v == variable {
				contains = true
				break
			}
		}
		if contains {
			relevant = append(relevant, factor)
		} else {
			irrelevant = append(irrelevant, factor)
		}
	}

	if len(relevant) == 0 {
		return factorList
	}

	// Multiply relevant factors
	product := relevant[0]
	for i := 1; i < len(relevant); i++ {
		newProduct, err := product.Multiply(relevant[i])
		if err != nil {
			// Skip on error
			continue
		}
		product = newProduct
	}

	// Marginalize out the variable
	marginalized, err := product.Marginalize([]string{variable})
	if err != nil {
		// If marginalization fails, return irrelevant factors
		return irrelevant
	}

	// Return irrelevant + marginalized
	result := append(irrelevant, marginalized)
	return result
}

// MAP computes the maximum a posteriori assignment
func (ve *VariableElimination) MAP(variables []string, evidence map[string]int) (map[string]int, error) {
	// Convert all CPDs to factors
	factorList := make([]*factors.DiscreteFactor, 0)
	for _, cpd := range ve.Model.GetCPDs() {
		factor, err := cpd.ToFactor()
		if err != nil {
			return nil, err
		}
		factorList = append(factorList, factor)
	}

	// Reduce factors by evidence
	reducedFactors := make([]*factors.DiscreteFactor, 0)
	for _, factor := range factorList {
		reduced, err := factor.Reduce(evidence)
		if err != nil {
			return nil, err
		}
		reducedFactors = append(reducedFactors, reduced)
	}

	// Find variables to eliminate (not in query or evidence)
	allVars := make(map[string]bool)
	for _, node := range ve.Model.Nodes() {
		allVars[node] = true
	}

	queryVars := make(map[string]bool)
	for _, v := range variables {
		queryVars[v] = true
		delete(allVars, v)
	}

	for v := range evidence {
		delete(allVars, v)
	}

	toEliminate := make([]string, 0, len(allVars))
	for v := range allVars {
		toEliminate = append(toEliminate, v)
	}
	sort.Strings(toEliminate)

	// Eliminate variables using max-marginalization
	currentFactors := reducedFactors
	for _, v := range toEliminate {
		currentFactors = ve.maxEliminateVariable(v, currentFactors)
	}

	// Multiply remaining factors
	if len(currentFactors) == 0 {
		return nil, fmt.Errorf("no factors remaining after elimination")
	}

	result := currentFactors[0]
	for i := 1; i < len(currentFactors); i++ {
		newResult, err := result.Multiply(currentFactors[i])
		if err != nil {
			return nil, err
		}
		result = newResult
	}

	// Find maximum assignment
	maxIdx := 0
	maxVal := result.Values[0]
	for i := 1; i < len(result.Values); i++ {
		if result.Values[i] > maxVal {
			maxVal = result.Values[i]
			maxIdx = i
		}
	}

	// Convert index to assignment
	assignment := make(map[string]int)
	idx := maxIdx
	for i := len(result.Variables) - 1; i >= 0; i-- {
		v := result.Variables[i]
		card := ve.Model.Cardinality[v]
		assignment[v] = idx % card
		idx /= card
	}

	return assignment, nil
}

func (ve *VariableElimination) maxEliminateVariable(variable string, factorList []*factors.DiscreteFactor) []*factors.DiscreteFactor {
	// Find factors containing the variable
	relevant := make([]*factors.DiscreteFactor, 0)
	irrelevant := make([]*factors.DiscreteFactor, 0)

	for _, factor := range factorList {
		contains := false
		for _, v := range factor.Variables {
			if v == variable {
				contains = true
				break
			}
		}
		if contains {
			relevant = append(relevant, factor)
		} else {
			irrelevant = append(irrelevant, factor)
		}
	}

	if len(relevant) == 0 {
		return factorList
	}

	// Multiply relevant factors
	product := relevant[0]
	for i := 1; i < len(relevant); i++ {
		newProduct, err := product.Multiply(relevant[i])
		if err != nil {
			continue
		}
		product = newProduct
	}

	// Max-marginalize out the variable
	marginalized, err := product.MaxMarginalize([]string{variable})
	if err != nil {
		return irrelevant
	}

	// Return irrelevant + marginalized
	result := append(irrelevant, marginalized)
	return result
}
