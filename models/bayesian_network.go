// Package models provides Bayesian Network model implementations
package models

import (
	"fmt"
	"math/rand"
	"sort"
	
	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/graph"
)

// DiscreteFactor is an alias to avoid import cycles
type DiscreteFactor = factors.DiscreteFactor

// BayesianNetwork represents a discrete Bayesian Network
type BayesianNetwork struct {
	DAG         *graph.DAG
	CPDs        map[string]*factors.TabularCPD
	Cardinality map[string]int
}

// NewBayesianNetwork creates a new Bayesian Network
func NewBayesianNetwork(edges [][2]string) (*BayesianNetwork, error) {
	dag, err := graph.NewDAGFromEdges(edges)
	if err != nil {
		return nil, err
	}

	return &BayesianNetwork{
		DAG:         dag,
		CPDs:        make(map[string]*factors.TabularCPD),
		Cardinality: make(map[string]int),
	}, nil
}

// AddCPD adds a CPD to the network
func (bn *BayesianNetwork) AddCPD(cpd *factors.TabularCPD) error {
	// Check if variable exists in DAG
	found := false
	for _, node := range bn.DAG.Nodes() {
		if node == cpd.Variable {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("variable %s not in network", cpd.Variable)
	}

	// Check if evidence matches parents
	parents := bn.DAG.Parents(cpd.Variable)
	sort.Strings(parents)
	evidenceSorted := make([]string, len(cpd.Evidence))
	copy(evidenceSorted, cpd.Evidence)
	sort.Strings(evidenceSorted)

	if len(parents) != len(evidenceSorted) {
		return fmt.Errorf("CPD evidence does not match parents for %s", cpd.Variable)
	}
	for i := range parents {
		if parents[i] != evidenceSorted[i] {
			return fmt.Errorf("CPD evidence does not match parents for %s", cpd.Variable)
		}
	}

	bn.CPDs[cpd.Variable] = cpd
	bn.Cardinality[cpd.Variable] = cpd.VariableCard
	for k, v := range cpd.EvidenceCard {
		bn.Cardinality[k] = v
	}

	return nil
}

// GetCPD returns the CPD for a variable
func (bn *BayesianNetwork) GetCPD(variable string) (*factors.TabularCPD, error) {
	cpd, ok := bn.CPDs[variable]
	if !ok {
		return nil, fmt.Errorf("no CPD found for variable %s", variable)
	}
	return cpd, nil
}

// GetCPDs returns all CPDs in the network
func (bn *BayesianNetwork) GetCPDs() []*factors.TabularCPD {
	cpds := make([]*factors.TabularCPD, 0, len(bn.CPDs))
	for _, cpd := range bn.CPDs {
		cpds = append(cpds, cpd)
	}
	return cpds
}

// CheckModel validates that the network is properly specified
func (bn *BayesianNetwork) CheckModel() error {
	// Check that all nodes have CPDs
	for _, node := range bn.DAG.Nodes() {
		if _, ok := bn.CPDs[node]; !ok {
			return fmt.Errorf("node %s has no CPD", node)
		}
	}

	// Check that all CPDs are consistent with structure
	for variable, cpd := range bn.CPDs {
		parents := bn.DAG.Parents(variable)
		sort.Strings(parents)
		evidenceSorted := make([]string, len(cpd.Evidence))
		copy(evidenceSorted, cpd.Evidence)
		sort.Strings(evidenceSorted)

		if len(parents) != len(evidenceSorted) {
			return fmt.Errorf("CPD evidence count mismatch for %s", variable)
		}
		for i := range parents {
			if parents[i] != evidenceSorted[i] {
				return fmt.Errorf("CPD evidence mismatch for %s", variable)
			}
		}
	}

	return nil
}

// Nodes returns all nodes in the network
func (bn *BayesianNetwork) Nodes() []string {
	return bn.DAG.Nodes()
}

// Edges returns all edges in the network
func (bn *BayesianNetwork) Edges() [][2]string {
	return bn.DAG.Edges()
}

// Simulate generates samples from the Bayesian Network
func (bn *BayesianNetwork) Simulate(nSamples int, seed int64) ([]map[string]int, error) {
	if err := bn.CheckModel(); err != nil {
		return nil, err
	}

	r := rand.New(rand.NewSource(seed))

	// Get topological order
	order, err := bn.DAG.TopologicalSort()
	if err != nil {
		return nil, err
	}

	samples := make([]map[string]int, nSamples)

	for i := 0; i < nSamples; i++ {
		sample := make(map[string]int)

		for _, node := range order {
			cpd := bn.CPDs[node]

			// Get parent values
			evidenceValues := make(map[string]int)
			for _, parent := range cpd.Evidence {
				evidenceValues[parent] = sample[parent]
			}

			// Calculate row index
			rowIdx := 0
			stride := 1
			for j := len(cpd.Evidence) - 1; j >= 0; j-- {
				e := cpd.Evidence[j]
				rowIdx += evidenceValues[e] * stride
				stride *= cpd.EvidenceCard[e]
			}

			// Sample from the distribution
			probs := cpd.Values[rowIdx]
			sample[node] = sampleCategorical(probs, r)
		}

		samples[i] = sample
	}

	return samples, nil
}

func sampleCategorical(probs []float64, r *rand.Rand) int {
	u := r.Float64()
	cumSum := 0.0
	for i, p := range probs {
		cumSum += p
		if u <= cumSum {
			return i
		}
	}
	return len(probs) - 1
}

// Predict predicts missing values in partial observations
func (bn *BayesianNetwork) Predict(observations []map[string]int) (map[string][]int, error) {
	if err := bn.CheckModel(); err != nil {
		return nil, err
	}

	allVars := bn.Nodes()

	// Find which variables to predict
	toPredictMap := make(map[string]bool)
	for _, obs := range observations {
		for _, v := range allVars {
			if _, ok := obs[v]; !ok {
				toPredictMap[v] = true
			}
		}
	}

	toPredict := make([]string, 0, len(toPredictMap))
	for v := range toPredictMap {
		toPredict = append(toPredict, v)
	}
	sort.Strings(toPredict)

	predictions := make(map[string][]int)
	for _, v := range toPredict {
		predictions[v] = make([]int, len(observations))
	}

	// For each observation
	for i, obs := range observations {
		for _, v := range toPredict {
			if _, ok := obs[v]; ok {
				predictions[v][i] = obs[v]
				continue
			}

			// Predict using MAP (maximum a posteriori)
			pred, err := bn.predictSingle(v, obs)
			if err != nil {
				return nil, err
			}
			predictions[v][i] = pred
		}
	}

	return predictions, nil
}

func (bn *BayesianNetwork) predictSingle(variable string, evidence map[string]int) (int, error) {
	// Check if all parents are observed - fast path
	cpd := bn.CPDs[variable]
	
	// Check if all parent variables are in evidence
	allPresent := true
	evidenceValues := make(map[string]int)
	for _, e := range cpd.Evidence {
		if val, ok := evidence[e]; ok {
			evidenceValues[e] = val
		} else {
			allPresent = false
			break
		}
	}
	
	if allPresent && len(cpd.Evidence) > 0 {
		// Fast path: Calculate directly from CPD
		rowIdx := 0
		stride := 1
		for j := len(cpd.Evidence) - 1; j >= 0; j-- {
			e := cpd.Evidence[j]
			rowIdx += evidenceValues[e] * stride
			stride *= cpd.EvidenceCard[e]
		}
		
		// Find argmax
		probs := cpd.Values[rowIdx]
		maxIdx := 0
		maxProb := probs[0]
		for i := 1; i < len(probs); i++ {
			if probs[i] > maxProb {
				maxProb = probs[i]
				maxIdx = i
			}
		}
		return maxIdx, nil
	}
	
	// Need full inference using Variable Elimination
	return bn.predictUsingInference(variable, evidence)
}

func (bn *BayesianNetwork) predictUsingInference(variable string, evidence map[string]int) (int, error) {
	// Convert all CPDs to factors
	factorList := make([]*DiscreteFactor, 0)
	for _, cpd := range bn.CPDs {
		factor, err := cpd.ToFactor()
		if err != nil {
			return 0, err
		}
		factorList = append(factorList, factor)
	}
	
	// Reduce factors by evidence
	reducedFactors := make([]*DiscreteFactor, 0)
	for _, factor := range factorList {
		reduced, err := factor.Reduce(evidence)
		if err != nil {
			return 0, err
		}
		reducedFactors = append(reducedFactors, reduced)
	}
	
	// Find variables to eliminate (all except variable and evidence)
	toEliminate := make([]string, 0)
	for _, node := range bn.Nodes() {
		if node != variable {
			if _, ok := evidence[node]; !ok {
				toEliminate = append(toEliminate, node)
			}
		}
	}
	
	// Eliminate variables one by one
	currentFactors := reducedFactors
	for _, v := range toEliminate {
		currentFactors = bn.eliminateVariable(v, currentFactors)
	}
	
	// Multiply remaining factors
	if len(currentFactors) == 0 {
		// Default to state 0
		return 0, nil
	}
	
	result := currentFactors[0]
	for i := 1; i < len(currentFactors); i++ {
		newResult, err := result.Multiply(currentFactors[i])
		if err != nil {
			return 0, err
		}
		result = newResult
	}
	
	// Find MAP assignment (argmax)
	maxIdx := 0
	maxVal := result.Values[0]
	for i := 1; i < len(result.Values); i++ {
		if result.Values[i] > maxVal {
			maxVal = result.Values[i]
			maxIdx = i
		}
	}
	
	// Convert index to variable state
	// For single variable, index is just the state
	if len(result.Variables) == 1 {
		return maxIdx, nil
	}
	
	// Multiple variables - extract the one we want
	varPos := -1
	for i, v := range result.Variables {
		if v == variable {
			varPos = i
			break
		}
	}
	
	if varPos == -1 {
		return 0, fmt.Errorf("variable %s not found in result", variable)
	}
	
	// Calculate stride for this variable
	stride := 1
	for i := len(result.Variables) - 1; i > varPos; i-- {
		stride *= bn.Cardinality[result.Variables[i]]
	}
	
	// Extract state for this variable
	varState := (maxIdx / stride) % bn.Cardinality[variable]
	return varState, nil
}

func (bn *BayesianNetwork) eliminateVariable(variable string, factorList []*DiscreteFactor) []*DiscreteFactor {
	// Find factors containing the variable
	relevant := make([]*DiscreteFactor, 0)
	irrelevant := make([]*DiscreteFactor, 0)
	
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
	
	// Marginalize out the variable
	marginalized, err := product.Marginalize([]string{variable})
	if err != nil {
		return irrelevant
	}
	
	// Return irrelevant + marginalized
	result := append(irrelevant, marginalized)
	return result
}

// Copy creates a deep copy of the Bayesian Network
func (bn *BayesianNetwork) Copy() *BayesianNetwork {
	newBN := &BayesianNetwork{
		DAG:         bn.DAG.Copy(),
		CPDs:        make(map[string]*factors.TabularCPD),
		Cardinality: make(map[string]int),
	}

	for k, v := range bn.CPDs {
		newBN.CPDs[k] = v.Copy()
	}

	for k, v := range bn.Cardinality {
		newBN.Cardinality[k] = v
	}

	return newBN
}

// Fit learns the CPD parameters from data
func (bn *BayesianNetwork) Fit(data []map[string]int) error {
	// For each node, learn its CPD from data
	for _, node := range bn.Nodes() {
		cpd, err := bn.learnCPD(node, data)
		if err != nil {
			return err
		}
		bn.CPDs[node] = cpd
		bn.Cardinality[node] = cpd.VariableCard
		for k, v := range cpd.EvidenceCard {
			bn.Cardinality[k] = v
		}
	}

	return nil
}

func (bn *BayesianNetwork) learnCPD(variable string, data []map[string]int) (*factors.TabularCPD, error) {
	parents := bn.DAG.Parents(variable)
	sort.Strings(parents)

	// Determine cardinality from data
	varCard := 0
	evidenceCard := make(map[string]int)

	for _, sample := range data {
		if val, ok := sample[variable]; ok {
			if val+1 > varCard {
				varCard = val + 1
			}
		}
		for _, p := range parents {
			if val, ok := sample[p]; ok {
				if val+1 > evidenceCard[p] {
					evidenceCard[p] = val + 1
				}
			}
		}
	}

	// Count occurrences
	numRows := 1
	for _, p := range parents {
		numRows *= evidenceCard[p]
	}

	counts := make([][]float64, numRows)
	for i := range counts {
		counts[i] = make([]float64, varCard)
	}

	// Count from data
	for _, sample := range data {
		// Calculate row index
		rowIdx := 0
		stride := 1
		valid := true
		for j := len(parents) - 1; j >= 0; j-- {
			p := parents[j]
			val, ok := sample[p]
			if !ok {
				valid = false
				break
			}
			rowIdx += val * stride
			stride *= evidenceCard[p]
		}

		if !valid {
			continue
		}

		val, ok := sample[variable]
		if !ok {
			continue
		}

		counts[rowIdx][val]++
	}

	// Normalize to get probabilities (with Laplace smoothing)
	values := make([][]float64, numRows)
	for i := range values {
		values[i] = make([]float64, varCard)
		sum := 0.0
		for j := range values[i] {
			counts[i][j] += 1.0 // Laplace smoothing
			sum += counts[i][j]
		}
		for j := range values[i] {
			values[i][j] = counts[i][j] / sum
		}
	}

	return factors.NewTabularCPD(variable, varCard, values, parents, evidenceCard)
}
