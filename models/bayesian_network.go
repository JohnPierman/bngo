// Package models provides Bayesian Network model implementations
package models

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/graph"
)

// DiscreteFactor is an alias to avoid import cycles
type DiscreteFactor = factors.DiscreteFactor

// VariableType represents the type of a variable
type VariableType string

const (
	Discrete   VariableType = "discrete"
	Continuous VariableType = "continuous"
)

// BayesianNetwork represents a Bayesian Network with discrete and/or continuous variables
type BayesianNetwork struct {
	DAG          *graph.DAG
	CPDs         map[string]*factors.TabularCPD        // For discrete variables
	GaussianCPDs map[string]*factors.LinearGaussianCPD // For continuous variables
	VariableType map[string]VariableType               // Track variable types
	Cardinality  map[string]int                        // For discrete variables only
}

// NewBayesianNetwork creates a new Bayesian Network
func NewBayesianNetwork(edges [][2]string) (*BayesianNetwork, error) {
	dag, err := graph.NewDAGFromEdges(edges)
	if err != nil {
		return nil, err
	}

	return &BayesianNetwork{
		DAG:          dag,
		CPDs:         make(map[string]*factors.TabularCPD),
		GaussianCPDs: make(map[string]*factors.LinearGaussianCPD),
		VariableType: make(map[string]VariableType),
		Cardinality:  make(map[string]int),
	}, nil
}

// AddCPD adds a discrete CPD to the network
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
	bn.VariableType[cpd.Variable] = Discrete
	bn.Cardinality[cpd.Variable] = cpd.VariableCard
	for k, v := range cpd.EvidenceCard {
		bn.Cardinality[k] = v
		bn.VariableType[k] = Discrete
	}

	return nil
}

// AddGaussianCPD adds a continuous Gaussian CPD to the network
func (bn *BayesianNetwork) AddGaussianCPD(cpd *factors.LinearGaussianCPD) error {
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

	// Check if parents match
	parents := bn.DAG.Parents(cpd.Variable)
	sort.Strings(parents)
	cpdParents := make([]string, len(cpd.Parents))
	copy(cpdParents, cpd.Parents)
	sort.Strings(cpdParents)

	if len(parents) != len(cpdParents) {
		return fmt.Errorf("CPD parents do not match DAG parents for %s", cpd.Variable)
	}
	for i := range parents {
		if parents[i] != cpdParents[i] {
			return fmt.Errorf("CPD parents do not match DAG parents for %s", cpd.Variable)
		}
	}

	bn.GaussianCPDs[cpd.Variable] = cpd
	bn.VariableType[cpd.Variable] = Continuous

	// Set parent types based on CPD
	for parent, ptype := range cpd.ParentTypes {
		if ptype == "discrete" {
			if _, exists := bn.VariableType[parent]; !exists {
				bn.VariableType[parent] = Discrete
			}
			if card, ok := cpd.Cardinality[parent]; ok {
				bn.Cardinality[parent] = card
			}
		} else {
			if _, exists := bn.VariableType[parent]; !exists {
				bn.VariableType[parent] = Continuous
			}
		}
	}

	return nil
}

// GetCPD returns the discrete CPD for a variable
func (bn *BayesianNetwork) GetCPD(variable string) (*factors.TabularCPD, error) {
	cpd, ok := bn.CPDs[variable]
	if !ok {
		return nil, fmt.Errorf("no discrete CPD found for variable %s", variable)
	}
	return cpd, nil
}

// GetGaussianCPD returns the Gaussian CPD for a variable
func (bn *BayesianNetwork) GetGaussianCPD(variable string) (*factors.LinearGaussianCPD, error) {
	cpd, ok := bn.GaussianCPDs[variable]
	if !ok {
		return nil, fmt.Errorf("no Gaussian CPD found for variable %s", variable)
	}
	return cpd, nil
}

// GetCPDs returns all discrete CPDs in the network
func (bn *BayesianNetwork) GetCPDs() []*factors.TabularCPD {
	cpds := make([]*factors.TabularCPD, 0, len(bn.CPDs))
	for _, cpd := range bn.CPDs {
		cpds = append(cpds, cpd)
	}
	return cpds
}

// GetGaussianCPDs returns all Gaussian CPDs in the network
func (bn *BayesianNetwork) GetGaussianCPDs() []*factors.LinearGaussianCPD {
	cpds := make([]*factors.LinearGaussianCPD, 0, len(bn.GaussianCPDs))
	for _, cpd := range bn.GaussianCPDs {
		cpds = append(cpds, cpd)
	}
	return cpds
}

// IsDiscrete returns true if the variable is discrete
func (bn *BayesianNetwork) IsDiscrete(variable string) bool {
	vtype, ok := bn.VariableType[variable]
	return ok && vtype == Discrete
}

// IsContinuous returns true if the variable is continuous
func (bn *BayesianNetwork) IsContinuous(variable string) bool {
	vtype, ok := bn.VariableType[variable]
	return ok && vtype == Continuous
}

// CheckModel validates that the network is properly specified
func (bn *BayesianNetwork) CheckModel() error {
	// Check that all nodes have CPDs
	for _, node := range bn.DAG.Nodes() {
		hasDiscreteCPD := false
		hasGaussianCPD := false

		if _, ok := bn.CPDs[node]; ok {
			hasDiscreteCPD = true
		}
		if _, ok := bn.GaussianCPDs[node]; ok {
			hasGaussianCPD = true
		}

		if !hasDiscreteCPD && !hasGaussianCPD {
			return fmt.Errorf("node %s has no CPD", node)
		}
		if hasDiscreteCPD && hasGaussianCPD {
			return fmt.Errorf("node %s has both discrete and Gaussian CPD", node)
		}
	}

	// Check that all discrete CPDs are consistent with structure
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

	// Check that all Gaussian CPDs are consistent with structure
	for variable, cpd := range bn.GaussianCPDs {
		parents := bn.DAG.Parents(variable)
		sort.Strings(parents)
		cpdParents := make([]string, len(cpd.Parents))
		copy(cpdParents, cpd.Parents)
		sort.Strings(cpdParents)

		if len(parents) != len(cpdParents) {
			return fmt.Errorf("gaussian CPD parent count mismatch for %s", variable)
		}
		for i := range parents {
			if parents[i] != cpdParents[i] {
				return fmt.Errorf("gaussian CPD parent mismatch for %s", variable)
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

// Sample represents a sample from the network with both discrete and continuous values
type Sample struct {
	Discrete   map[string]int
	Continuous map[string]float64
}

// Simulate generates samples from the Bayesian Network (old discrete-only version, deprecated)
func (bn *BayesianNetwork) Simulate(nSamples int, seed int64) ([]map[string]int, error) {
	// Check if all variables are discrete
	for _, node := range bn.DAG.Nodes() {
		if bn.IsContinuous(node) {
			return nil, fmt.Errorf("network contains continuous variables, use SimulateMixed instead")
		}
	}

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

// SimulateMixed generates samples from a Bayesian Network with mixed discrete/continuous variables
func (bn *BayesianNetwork) SimulateMixed(nSamples int, seed int64) ([]Sample, error) {
	if err := bn.CheckModel(); err != nil {
		return nil, err
	}

	r := rand.New(rand.NewSource(seed))

	// Get topological order
	order, err := bn.DAG.TopologicalSort()
	if err != nil {
		return nil, err
	}

	samples := make([]Sample, nSamples)

	for i := 0; i < nSamples; i++ {
		sample := Sample{
			Discrete:   make(map[string]int),
			Continuous: make(map[string]float64),
		}

		for _, node := range order {
			if bn.IsDiscrete(node) {
				// Sample discrete variable
				cpd := bn.CPDs[node]

				// Get parent values
				evidenceValues := make(map[string]int)
				for _, parent := range cpd.Evidence {
					evidenceValues[parent] = sample.Discrete[parent]
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
				sample.Discrete[node] = sampleCategorical(probs, r)
			} else {
				// Sample continuous variable
				cpd := bn.GaussianCPDs[node]

				// Get parent values
				parentValues := make(map[string]interface{})
				for _, parent := range cpd.Parents {
					if bn.IsDiscrete(parent) {
						parentValues[parent] = sample.Discrete[parent]
					} else {
						parentValues[parent] = sample.Continuous[parent]
					}
				}

				// Sample from Gaussian
				val, err := cpd.Sample(parentValues, r)
				if err != nil {
					return nil, fmt.Errorf("failed to sample %s: %v", node, err)
				}
				sample.Continuous[node] = val
			}
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
		DAG:          bn.DAG.Copy(),
		CPDs:         make(map[string]*factors.TabularCPD),
		GaussianCPDs: make(map[string]*factors.LinearGaussianCPD),
		VariableType: make(map[string]VariableType),
		Cardinality:  make(map[string]int),
	}

	for k, v := range bn.CPDs {
		newBN.CPDs[k] = v.Copy()
	}

	for k, v := range bn.GaussianCPDs {
		newBN.GaussianCPDs[k] = v.Copy()
	}

	for k, v := range bn.VariableType {
		newBN.VariableType[k] = v
	}

	for k, v := range bn.Cardinality {
		newBN.Cardinality[k] = v
	}

	return newBN
}

// Fit learns the CPD parameters from data (discrete variables only)
func (bn *BayesianNetwork) Fit(data []map[string]int) error {
	// Check if all variables are discrete
	for _, node := range bn.DAG.Nodes() {
		if bn.IsContinuous(node) {
			return fmt.Errorf("network contains continuous variables, use FitMixed instead")
		}
	}

	// For each node, learn its CPD from data
	for _, node := range bn.Nodes() {
		cpd, err := bn.learnCPD(node, data)
		if err != nil {
			return err
		}
		bn.CPDs[node] = cpd
		bn.VariableType[node] = Discrete
		bn.Cardinality[node] = cpd.VariableCard
		for k, v := range cpd.EvidenceCard {
			bn.Cardinality[k] = v
			bn.VariableType[k] = Discrete
		}
	}

	return nil
}

// FitMixed learns CPD parameters from mixed discrete/continuous data
func (bn *BayesianNetwork) FitMixed(data []Sample) error {
	// Learn each node's CPD
	for _, node := range bn.Nodes() {
		if bn.IsDiscrete(node) || bn.VariableType[node] == "" {
			// Try to determine from data if not specified
			hasIntData := false
			hasFloatData := false

			for _, sample := range data {
				if _, ok := sample.Discrete[node]; ok {
					hasIntData = true
				}
				if _, ok := sample.Continuous[node]; ok {
					hasFloatData = true
				}
			}

			if hasIntData {
				cpd, err := bn.learnDiscreteCPDFromMixed(node, data)
				if err != nil {
					return err
				}
				bn.CPDs[node] = cpd
				bn.VariableType[node] = Discrete
				bn.Cardinality[node] = cpd.VariableCard
				for k, v := range cpd.EvidenceCard {
					bn.Cardinality[k] = v
				}
			} else if hasFloatData {
				cpd, err := bn.learnGaussianCPDFromMixed(node, data)
				if err != nil {
					return err
				}
				bn.GaussianCPDs[node] = cpd
				bn.VariableType[node] = Continuous
			}
		} else if bn.IsContinuous(node) {
			cpd, err := bn.learnGaussianCPDFromMixed(node, data)
			if err != nil {
				return err
			}
			bn.GaussianCPDs[node] = cpd
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

// learnDiscreteCPDFromMixed learns discrete CPD from mixed data
func (bn *BayesianNetwork) learnDiscreteCPDFromMixed(variable string, data []Sample) (*factors.TabularCPD, error) {
	parents := bn.DAG.Parents(variable)
	sort.Strings(parents)

	// Determine cardinality from data
	varCard := 0
	evidenceCard := make(map[string]int)

	for _, sample := range data {
		if val, ok := sample.Discrete[variable]; ok {
			if val+1 > varCard {
				varCard = val + 1
			}
		}
		for _, p := range parents {
			if val, ok := sample.Discrete[p]; ok {
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
			val, ok := sample.Discrete[p]
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

		val, ok := sample.Discrete[variable]
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

// learnGaussianCPDFromMixed learns Gaussian CPD from mixed data using linear regression
func (bn *BayesianNetwork) learnGaussianCPDFromMixed(variable string, data []Sample) (*factors.LinearGaussianCPD, error) {
	parents := bn.DAG.Parents(variable)
	sort.Strings(parents)

	if len(parents) == 0 {
		// No parents: just compute mean and variance
		sum := 0.0
		sumSq := 0.0
		count := 0.0

		for _, sample := range data {
			if val, ok := sample.Continuous[variable]; ok {
				sum += val
				sumSq += val * val
				count++
			}
		}

		if count == 0 {
			return nil, fmt.Errorf("no data for variable %s", variable)
		}

		mean := sum / count
		variance := (sumSq / count) - (mean * mean)
		if variance < 1e-6 {
			variance = 1e-6 // Minimum variance
		}

		return factors.NewLinearGaussianCPD(variable, []string{}, mean, map[string]float64{}, variance)
	}

	// Check if all parents are continuous (linear regression)
	allContinuous := true
	for _, p := range parents {
		if bn.IsDiscrete(p) {
			allContinuous = false
			break
		}
	}

	if allContinuous {
		// Linear regression: X = β₀ + Σᵢ βᵢYᵢ + ε
		// Using simple least squares

		// Collect data points
		var xVals []float64
		var yMatrix [][]float64 // Each row is [1, y1, y2, ..., yn]

		for _, sample := range data {
			xVal, okX := sample.Continuous[variable]
			if !okX {
				continue
			}

			row := []float64{1.0} // Intercept
			valid := true
			for _, p := range parents {
				pVal, ok := sample.Continuous[p]
				if !ok {
					valid = false
					break
				}
				row = append(row, pVal)
			}

			if valid {
				xVals = append(xVals, xVal)
				yMatrix = append(yMatrix, row)
			}
		}

		if len(xVals) < len(parents)+1 {
			return nil, fmt.Errorf("insufficient data for learning Gaussian CPD for %s", variable)
		}

		// Solve using normal equations: β = (Y^T Y)^(-1) Y^T X
		coeffs, err := solveLinearRegression(yMatrix, xVals)
		if err != nil {
			return nil, fmt.Errorf("failed to solve linear regression: %v", err)
		}

		intercept := coeffs[0]
		parentCoeffs := make(map[string]float64)
		for i, p := range parents {
			parentCoeffs[p] = coeffs[i+1]
		}

		// Compute residual variance
		sumSqResid := 0.0
		for i, row := range yMatrix {
			predicted := intercept
			for j, p := range parents {
				predicted += parentCoeffs[p] * row[j+1]
			}
			residual := xVals[i] - predicted
			sumSqResid += residual * residual
		}
		variance := sumSqResid / float64(len(xVals))
		if variance < 1e-6 {
			variance = 1e-6
		}

		return factors.NewLinearGaussianCPD(variable, parents, intercept, parentCoeffs, variance)
	}

	// Mixed discrete/continuous parents or all discrete parents
	// Create separate Gaussian for each discrete parent configuration
	// This is more complex - simplified implementation
	return nil, fmt.Errorf("learning Gaussian CPD with discrete parents not yet fully implemented")
}

// solveLinearRegression solves β = (Y^T Y)^(-1) Y^T X using normal equations
func solveLinearRegression(Y [][]float64, X []float64) ([]float64, error) {
	if len(Y) == 0 || len(Y) != len(X) {
		return nil, fmt.Errorf("invalid input dimensions")
	}

	n := len(Y)
	p := len(Y[0])

	// Compute Y^T Y
	YtY := make([][]float64, p)
	for i := range YtY {
		YtY[i] = make([]float64, p)
		for j := range YtY[i] {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += Y[k][i] * Y[k][j]
			}
			YtY[i][j] = sum
		}
	}

	// Compute Y^T X
	YtX := make([]float64, p)
	for i := 0; i < p; i++ {
		sum := 0.0
		for k := 0; k < n; k++ {
			sum += Y[k][i] * X[k]
		}
		YtX[i] = sum
	}

	// Solve YtY * β = YtX using Gaussian elimination
	// Create augmented matrix
	augmented := make([][]float64, p)
	for i := range augmented {
		augmented[i] = make([]float64, p+1)
		copy(augmented[i][:p], YtY[i])
		augmented[i][p] = YtX[i]
	}

	// Forward elimination
	for i := 0; i < p; i++ {
		// Find pivot
		maxRow := i
		for k := i + 1; k < p; k++ {
			if math.Abs(augmented[k][i]) > math.Abs(augmented[maxRow][i]) {
				maxRow = k
			}
		}
		augmented[i], augmented[maxRow] = augmented[maxRow], augmented[i]

		if math.Abs(augmented[i][i]) < 1e-10 {
			return nil, fmt.Errorf("singular matrix in linear regression")
		}

		// Eliminate
		for k := i + 1; k < p; k++ {
			factor := augmented[k][i] / augmented[i][i]
			for j := i; j <= p; j++ {
				augmented[k][j] -= factor * augmented[i][j]
			}
		}
	}

	// Back substitution
	beta := make([]float64, p)
	for i := p - 1; i >= 0; i-- {
		beta[i] = augmented[i][p]
		for j := i + 1; j < p; j++ {
			beta[i] -= augmented[i][j] * beta[j]
		}
		beta[i] /= augmented[i][i]
	}

	return beta, nil
}
