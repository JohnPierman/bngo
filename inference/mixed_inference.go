package inference

import (
	"fmt"
	"sort"

	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/models"
)

// MixedEvidence stores evidence that can be both discrete and continuous
type MixedEvidence struct {
	Discrete   map[string]int
	Continuous map[string]float64
}

// MixedQueryResult stores the result of a mixed query
type MixedQueryResult struct {
	DiscreteVars   []string
	ContinuousVars []string
	Cardinality    map[string]int
	Values         []float64
	Mean           map[string]float64
	Covariance     map[string]map[string]float64
}

// MixedVariableElimination performs exact inference on mixed models
type MixedVariableElimination struct {
	Model *models.BayesianNetwork
}

// NewMixedVariableElimination creates a new mixed inference engine
func NewMixedVariableElimination(model *models.BayesianNetwork) (*MixedVariableElimination, error) {
	if err := model.CheckModel(); err != nil {
		return nil, err
	}
	return &MixedVariableElimination{Model: model}, nil
}

// Query computes P(variables | evidence) for mixed models
func (mve *MixedVariableElimination) Query(discreteVars, continuousVars []string, evidence MixedEvidence) (*MixedQueryResult, error) {
	allDiscrete := true
	for _, node := range mve.Model.Nodes() {
		if mve.Model.IsContinuous(node) {
			allDiscrete = false
			break
		}
	}

	allContinuous := true
	for _, node := range mve.Model.Nodes() {
		if mve.Model.IsDiscrete(node) {
			allContinuous = false
			break
		}
	}

	if allDiscrete && len(continuousVars) > 0 {
		return nil, fmt.Errorf("model is discrete-only, cannot query continuous variables")
	}
	if allContinuous && len(discreteVars) > 0 {
		return nil, fmt.Errorf("model is continuous-only, cannot query discrete variables")
	}

	if allDiscrete {
		return mve.queryDiscrete(discreteVars, evidence)
	}

	if allContinuous {
		return mve.queryContinuous(continuousVars, evidence)
	}

	return mve.queryMixedCLG(discreteVars, continuousVars, evidence)
}

// queryDiscrete handles queries on purely discrete models
func (mve *MixedVariableElimination) queryDiscrete(discreteVars []string, evidence MixedEvidence) (*MixedQueryResult, error) {
	ve, err := NewVariableElimination(mve.Model)
	if err != nil {
		return nil, err
	}

	result, err := ve.Query(discreteVars, evidence.Discrete)
	if err != nil {
		return nil, err
	}

	return &MixedQueryResult{
		DiscreteVars: discreteVars,
		Cardinality:  result.Cardinality,
		Values:       result.Values,
	}, nil
}

// queryContinuous handles queries on purely continuous models
func (mve *MixedVariableElimination) queryContinuous(continuousVars []string, evidence MixedEvidence) (*MixedQueryResult, error) {
	factorList := make([]*factors.GaussianFactor, 0)
	for _, cpd := range mve.Model.GetGaussianCPDs() {
		factor, err := cpd.ToFactor()
		if err != nil {
			continue
		}
		factorList = append(factorList, factor)
	}

	if len(factorList) == 0 {
		return nil, fmt.Errorf("no compatible Gaussian factors found")
	}

	for i := range factorList {
		reduced, err := factorList[i].Reduce(evidence.Continuous)
		if err == nil {
			factorList[i] = reduced
		}
	}

	allVars := make(map[string]bool)
	for _, node := range mve.Model.Nodes() {
		allVars[node] = true
	}

	for _, v := range continuousVars {
		delete(allVars, v)
	}

	for v := range evidence.Continuous {
		delete(allVars, v)
	}

	toEliminate := make([]string, 0, len(allVars))
	for v := range allVars {
		toEliminate = append(toEliminate, v)
	}
	sort.Strings(toEliminate)

	currentFactors := factorList
	for _, v := range toEliminate {
		currentFactors = mve.eliminateContinuousVariable(v, currentFactors)
	}

	if len(currentFactors) == 0 {
		return nil, fmt.Errorf("no factors remaining")
	}

	result := currentFactors[0]
	for i := 1; i < len(currentFactors); i++ {
		newResult, err := result.Multiply(currentFactors[i])
		if err != nil {
			return nil, err
		}
		result = newResult
	}

	return &MixedQueryResult{
		ContinuousVars: continuousVars,
		Mean:           result.Mean,
		Covariance:     result.Covariance,
	}, nil
}

// queryMixedCLG handles queries on mixed models
func (mve *MixedVariableElimination) queryMixedCLG(discreteVars, continuousVars []string, evidence MixedEvidence) (*MixedQueryResult, error) {
	if len(continuousVars) == 0 {
		return mve.queryDiscreteInMixed(discreteVars, evidence)
	}

	if len(discreteVars) == 0 {
		return mve.queryContinuousInMixed(continuousVars, evidence)
	}

	return nil, fmt.Errorf("queries on both discrete and continuous variables not yet implemented")
}

// queryDiscreteInMixed handles discrete queries in mixed models
func (mve *MixedVariableElimination) queryDiscreteInMixed(discreteVars []string, evidence MixedEvidence) (*MixedQueryResult, error) {
	factorList := make([]*factors.DiscreteFactor, 0)
	for _, cpd := range mve.Model.GetCPDs() {
		factor, err := cpd.ToFactor()
		if err != nil {
			return nil, err
		}
		factorList = append(factorList, factor)
	}

	reducedFactors := make([]*factors.DiscreteFactor, 0)
	for _, factor := range factorList {
		reduced, err := factor.Reduce(evidence.Discrete)
		if err != nil {
			return nil, err
		}
		reducedFactors = append(reducedFactors, reduced)
	}

	allDiscreteVars := make(map[string]bool)
	for _, node := range mve.Model.Nodes() {
		if mve.Model.IsDiscrete(node) {
			allDiscreteVars[node] = true
		}
	}

	for _, v := range discreteVars {
		delete(allDiscreteVars, v)
	}

	for v := range evidence.Discrete {
		delete(allDiscreteVars, v)
	}

	toEliminate := make([]string, 0)
	for v := range allDiscreteVars {
		toEliminate = append(toEliminate, v)
	}
	sort.Strings(toEliminate)

	currentFactors := reducedFactors
	for _, v := range toEliminate {
		currentFactors = mve.eliminateDiscreteVariable(v, currentFactors)
	}

	if len(currentFactors) == 0 {
		return nil, fmt.Errorf("no factors remaining")
	}

	result := currentFactors[0]
	for i := 1; i < len(currentFactors); i++ {
		newResult, err := result.Multiply(currentFactors[i])
		if err != nil {
			return nil, err
		}
		result = newResult
	}

	if err := result.Normalize(); err != nil {
		return nil, err
	}

	return &MixedQueryResult{
		DiscreteVars: discreteVars,
		Cardinality:  result.Cardinality,
		Values:       result.Values,
	}, nil
}

// queryContinuousInMixed handles continuous queries in mixed models
func (mve *MixedVariableElimination) queryContinuousInMixed(continuousVars []string, evidence MixedEvidence) (*MixedQueryResult, error) {
	gaussianFactors := make([]*factors.GaussianFactor, 0)
	for _, cpd := range mve.Model.GetGaussianCPDs() {
		if mve.allContinuousParents(cpd) {
			factor, err := cpd.ToFactor()
			if err == nil {
				gaussianFactors = append(gaussianFactors, factor)
			}
		}
	}

	if len(gaussianFactors) == 0 {
		return nil, fmt.Errorf("no suitable Gaussian factors")
	}

	for i := range gaussianFactors {
		reduced, err := gaussianFactors[i].Reduce(evidence.Continuous)
		if err == nil {
			gaussianFactors[i] = reduced
		}
	}

	allContinuousVars := make(map[string]bool)
	for _, node := range mve.Model.Nodes() {
		if mve.Model.IsContinuous(node) {
			allContinuousVars[node] = true
		}
	}

	for _, v := range continuousVars {
		delete(allContinuousVars, v)
	}

	for v := range evidence.Continuous {
		delete(allContinuousVars, v)
	}

	toEliminate := make([]string, 0)
	for v := range allContinuousVars {
		toEliminate = append(toEliminate, v)
	}
	sort.Strings(toEliminate)

	currentFactors := gaussianFactors
	for _, v := range toEliminate {
		currentFactors = mve.eliminateContinuousVariable(v, currentFactors)
	}

	if len(currentFactors) == 0 {
		return nil, fmt.Errorf("no factors remaining")
	}

	result := currentFactors[0]
	for i := 1; i < len(currentFactors); i++ {
		newResult, err := result.Multiply(currentFactors[i])
		if err != nil {
			return nil, err
		}
		result = newResult
	}

	return &MixedQueryResult{
		ContinuousVars: continuousVars,
		Mean:           result.Mean,
		Covariance:     result.Covariance,
	}, nil
}

func (mve *MixedVariableElimination) eliminateDiscreteVariable(variable string, factorList []*factors.DiscreteFactor) []*factors.DiscreteFactor {
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

	product := relevant[0]
	for i := 1; i < len(relevant); i++ {
		newProduct, err := product.Multiply(relevant[i])
		if err != nil {
			continue
		}
		product = newProduct
	}

	marginalized, err := product.Marginalize([]string{variable})
	if err != nil {
		return irrelevant
	}

	return append(irrelevant, marginalized)
}

func (mve *MixedVariableElimination) eliminateContinuousVariable(variable string, factorList []*factors.GaussianFactor) []*factors.GaussianFactor {
	relevant := make([]*factors.GaussianFactor, 0)
	irrelevant := make([]*factors.GaussianFactor, 0)

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

	product := relevant[0]
	for i := 1; i < len(relevant); i++ {
		newProduct, err := product.Multiply(relevant[i])
		if err != nil {
			continue
		}
		product = newProduct
	}

	marginalized, err := product.Marginalize([]string{variable})
	if err != nil {
		return irrelevant
	}

	return append(irrelevant, marginalized)
}

func (mve *MixedVariableElimination) allContinuousParents(cpd *factors.LinearGaussianCPD) bool {
	for _, ptype := range cpd.ParentTypes {
		if ptype != "continuous" {
			return false
		}
	}
	return true
}
