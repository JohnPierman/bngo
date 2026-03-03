package inference

import (
	"fmt"
	"sort"

	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/models"
)

// MixedEvidence holds both discrete and continuous observed values for inference queries.
type MixedEvidence struct {
	Discrete   map[string]int
	Continuous map[string]float64
}

// ContinuousResult holds the posterior distribution over continuous query variables.
// When hidden discrete variables exist, this represents the moment-matched
// Gaussian approximation of the true Gaussian mixture posterior.
type ContinuousResult struct {
	Variables  []string
	Mean       map[string]float64
	Covariance map[string]map[string]float64
}

// MixedVariableElimination performs exact inference on Bayesian networks
// containing both discrete and continuous (Linear Gaussian) variables.
//
// Supported network types:
//   - Purely continuous (all Linear Gaussian CPDs)
//   - Mixed discrete/continuous (Conditional Linear Gaussian models)
//   - Purely discrete (delegates to standard VariableElimination)
//
// In CLG models, discrete variables must have only discrete parents.
// Continuous variables may have discrete parents, continuous parents, or both
// (where discrete parents index different Gaussian parameter sets).
type MixedVariableElimination struct {
	Model *models.BayesianNetwork
}

// NewMixedVariableElimination creates a new mixed inference engine.
func NewMixedVariableElimination(model *models.BayesianNetwork) (*MixedVariableElimination, error) {
	if err := model.CheckModel(); err != nil {
		return nil, err
	}
	return &MixedVariableElimination{Model: model}, nil
}

// QueryContinuous computes the posterior P(queryVars | evidence) for continuous
// query variables. Returns the posterior mean and covariance.
//
// The algorithm enumerates all configurations of hidden discrete variables,
// builds a conditional Gaussian for each, and computes a weighted mixture.
func (mve *MixedVariableElimination) QueryContinuous(
	queryVars []string,
	evidence MixedEvidence,
) (*ContinuousResult, error) {
	if len(queryVars) == 0 {
		return nil, fmt.Errorf("query variables cannot be empty")
	}

	sorted := make([]string, len(queryVars))
	copy(sorted, queryVars)
	sort.Strings(sorted)
	queryVars = sorted

	if err := mve.validateContinuousQuery(queryVars, evidence); err != nil {
		return nil, err
	}

	hiddenDiscrete := mve.findHiddenDiscreteVars(evidence)
	configs := enumerateDiscreteConfigs(hiddenDiscrete, mve.Model.Cardinality)

	results := make([]weightedGaussianResult, 0, len(configs))

	for _, hiddenConfig := range configs {
		fullConfig := mergeDiscreteConfigs(evidence.Discrete, hiddenConfig)

		discreteProb := mve.computeDiscreteProb(fullConfig)
		if discreteProb < 1e-300 {
			continue
		}

		joint, err := mve.buildJointGaussian(fullConfig)
		if err != nil {
			return nil, fmt.Errorf("building joint Gaussian: %w", err)
		}

		weight := discreteProb

		if len(evidence.Continuous) > 0 {
			likelihood, lErr := mve.continuousLikelihood(joint, evidence.Continuous)
			if lErr != nil {
				return nil, fmt.Errorf("computing continuous likelihood: %w", lErr)
			}
			weight *= likelihood

			joint, err = joint.Reduce(evidence.Continuous)
			if err != nil {
				return nil, fmt.Errorf("conditioning on continuous evidence: %w", err)
			}
		}

		marginal, err := extractMarginal(joint, queryVars)
		if err != nil {
			return nil, fmt.Errorf("extracting marginal: %w", err)
		}

		results = append(results, weightedGaussianResult{
			weight: weight,
			mean:   marginal.Mean,
			cov:    marginal.Covariance,
		})
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("no valid discrete configurations found")
	}

	return mve.computeMixtureResult(queryVars, results)
}

// QueryDiscrete computes P(queryVars | evidence) for discrete query variables.
// Supports continuous evidence via likelihood weighting over discrete configurations.
func (mve *MixedVariableElimination) QueryDiscrete(
	queryVars []string,
	evidence MixedEvidence,
) (*factors.DiscreteFactor, error) {
	if len(queryVars) == 0 {
		return nil, fmt.Errorf("query variables cannot be empty")
	}

	sorted := make([]string, len(queryVars))
	copy(sorted, queryVars)
	sort.Strings(sorted)
	queryVars = sorted

	if err := mve.validateDiscreteQuery(queryVars, evidence); err != nil {
		return nil, err
	}

	if len(evidence.Continuous) == 0 {
		ve := &VariableElimination{Model: mve.Model}
		return ve.Query(queryVars, evidence.Discrete)
	}

	return mve.queryDiscreteWithContinuousEvidence(queryVars, evidence)
}

func (mve *MixedVariableElimination) queryDiscreteWithContinuousEvidence(
	queryVars []string,
	evidence MixedEvidence,
) (*factors.DiscreteFactor, error) {
	enumVars := make([]string, 0)
	for _, node := range mve.Model.Nodes() {
		if !mve.Model.IsDiscrete(node) {
			continue
		}
		if evidence.Discrete != nil {
			if _, ok := evidence.Discrete[node]; ok {
				continue
			}
		}
		enumVars = append(enumVars, node)
	}

	configs := enumerateDiscreteConfigs(enumVars, mve.Model.Cardinality)

	queryCard := make(map[string]int)
	for _, v := range queryVars {
		queryCard[v] = mve.Model.Cardinality[v]
	}

	resultSize := 1
	for _, v := range queryVars {
		resultSize *= queryCard[v]
	}
	resultValues := make([]float64, resultSize)

	for _, config := range configs {
		fullConfig := mergeDiscreteConfigs(evidence.Discrete, config)

		discreteProb := mve.computeDiscreteProb(fullConfig)
		if discreteProb < 1e-300 {
			continue
		}

		joint, err := mve.buildJointGaussian(fullConfig)
		if err != nil {
			continue
		}

		likelihood, err := mve.continuousLikelihood(joint, evidence.Continuous)
		if err != nil {
			continue
		}

		weight := discreteProb * likelihood

		idx := 0
		stride := 1
		for i := len(queryVars) - 1; i >= 0; i-- {
			v := queryVars[i]
			idx += config[v] * stride
			stride *= queryCard[v]
		}
		resultValues[idx] += weight
	}

	result, err := factors.NewDiscreteFactor(queryVars, queryCard, resultValues)
	if err != nil {
		return nil, err
	}

	if err := result.Normalize(); err != nil {
		return nil, fmt.Errorf("normalizing result: %w (evidence may be impossible)", err)
	}

	return result, nil
}

// buildJointGaussian constructs the joint Gaussian distribution over all
// continuous variables, conditioned on a specific discrete variable assignment.
//
// For each continuous node processed in topological order:
//   - Root nodes (no parents) contribute their prior mean and variance.
//   - Nodes with discrete parents use the Gaussian parameters indexed
//     by the parent configuration.
//   - Nodes with continuous parents use the linear Gaussian relationship
//     to compute mean, variance, and cross-covariances incrementally.
func (mve *MixedVariableElimination) buildJointGaussian(
	discreteConfig map[string]int,
) (*factors.GaussianFactor, error) {
	order, err := mve.Model.DAG.TopologicalSort()
	if err != nil {
		return nil, err
	}

	continuousVars := make([]string, 0)
	for _, node := range order {
		if mve.Model.IsContinuous(node) {
			continuousVars = append(continuousVars, node)
		}
	}

	if len(continuousVars) == 0 {
		return nil, fmt.Errorf("no continuous variables in network")
	}

	mean := make(map[string]float64)
	cov := make(map[string]map[string]float64)
	for _, v := range continuousVars {
		cov[v] = make(map[string]float64)
	}

	for _, node := range continuousVars {
		cpd, ok := mve.Model.GaussianCPDs[node]
		if !ok {
			return nil, fmt.Errorf("no Gaussian CPD for variable %s", node)
		}

		if len(cpd.DiscreteStates) > 0 {
			if err := mve.processDiscreteParentNode(node, cpd, discreteConfig, continuousVars, mean, cov); err != nil {
				return nil, err
			}
		} else {
			mve.processContinuousParentNode(node, cpd, continuousVars, mean, cov)
		}
	}

	ensureSymmetricCovariance(continuousVars, cov)

	return factors.NewGaussianFactor(continuousVars, mean, cov)
}

func (mve *MixedVariableElimination) processDiscreteParentNode(
	node string,
	cpd *factors.LinearGaussianCPD,
	discreteConfig map[string]int,
	continuousVars []string,
	mean map[string]float64,
	cov map[string]map[string]float64,
) error {
	stateKey := buildStateKey(cpd.Parents, discreteConfig)
	params, found := cpd.DiscreteStates[stateKey]
	if !found {
		return fmt.Errorf("no parameters for state key %q of variable %s", stateKey, node)
	}

	mean[node] = params.Mean
	cov[node][node] = params.Variance

	for _, other := range continuousVars {
		if other == node {
			continue
		}
		if _, exists := cov[node][other]; !exists {
			cov[node][other] = 0.0
		}
		if _, exists := cov[other][node]; !exists {
			cov[other][node] = 0.0
		}
	}
	return nil
}

// processContinuousParentNode computes the joint distribution parameters for a
// node with continuous parents using the linear relationship:
//
//	E[X] = β₀ + Σ βᵢ E[Yᵢ]
//	Cov(X, Z) = Σ βᵢ Cov(Yᵢ, Z) for any node Z
//	Var(X) = Σᵢⱼ βᵢ βⱼ Cov(Yᵢ, Yⱼ) + σ²
func (mve *MixedVariableElimination) processContinuousParentNode(
	node string,
	cpd *factors.LinearGaussianCPD,
	continuousVars []string,
	mean map[string]float64,
	cov map[string]map[string]float64,
) {
	nodeMean := cpd.Intercept
	for _, parent := range cpd.Parents {
		nodeMean += cpd.Coefficients[parent] * mean[parent]
	}
	mean[node] = nodeMean

	nodeVar := cpd.Variance
	for _, pi := range cpd.Parents {
		for _, pj := range cpd.Parents {
			nodeVar += cpd.Coefficients[pi] * cpd.Coefficients[pj] * cov[pi][pj]
		}
	}
	cov[node][node] = nodeVar

	for _, other := range continuousVars {
		if other == node {
			continue
		}
		covVal := 0.0
		for _, parent := range cpd.Parents {
			if c, exists := cov[parent][other]; exists {
				covVal += cpd.Coefficients[parent] * c
			}
		}
		cov[node][other] = covVal
		cov[other][node] = covVal
	}
}

// computeDiscreteProb computes P(discrete configuration) using the chain rule
// over all discrete CPDs in the model.
func (mve *MixedVariableElimination) computeDiscreteProb(config map[string]int) float64 {
	prob := 1.0
	for variable, cpd := range mve.Model.CPDs {
		varState, ok := config[variable]
		if !ok {
			return 0.0
		}

		evidenceValues := make(map[string]int)
		for _, parent := range cpd.Evidence {
			val, pOk := config[parent]
			if !pOk {
				return 0.0
			}
			evidenceValues[parent] = val
		}

		p, err := cpd.GetValue(varState, evidenceValues)
		if err != nil {
			return 0.0
		}
		prob *= p
	}
	return prob
}

// continuousLikelihood computes P(continuous evidence | discrete config) by
// marginalizing the joint Gaussian to the evidence variables and evaluating the PDF.
func (mve *MixedVariableElimination) continuousLikelihood(
	joint *factors.GaussianFactor,
	evidence map[string]float64,
) (float64, error) {
	evidenceSet := make(map[string]bool, len(evidence))
	for v := range evidence {
		evidenceSet[v] = true
	}

	toMarginalize := make([]string, 0)
	for _, v := range joint.Variables {
		if !evidenceSet[v] {
			toMarginalize = append(toMarginalize, v)
		}
	}

	marginal := joint
	if len(toMarginalize) > 0 {
		var err error
		marginal, err = joint.Marginalize(toMarginalize)
		if err != nil {
			return 0, fmt.Errorf("marginalizing for likelihood: %w", err)
		}
	}

	return marginal.PDF(evidence)
}

type weightedGaussianResult struct {
	weight float64
	mean   map[string]float64
	cov    map[string]map[string]float64
}

func (mve *MixedVariableElimination) computeMixtureResult(
	queryVars []string,
	results []weightedGaussianResult,
) (*ContinuousResult, error) {
	totalWeight := 0.0
	for _, r := range results {
		totalWeight += r.weight
	}
	if totalWeight < 1e-300 {
		return nil, fmt.Errorf("total weight is zero; evidence may be impossible")
	}

	for i := range results {
		results[i].weight /= totalWeight
	}

	mixtureMean := make(map[string]float64)
	for _, v := range queryVars {
		for _, r := range results {
			mixtureMean[v] += r.weight * r.mean[v]
		}
	}

	mixtureCov := make(map[string]map[string]float64)
	for _, v1 := range queryVars {
		mixtureCov[v1] = make(map[string]float64)
		for _, v2 := range queryVars {
			val := 0.0
			for _, r := range results {
				val += r.weight * (r.cov[v1][v2] + r.mean[v1]*r.mean[v2])
			}
			val -= mixtureMean[v1] * mixtureMean[v2]
			mixtureCov[v1][v2] = val
		}
	}

	return &ContinuousResult{
		Variables:  queryVars,
		Mean:       mixtureMean,
		Covariance: mixtureCov,
	}, nil
}

func (mve *MixedVariableElimination) validateContinuousQuery(
	queryVars []string,
	evidence MixedEvidence,
) error {
	for _, v := range queryVars {
		if !mve.Model.IsContinuous(v) {
			return fmt.Errorf("query variable %s is not continuous", v)
		}
		if evidence.Discrete != nil {
			if _, ok := evidence.Discrete[v]; ok {
				return fmt.Errorf("query variable %s is also in discrete evidence", v)
			}
		}
		if evidence.Continuous != nil {
			if _, ok := evidence.Continuous[v]; ok {
				return fmt.Errorf("query variable %s is also in continuous evidence", v)
			}
		}
	}
	return mve.validateEvidence(evidence)
}

func (mve *MixedVariableElimination) validateDiscreteQuery(
	queryVars []string,
	evidence MixedEvidence,
) error {
	for _, v := range queryVars {
		if !mve.Model.IsDiscrete(v) {
			return fmt.Errorf("query variable %s is not discrete", v)
		}
		if evidence.Discrete != nil {
			if _, ok := evidence.Discrete[v]; ok {
				return fmt.Errorf("query variable %s is also in discrete evidence", v)
			}
		}
	}
	return mve.validateEvidence(evidence)
}

func (mve *MixedVariableElimination) validateEvidence(evidence MixedEvidence) error {
	nodes := make(map[string]bool)
	for _, n := range mve.Model.Nodes() {
		nodes[n] = true
	}

	for v := range evidence.Discrete {
		if !nodes[v] {
			return fmt.Errorf("discrete evidence variable %s not in model", v)
		}
		if !mve.Model.IsDiscrete(v) {
			return fmt.Errorf("discrete evidence variable %s is not a discrete variable", v)
		}
	}

	for v := range evidence.Continuous {
		if !nodes[v] {
			return fmt.Errorf("continuous evidence variable %s not in model", v)
		}
		if !mve.Model.IsContinuous(v) {
			return fmt.Errorf("continuous evidence variable %s is not a continuous variable", v)
		}
	}

	return nil
}

func (mve *MixedVariableElimination) findHiddenDiscreteVars(evidence MixedEvidence) []string {
	hidden := make([]string, 0)
	for _, node := range mve.Model.Nodes() {
		if !mve.Model.IsDiscrete(node) {
			continue
		}
		if evidence.Discrete != nil {
			if _, ok := evidence.Discrete[node]; ok {
				continue
			}
		}
		hidden = append(hidden, node)
	}
	return hidden
}

func extractMarginal(joint *factors.GaussianFactor, vars []string) (*factors.GaussianFactor, error) {
	querySet := make(map[string]bool, len(vars))
	for _, v := range vars {
		querySet[v] = true
	}

	toMarginalize := make([]string, 0)
	for _, v := range joint.Variables {
		if !querySet[v] {
			toMarginalize = append(toMarginalize, v)
		}
	}

	if len(toMarginalize) == 0 {
		return joint, nil
	}

	return joint.Marginalize(toMarginalize)
}

func mergeDiscreteConfigs(a, b map[string]int) map[string]int {
	result := make(map[string]int, len(a)+len(b))
	for k, v := range a {
		result[k] = v
	}
	for k, v := range b {
		result[k] = v
	}
	return result
}

func buildStateKey(parents []string, config map[string]int) string {
	key := ""
	for i, parent := range parents {
		if i > 0 {
			key += ","
		}
		key += fmt.Sprintf("%v", config[parent])
	}
	return key
}

func ensureSymmetricCovariance(vars []string, cov map[string]map[string]float64) {
	for _, v1 := range vars {
		for _, v2 := range vars {
			if _, ok := cov[v1][v2]; !ok {
				cov[v1][v2] = 0.0
			}
		}
	}
}

func enumerateDiscreteConfigs(vars []string, cardinality map[string]int) []map[string]int {
	if len(vars) == 0 {
		return []map[string]int{{}}
	}

	total := 1
	for _, v := range vars {
		total *= cardinality[v]
	}

	configs := make([]map[string]int, total)
	for i := range configs {
		configs[i] = make(map[string]int)
		idx := i
		for j := len(vars) - 1; j >= 0; j-- {
			card := cardinality[vars[j]]
			configs[i][vars[j]] = idx % card
			idx /= card
		}
	}

	return configs
}
