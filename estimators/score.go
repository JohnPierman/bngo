package estimators

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/JohnPierman/bngo/graph"
)

// ScoringMethod is a decomposable network score: the score of a network is the sum
// of the local score of every variable given its parents. Search depends on that
// decomposition, because it lets one edge change be priced by rescoring one or two
// families instead of the whole network.
type ScoringMethod interface {
	// LocalScore returns the score of one variable given a parent set.
	LocalScore(variable string, parents []string) (float64, error)
	// Name identifies the score in diagnostics.
	Name() string
}

// StructureEstimator is implemented by every structure learning algorithm, so
// callers can swap one for another.
type StructureEstimator interface {
	Estimate() (*graph.DAG, error)
}

// familyCounts holds how often each state of a variable occurs under each parent
// configuration that actually appears in the data. Configurations that never occur
// are absent rather than stored as zeros, which keeps the memory in proportion to
// the sample instead of to the product of the parent cardinalities. That is what
// makes scoring a large parent set on a large network affordable.
type familyCounts struct {
	perConfig map[int64][]float64
	totals    map[int64]float64
}

// count builds the sufficient statistics of one family, skipping rows where the
// variable or any of its parents is unobserved. It also returns how many
// configurations the parent set can take, which the penalty terms need.
func (d *columnData) count(variable string, parents []string) (*familyCounts, float64, error) {
	childCard := d.states(variable)
	if childCard <= 0 {
		return nil, 0, fmt.Errorf("variable %s has no observed states", variable)
	}

	child, ok := d.column(variable)
	if !ok {
		return nil, 0, fmt.Errorf("variable %s is not in the data", variable)
	}

	indexer, err := d.indexer(parents)
	if err != nil {
		return nil, 0, err
	}

	counts := &familyCounts{
		perConfig: make(map[int64][]float64),
		totals:    make(map[int64]float64),
	}

	for row := 0; row < d.rows; row++ {
		value := child[row]
		if value == missingState {
			continue
		}

		config, ok := indexer.at(row)
		if !ok {
			continue
		}

		states, seen := counts.perConfig[config]
		if !seen {
			states = make([]float64, childCard)
			counts.perConfig[config] = states
		}
		states[value]++
		counts.totals[config]++
	}

	return counts, indexer.configs, nil
}

// localScorer prices one family from its sufficient statistics.
type localScorer func(counts *familyCounts, numConfigs float64, childCard int) float64

// Score is a decomposable network score. It caches the local score of every family
// it is asked about, because search asks about the same family many times over. A
// Score is therefore not safe for concurrent use.
type Score struct {
	data   *columnData
	scorer localScorer
	name   string
	cache  map[string]float64
}

// newScore assembles a score from its data and its per family term.
func newScore(data *columnData, name string, scorer localScorer) *Score {
	return &Score{
		data:   data,
		scorer: scorer,
		name:   name,
		cache:  make(map[string]float64),
	}
}

// NewBIC returns the Bayesian Information Criterion, the log likelihood penalised by
// half the log of the sample size per free parameter. BIC is consistent, and its
// comparatively heavy penalty keeps parent sets small, which is usually what you want
// on a large network.
//
// A non-empty entry in cardinality overrides the number of states read off the data;
// pass nil to take every cardinality from the data.
func NewBIC(data []map[string]int, cardinality map[string]int) *Score {
	return newBIC(newColumnData(data, cardinality))
}

// NewAIC returns Akaike's Information Criterion, the log likelihood penalised by one
// per free parameter. Its lighter penalty admits denser networks than BIC.
func NewAIC(data []map[string]int, cardinality map[string]int) *Score {
	return newScore(newColumnData(data, cardinality), "AIC", penalizedLikelihood(1.0))
}

// NewK2 returns the K2 score of Cooper and Herskovits: the marginal likelihood under
// a Dirichlet prior that puts one pseudo-count on every cell.
func NewK2(data []map[string]int, cardinality map[string]int) *Score {
	uniform := func(_, _ float64) float64 { return 1.0 }
	return newScore(newColumnData(data, cardinality), "K2", dirichlet(uniform))
}

// NewBDeu returns the Bayesian Dirichlet equivalent uniform score. The
// equivalentSampleSize is the total prior weight, spread evenly over the cells of the
// table, so unlike K2 the strength of the prior does not grow with the size of the
// parent set. It must be positive; 10 is a common choice.
//
// BDeu is score equivalent, meaning two structures encoding the same independencies
// receive the same score, which is what makes it the usual choice for the greedy
// search inside MMHC.
func NewBDeu(data []map[string]int, cardinality map[string]int, equivalentSampleSize float64) (*Score, error) {
	return newBDeu(newColumnData(data, cardinality), equivalentSampleSize)
}

// newBIC builds BIC over an existing columnar view.
func newBIC(data *columnData) *Score {
	penalty := 0.0
	if data.rows > 0 {
		penalty = 0.5 * math.Log(float64(data.rows))
	}
	return newScore(data, "BIC", penalizedLikelihood(penalty))
}

// newBDeu builds BDeu over an existing columnar view.
func newBDeu(data *columnData, equivalentSampleSize float64) (*Score, error) {
	if equivalentSampleSize <= 0 {
		return nil, fmt.Errorf("BDeu: equivalent sample size must be positive, got %v", equivalentSampleSize)
	}

	spread := func(numConfigs, childCard float64) float64 {
		return equivalentSampleSize / (numConfigs * childCard)
	}

	return newScore(data, "BDeu", dirichlet(spread)), nil
}

// penalizedLikelihood builds the shared body of BIC and AIC, which differ only in
// what one free parameter costs.
func penalizedLikelihood(penaltyPerParameter float64) localScorer {
	return func(counts *familyCounts, numConfigs float64, childCard int) float64 {
		freeParameters := numConfigs * float64(childCard-1)
		return logLikelihood(counts) - penaltyPerParameter*freeParameters
	}
}

// logLikelihood returns the maximised log likelihood of one family. Cells that were
// never observed contribute nothing, so only the configurations present in the counts
// are visited.
func logLikelihood(counts *familyCounts) float64 {
	total := 0.0

	for config, states := range counts.perConfig {
		configTotal := counts.totals[config]
		for _, count := range states {
			if count > 0 {
				total += count * math.Log(count/configTotal)
			}
		}
	}

	return total
}

// dirichlet builds the shared body of the Bayesian scores, which differ only in how
// the prior weight of one cell is chosen. Parent configurations absent from the data
// contribute exactly zero, so skipping them is exact rather than an approximation.
func dirichlet(alphaCell func(numConfigs, childCard float64) float64) localScorer {
	return func(counts *familyCounts, numConfigs float64, childCard int) float64 {
		cell := alphaCell(numConfigs, float64(childCard))
		perConfig := cell * float64(childCard)

		lgammaCell, _ := math.Lgamma(cell)
		lgammaConfig, _ := math.Lgamma(perConfig)

		total := 0.0
		for config, states := range counts.perConfig {
			observed, _ := math.Lgamma(perConfig + counts.totals[config])
			total += lgammaConfig - observed

			for _, count := range states {
				if count > 0 {
					shifted, _ := math.Lgamma(cell + count)
					total += shifted - lgammaCell
				}
			}
		}

		return total
	}
}

// LocalScore returns the score of a variable given a parent set, reusing a cached
// value when the same family has been priced before. The parent slice is not
// modified.
func (s *Score) LocalScore(variable string, parents []string) (float64, error) {
	sorted := sortedCopy(parents)
	key := familyKey(variable, sorted)

	if cached, ok := s.cache[key]; ok {
		return cached, nil
	}

	counts, numConfigs, err := s.data.count(variable, sorted)
	if err != nil {
		return 0, fmt.Errorf("local score of %s: %w", variable, err)
	}

	score := s.scorer(counts, numConfigs, s.data.states(variable))
	s.cache[key] = score

	return score, nil
}

// Name returns the name of the score.
func (s *Score) Name() string {
	return s.name
}

// Variables returns the variables of the data in sorted order.
func (s *Score) Variables() []string {
	variables := make([]string, len(s.data.variables))
	copy(variables, s.data.variables)
	return variables
}

// Cardinality returns the number of states of each variable.
func (s *Score) Cardinality() map[string]int {
	cardinality := make(map[string]int, len(s.data.cardinality))
	for variable, card := range s.data.cardinality {
		cardinality[variable] = card
	}
	return cardinality
}

// SampleSize returns the number of rows the score was built from.
func (s *Score) SampleSize() int {
	return s.data.rows
}

// ScoreDAG returns the score of a whole network as the sum of its local scores.
func ScoreDAG(dag *graph.DAG, score ScoringMethod) (float64, error) {
	if dag == nil {
		return 0, fmt.Errorf("score DAG: no graph given")
	}
	if score == nil {
		return 0, fmt.Errorf("score DAG: no scoring method given")
	}

	total := 0.0
	for _, node := range dag.Nodes() {
		local, err := score.LocalScore(node, dag.Parents(node))
		if err != nil {
			return 0, err
		}
		total += local
	}

	return total, nil
}

// familyKey identifies a variable and a sorted parent set. The separator cannot occur
// in a variable name read from a CSV header, so distinct families cannot collide on
// one key.
func familyKey(variable string, sortedParents []string) string {
	return variable + "\x00" + strings.Join(sortedParents, "\x00")
}

// sortedCopy returns the values sorted, leaving the argument untouched.
func sortedCopy(values []string) []string {
	sorted := make([]string, len(values))
	copy(sorted, values)
	sort.Strings(sorted)
	return sorted
}
