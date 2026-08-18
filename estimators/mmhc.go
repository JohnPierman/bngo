package estimators

import (
	"strings"

	"github.com/JohnPierman/bngo/graph"
)

const (
	// defaultAlpha is the significance level of the independence tests.
	defaultAlpha = 0.05
	// defaultMaxConditioningSetSize bounds how large a conditioning set may grow.
	// The number of subsets to test grows exponentially in this bound, and the
	// tests lose power as the strata thin out, so it is capped rather than left to
	// reach the size of the neighbourhood.
	defaultMaxConditioningSetSize = 3
	// defaultEquivalentSampleSize is the prior weight of the BDeu score MMHC
	// searches with, following the original description of the algorithm.
	defaultEquivalentSampleSize = 10.0
)

// MMHCEstimator implements Max-Min Hill Climbing, the hybrid structure learning
// algorithm of Tsamardinos, Brown and Aliferis (2006).
//
// It runs in two phases:
//
//  1. Max-Min Parents and Children (MMPC) finds, for every variable, a small set
//     of candidate neighbours, using conditional independence tests only. The
//     forward phase repeatedly admits the variable whose association with the
//     target survives conditioning best; the backward phase drops any candidate
//     that turns out independent of the target given some subset of the others.
//     Keeping only the pairs that name each other leaves an undirected skeleton.
//  2. Greedy hill climbing then searches for the highest scoring network, but is
//     restricted to the edges of that skeleton, which also settles their direction.
//
// The restriction is what makes the algorithm suited to large networks. An
// unrestricted greedy step prices O(n^2) candidate changes on n variables, while a
// restricted step prices only as many as the skeleton allows, which on a sparse
// network is closer to O(n). MMPC pays for that with local tests whose cost
// depends on the size of a neighbourhood rather than on n.
//
// Compared with the PC algorithm in this package, MMHC tests conditioning sets
// drawn from a candidate neighbourhood rather than from every neighbour of a node
// in a graph that starts out complete, and it orients edges by score rather than
// by v-structures and Meek rules.
type MMHCEstimator struct {
	columns                *columnData
	variables              []string
	cardinality            map[string]int
	alpha                  float64
	maxConditioningSetSize int
	maxIndegree            int
	score                  ScoringMethod
	pValues                map[string]float64
}

// NewMMHC creates an MMHC estimator over the variables of the data. A non-empty
// entry in cardinality overrides the number of states read off the data; pass nil
// to take every cardinality from the data.
func NewMMHC(data []map[string]int, cardinality map[string]int) *MMHCEstimator {
	columns := newColumnData(data, cardinality)

	return &MMHCEstimator{
		columns:                columns,
		variables:              columns.variables,
		cardinality:            columns.cardinality,
		alpha:                  defaultAlpha,
		maxConditioningSetSize: defaultMaxConditioningSetSize,
		pValues:                make(map[string]float64),
	}
}

// SetAlpha sets the significance level of the independence tests. A smaller value
// admits fewer candidate neighbours, so the skeleton comes out sparser.
func (m *MMHCEstimator) SetAlpha(alpha float64) {
	m.alpha = alpha
}

// SetMaxConditioningSetSize bounds the size of the conditioning sets MMPC tests.
// A value of zero or less removes the bound, at exponential cost.
func (m *MMHCEstimator) SetMaxConditioningSetSize(size int) {
	m.maxConditioningSetSize = size
}

// SetMaxIndegree caps how many parents a variable may take during the search. A
// value of zero or less leaves it uncapped.
func (m *MMHCEstimator) SetMaxIndegree(parents int) {
	m.maxIndegree = parents
}

// SetScore replaces the score the search phase maximises. The default is BDeu with
// an equivalent sample size of 10.
func (m *MMHCEstimator) SetScore(score ScoringMethod) {
	m.score = score
}

// Estimate learns the structure: MMPC narrows the candidate parents, then greedy
// search picks and orients the edges among them.
func (m *MMHCEstimator) Estimate() (*graph.DAG, error) {
	skeleton := m.LearnSkeleton()

	allowed := make(map[string][]string, len(m.variables))
	for _, variable := range m.variables {
		allowed[variable] = skeleton.Neighbors(variable)
	}

	score, err := m.searchScore()
	if err != nil {
		return nil, err
	}

	search := newHillClimbFrom(m.columns, score)
	search.SetAllowedParents(allowed)
	search.SetMaxIndegree(m.maxIndegree)

	return search.Estimate()
}

// searchScore returns the configured score, or the BDeu default.
func (m *MMHCEstimator) searchScore() (ScoringMethod, error) {
	if m.score != nil {
		return m.score, nil
	}
	return newBDeu(m.columns, defaultEquivalentSampleSize)
}

// LearnSkeleton runs the MMPC phase on its own and returns the undirected
// skeleton. It is exported because the skeleton is useful in its own right: it is
// the set of edges any orientation may use, and inspecting it explains what the
// search phase was allowed to consider.
func (m *MMHCEstimator) LearnSkeleton() *graph.UndirectedGraph {
	neighbours := make(map[string][]string, len(m.variables))
	for _, target := range m.variables {
		neighbours[target] = m.parentsAndChildrenOf(target)
	}

	skeleton := graph.NewUndirectedGraph()
	for _, variable := range m.variables {
		skeleton.AddNode(variable)
	}

	// Symmetry correction: an edge survives only when both endpoints name the
	// other, which discards the false positives of a single local search.
	for _, target := range m.variables {
		for _, candidate := range neighbours[target] {
			if containsString(neighbours[candidate], target) {
				skeleton.AddEdge(target, candidate)
			}
		}
	}

	return skeleton
}

// parentsAndChildrenOf runs MMPC for one target: grow a candidate neighbourhood,
// then prune it.
//
// Each candidate carries a running maximum of its p-value over the conditioning
// subsets tested so far. Because the maximum over the subsets of a larger
// neighbourhood is the larger of that running maximum and the maximum over the
// subsets that contain the newly admitted variable, growing the neighbourhood only
// requires testing the latter. Without that bookkeeping every round would retest
// every subset of the whole neighbourhood from scratch.
func (m *MMHCEstimator) parentsAndChildrenOf(target string) []string {
	excluded := map[string]bool{target: true}
	largestPValue := make(map[string]float64, len(m.variables))

	// Round one conditions on nothing, which is the only subset of an empty
	// neighbourhood.
	for _, candidate := range m.variables {
		if candidate == target {
			continue
		}
		pValue := m.pValue(candidate, target, nil)
		if pValue > m.alpha {
			excluded[candidate] = true
			continue
		}
		largestPValue[candidate] = pValue
	}

	neighbourhood := make([]string, 0, len(m.variables))
	for {
		admitted, found := strongestCandidate(m.variables, largestPValue, excluded)
		if !found {
			break
		}

		neighbourhood = withParent(neighbourhood, admitted)
		delete(largestPValue, admitted)
		m.refreshLargestPValues(target, neighbourhood, admitted, largestPValue, excluded)
	}

	return m.pruneNeighbourhood(target, neighbourhood)
}

// strongestCandidate is the max-min heuristic of the forward phase. Of the
// variables still in play it returns the one whose association with the target best
// survives conditioning, which is the one whose largest p-value is smallest. The
// variables are visited in sorted order and ties keep the first, so the choice does
// not depend on map iteration order.
func strongestCandidate(variables []string, largestPValue map[string]float64,
	excluded map[string]bool) (string, bool) {

	best := ""
	bestPValue := 0.0
	found := false

	for _, candidate := range variables {
		if excluded[candidate] {
			continue
		}
		pValue, inPlay := largestPValue[candidate]
		if !inPlay {
			continue
		}
		if !found || pValue < bestPValue {
			best, bestPValue, found = candidate, pValue, true
		}
	}

	return best, found
}

// refreshLargestPValues updates every candidate against the subsets that contain
// the newly admitted variable, and excludes for good any candidate that turns out
// independent of the target: no larger neighbourhood can bring it back.
func (m *MMHCEstimator) refreshLargestPValues(target string, neighbourhood []string,
	admitted string, largestPValue map[string]float64, excluded map[string]bool) {

	for _, candidate := range m.variables {
		if excluded[candidate] {
			continue
		}
		if _, inPlay := largestPValue[candidate]; !inPlay {
			continue
		}

		pValue := m.maxPValueOverSubsetsWith(candidate, target, neighbourhood, admitted)
		if pValue > largestPValue[candidate] {
			largestPValue[candidate] = pValue
		}
		if largestPValue[candidate] > m.alpha {
			excluded[candidate] = true
			delete(largestPValue, candidate)
		}
	}
}

// maxPValueOverSubsetsWith returns the largest p-value over the conditioning subsets
// that contain one required variable, stopping at the first subset showing
// independence because that already settles the question.
func (m *MMHCEstimator) maxPValueOverSubsetsWith(x, y string, pool []string, required string) float64 {
	rest := withoutParent(pool, required)

	limit := m.maxConditioningSetSize
	if limit <= 0 || limit > len(pool) {
		limit = len(pool)
	}

	largest := 0.0
	for size := 0; size < limit; size++ {
		for _, subset := range combinations(rest, size) {
			pValue := m.pValue(x, y, withParent(subset, required))
			if pValue > largest {
				largest = pValue
			}
			if largest > m.alpha {
				return largest
			}
		}
	}

	return largest
}

// pruneNeighbourhood is the backward phase: a candidate that is independent of the
// target given some subset of the others is not a neighbour after all.
func (m *MMHCEstimator) pruneNeighbourhood(target string, neighbourhood []string) []string {
	kept := sortedCopy(neighbourhood)

	for _, candidate := range neighbourhood {
		remaining := withoutParent(kept, candidate)
		if m.maxPValueOverSubsets(candidate, target, remaining) > m.alpha {
			kept = remaining
		}
	}

	return kept
}

// maxPValueOverSubsets returns the largest p-value of the independence test over
// every conditioning subset up to the configured size. It stops at the first
// subset showing independence, because one such subset already settles the
// question and the remaining subsets cannot change the answer.
func (m *MMHCEstimator) maxPValueOverSubsets(x, y string, pool []string) float64 {
	limit := m.maxConditioningSetSize
	if limit <= 0 || limit > len(pool) {
		limit = len(pool)
	}

	largest := 0.0
	for size := 0; size <= limit; size++ {
		for _, subset := range combinations(pool, size) {
			pValue := m.pValue(x, y, subset)
			if pValue > largest {
				largest = pValue
			}
			if largest > m.alpha {
				return largest
			}
		}
	}

	return largest
}

// pValue runs one independence test, reusing a cached result when the same test
// has been run before. Caching matters because MMPC tests every pair from both
// ends, so without it every test would be run twice over.
func (m *MMHCEstimator) pValue(x, y string, conditioning []string) float64 {
	key := independenceKey(x, y, conditioning)
	if cached, ok := m.pValues[key]; ok {
		return cached
	}

	_, pValue := gSquare(m.columns, x, y, conditioning)
	m.pValues[key] = pValue

	return pValue
}

// independenceKey identifies a test regardless of which way round its two
// variables are given, since conditional independence is symmetric.
func independenceKey(x, y string, conditioning []string) string {
	first, second := x, y
	if second < first {
		first, second = second, first
	}
	return first + "\x00" + second + "\x00" + strings.Join(sortedCopy(conditioning), "\x00")
}

// containsString reports whether a slice holds a value.
func containsString(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}
