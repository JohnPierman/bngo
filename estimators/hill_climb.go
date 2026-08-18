package estimators

import (
	"fmt"
	"math"
	"sort"

	"github.com/JohnPierman/bngo/graph"
)

// defaultEpsilon is the smallest score improvement greedy search treats as real,
// so floating point noise cannot keep it moving.
const defaultEpsilon = 1e-8

// deltaTolerance is how close two score changes must be, relative to their size,
// to count as the same improvement.
//
// Two structures that encode the same independencies score identically in exact
// arithmetic, so the changes leading to them tie. In floating point those ties come
// out a fraction apart, and comparing the numbers directly would let rounding pick
// the winner. Treating them as equal hands the decision to the tie-break below, so
// the result follows the data rather than the last bit of a sum.
const deltaTolerance = 1e-9

// operationKind is one of the three edge changes greedy search considers.
type operationKind int

const (
	addEdge operationKind = iota
	removeEdge
	reverseEdge
)

// operation is a candidate edge change together with the score change it causes.
type operation struct {
	kind  operationKind
	from  string
	to    string
	delta float64
}

// HillClimbSearch learns a structure by greedy search: at each step it prices
// every legal edge addition, deletion and reversal, applies whichever single
// change improves the score most, and stops when none does.
//
// Because a decomposable score prices one change by rescoring only the one or two
// families it touches, and because every accepted change strictly increases the
// score of a finite space of networks, the search always terminates.
//
// On a network of n variables an unrestricted step considers O(n^2) changes. Pass
// SetAllowedParents to restrict each variable to a small candidate set and a step
// costs only the size of those sets, which is what makes greedy search practical
// on a large network. MMHCEstimator derives such a restriction from the data.
//
// Two limits are worth knowing. A score can only tell apart structures that encode
// different independencies, so the direction of an edge is recovered only as far as
// the data determines it. And greedy search returns the first network that no single
// change improves, which need not be the best one: reaching a collision of two
// parents on one child can require reversing two edges at once, and no single
// reversal of the first one improves the score. The undirected edges are the more
// dependable part of the answer, which is what MMHCEstimator.LearnSkeleton reports.
type HillClimbSearch struct {
	variables      []string
	score          ScoringMethod
	maxIndegree    int
	maxIterations  int
	epsilon        float64
	allowedParents map[string]map[string]bool
	candidateList  map[string][]string
}

// NewHillClimb creates a greedy search over the variables of the data, scored by
// BIC. A non-empty entry in cardinality overrides the number of states read off
// the data; pass nil to take every cardinality from the data.
func NewHillClimb(data []map[string]int, cardinality map[string]int) *HillClimbSearch {
	columns := newColumnData(data, cardinality)
	return newHillClimbFrom(columns, newBIC(columns))
}

// newHillClimbFrom builds a search over an existing columnar view, which is how MMHC
// reuses the one it already transposed.
func newHillClimbFrom(columns *columnData, score ScoringMethod) *HillClimbSearch {
	variables := make([]string, len(columns.variables))
	copy(variables, columns.variables)

	return &HillClimbSearch{
		variables: variables,
		score:     score,
		epsilon:   defaultEpsilon,
	}
}

// SetScore replaces the score driving the search.
func (h *HillClimbSearch) SetScore(score ScoringMethod) error {
	if score == nil {
		return fmt.Errorf("hill climb: no scoring method given")
	}
	h.score = score
	return nil
}

// SetMaxIndegree caps how many parents a variable may take. A value of zero or
// less leaves it uncapped. Capping it bounds both the size of the CPDs and the
// cost of scoring them.
func (h *HillClimbSearch) SetMaxIndegree(parents int) {
	h.maxIndegree = parents
}

// SetMaxIterations caps how many changes the search may apply. A value of zero or
// less leaves it uncapped, which is safe because the score strictly increases at
// every step.
func (h *HillClimbSearch) SetMaxIterations(iterations int) {
	h.maxIterations = iterations
}

// SetEpsilon sets the smallest score improvement worth applying.
func (h *HillClimbSearch) SetEpsilon(epsilon float64) {
	h.epsilon = epsilon
}

// SetAllowedParents restricts each variable to the given candidate parents, which
// is how a skeleton learned from the data, or knowledge of the domain, narrows the
// search. Passing nil removes the restriction. The argument is not modified.
func (h *HillClimbSearch) SetAllowedParents(allowed map[string][]string) {
	if allowed == nil {
		h.allowedParents = nil
		h.candidateList = nil
		return
	}

	h.allowedParents = make(map[string]map[string]bool, len(allowed))
	h.candidateList = make(map[string][]string, len(allowed))

	for child, parents := range allowed {
		set := make(map[string]bool, len(parents))
		for _, parent := range parents {
			set[parent] = true
		}
		h.allowedParents[child] = set
		h.candidateList[child] = sortedCopy(parents)
	}
}

// Estimate runs the search and returns the highest scoring network it reaches.
func (h *HillClimbSearch) Estimate() (*graph.DAG, error) {
	dag := graph.NewDAG()
	for _, variable := range h.variables {
		dag.AddNode(variable)
	}

	for iteration := 0; h.maxIterations <= 0 || iteration < h.maxIterations; iteration++ {
		best, found, err := h.bestOperation(dag)
		if err != nil {
			return nil, err
		}
		if !found || best.delta <= h.epsilon {
			break
		}
		if err := applyOperation(dag, best); err != nil {
			return nil, err
		}
	}

	return dag, nil
}

// bestOperation prices every legal change and returns the best one. Ties break on
// a fixed order, so the same data always yields the same network.
func (h *HillClimbSearch) bestOperation(dag *graph.DAG) (operation, bool, error) {
	best := operation{}
	found := false

	for _, to := range h.variables {
		for _, from := range h.candidateParents(to) {
			if from == to {
				continue
			}

			candidates, err := h.operationsFor(dag, from, to)
			if err != nil {
				return operation{}, false, err
			}

			for _, candidate := range candidates {
				if !found || beats(candidate, best) {
					best, found = candidate, true
				}
			}
		}
	}

	return best, found, nil
}

// beats reports whether one operation should be preferred over another, ordering
// equal deltas so the result does not depend on map iteration order.
func beats(candidate, best operation) bool {
	if !sameDelta(candidate.delta, best.delta) {
		return candidate.delta > best.delta
	}
	if candidate.kind != best.kind {
		return candidate.kind < best.kind
	}
	if candidate.from != best.from {
		return candidate.from < best.from
	}
	return candidate.to < best.to
}

// sameDelta reports whether two score changes are equal to within rounding.
func sameDelta(a, b float64) bool {
	scale := math.Max(math.Abs(a), math.Abs(b))
	return math.Abs(a-b) <= deltaTolerance*math.Max(1, scale)
}

// operationsFor returns the legal changes involving one ordered pair. A pair
// joined the other way round is handled when that pair is visited, so no change is
// priced twice.
func (h *HillClimbSearch) operationsFor(dag *graph.DAG, from, to string) ([]operation, error) {
	if dag.HasEdge(from, to) {
		return h.operationsOnEdge(dag, from, to)
	}
	if dag.HasEdge(to, from) {
		return nil, nil
	}
	return h.additionOf(dag, from, to)
}

// additionOf prices adding an edge, when adding it is legal at all.
func (h *HillClimbSearch) additionOf(dag *graph.DAG, from, to string) ([]operation, error) {
	if h.exceedsIndegree(dag, to) {
		return nil, nil
	}
	// The edge closes a cycle exactly when the child already reaches the parent.
	if hasDirectedPath(dag, to, from, noIgnoredEdge) {
		return nil, nil
	}

	delta, err := h.deltaOfAddingParent(to, dag.Parents(to), from)
	if err != nil {
		return nil, err
	}

	return []operation{{kind: addEdge, from: from, to: to, delta: delta}}, nil
}

// operationsOnEdge prices deleting and reversing an existing edge.
func (h *HillClimbSearch) operationsOnEdge(dag *graph.DAG, from, to string) ([]operation, error) {
	removal, err := h.deltaOfRemovingParent(to, dag.Parents(to), from)
	if err != nil {
		return nil, err
	}

	operations := []operation{{kind: removeEdge, from: from, to: to, delta: removal}}

	if !h.isAllowedParent(from, to) || h.exceedsIndegree(dag, from) {
		return operations, nil
	}
	// Reversing closes a cycle when the parent reaches the child by another route.
	if hasDirectedPath(dag, from, to, [2]string{from, to}) {
		return operations, nil
	}

	addition, err := h.deltaOfAddingParent(from, dag.Parents(from), to)
	if err != nil {
		return nil, err
	}

	return append(operations, operation{
		kind: reverseEdge, from: from, to: to, delta: removal + addition,
	}), nil
}

// deltaOfAddingParent returns the score change from giving a variable one more
// parent.
func (h *HillClimbSearch) deltaOfAddingParent(variable string, parents []string, parent string) (float64, error) {
	return h.scoreDelta(variable, parents, withParent(parents, parent))
}

// deltaOfRemovingParent returns the score change from taking one parent away.
func (h *HillClimbSearch) deltaOfRemovingParent(variable string, parents []string, parent string) (float64, error) {
	return h.scoreDelta(variable, parents, withoutParent(parents, parent))
}

// scoreDelta returns how much the local score of a variable changes between two
// parent sets.
func (h *HillClimbSearch) scoreDelta(variable string, before, after []string) (float64, error) {
	scoreBefore, err := h.score.LocalScore(variable, before)
	if err != nil {
		return 0, err
	}

	scoreAfter, err := h.score.LocalScore(variable, after)
	if err != nil {
		return 0, err
	}

	return scoreAfter - scoreBefore, nil
}

// candidateParents returns the variables allowed to parent a child, in a fixed
// order.
func (h *HillClimbSearch) candidateParents(child string) []string {
	if h.candidateList == nil {
		return h.variables
	}
	return h.candidateList[child]
}

// isAllowedParent reports whether one variable may parent another.
func (h *HillClimbSearch) isAllowedParent(child, parent string) bool {
	if h.allowedParents == nil {
		return true
	}
	return h.allowedParents[child][parent]
}

// exceedsIndegree reports whether a variable already has as many parents as it is
// allowed.
func (h *HillClimbSearch) exceedsIndegree(dag *graph.DAG, child string) bool {
	return h.maxIndegree > 0 && len(dag.Parents(child)) >= h.maxIndegree
}

// applyOperation carries out one edge change.
func applyOperation(dag *graph.DAG, op operation) error {
	switch op.kind {
	case addEdge:
		return dag.AddEdge(op.from, op.to)
	case removeEdge:
		dag.RemoveEdge(op.from, op.to)
		return nil
	case reverseEdge:
		dag.RemoveEdge(op.from, op.to)
		if err := dag.AddEdge(op.to, op.from); err != nil {
			return fmt.Errorf("reversing %s -> %s: %w", op.from, op.to, err)
		}
		return nil
	default:
		return fmt.Errorf("unknown operation kind %d", op.kind)
	}
}

// noIgnoredEdge is the empty edge, for path queries that ignore nothing.
var noIgnoredEdge = [2]string{}

// hasDirectedPath reports whether the graph holds a directed path from start to
// target while ignoring one edge. It never modifies the graph, so a candidate
// reversal can be tested without disturbing the search state.
func hasDirectedPath(dag *graph.DAG, start, target string, ignore [2]string) bool {
	if start == target {
		return true
	}

	visited := map[string]bool{start: true}
	stack := []string{start}

	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		for _, child := range dag.Children(node) {
			if node == ignore[0] && child == ignore[1] {
				continue
			}
			if child == target {
				return true
			}
			if !visited[child] {
				visited[child] = true
				stack = append(stack, child)
			}
		}
	}

	return false
}

// withParent returns the parent set with one variable added, leaving the argument
// untouched.
func withParent(parents []string, parent string) []string {
	extended := make([]string, 0, len(parents)+1)
	extended = append(extended, parents...)
	extended = append(extended, parent)
	sort.Strings(extended)
	return extended
}

// withoutParent returns the parent set with one variable removed, leaving the
// argument untouched.
func withoutParent(parents []string, parent string) []string {
	reduced := make([]string, 0, len(parents))
	for _, candidate := range parents {
		if candidate != parent {
			reduced = append(reduced, candidate)
		}
	}
	return reduced
}
