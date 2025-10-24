package estimators

import (
	"sort"

	"github.com/JohnPierman/bngo/graph"
)

// PCEstimator implements the PC (Peter-Clark) algorithm for structure learning
type PCEstimator struct {
	Data        []map[string]int
	Variables   []string
	Cardinality map[string]int
	Alpha       float64 // Significance level for independence tests
}

// NewPC creates a new PC estimator
func NewPC(data []map[string]int) *PCEstimator {
	// Extract variables and cardinality from data
	varSet := make(map[string]bool)
	cardinality := make(map[string]int)

	for _, sample := range data {
		for varName, value := range sample {
			varSet[varName] = true
			if value+1 > cardinality[varName] {
				cardinality[varName] = value + 1
			}
		}
	}

	variables := make([]string, 0, len(varSet))
	for v := range varSet {
		variables = append(variables, v)
	}
	sort.Strings(variables)

	return &PCEstimator{
		Data:        data,
		Variables:   variables,
		Cardinality: cardinality,
		Alpha:       0.05,
	}
}

// SetAlpha sets the significance level for independence tests
func (pc *PCEstimator) SetAlpha(alpha float64) {
	pc.Alpha = alpha
}

// Estimate learns the graph structure using the PC algorithm
func (pc *PCEstimator) Estimate() (*graph.DAG, error) {
	// Start with complete undirected graph
	ug := graph.NewUndirectedGraph()
	for _, v := range pc.Variables {
		ug.AddNode(v)
	}

	// Add all edges
	for i := 0; i < len(pc.Variables); i++ {
		for j := i + 1; j < len(pc.Variables); j++ {
			ug.AddEdge(pc.Variables[i], pc.Variables[j])
		}
	}

	// Separation sets for later orientation
	sepSets := make(map[string]map[string][]string)
	for _, v := range pc.Variables {
		sepSets[v] = make(map[string][]string)
	}

	// Phase 1: Edge removal
	maxCondSetSize := len(pc.Variables) - 2
	for condSetSize := 0; condSetSize <= maxCondSetSize; condSetSize++ {
		changed := false

		for _, x := range pc.Variables {
			neighbors := ug.Neighbors(x)

			for _, y := range neighbors {
				// Get potential conditioning sets (neighbors of X excluding Y)
				potentialCond := make([]string, 0)
				for _, n := range neighbors {
					if n != y {
						potentialCond = append(potentialCond, n)
					}
				}

				// Test all subsets of size condSetSize
				condSets := combinations(potentialCond, condSetSize)

				for _, condSet := range condSets {
					// Test conditional independence
					_, pValue := ChiSquareTest(pc.Data, x, y, condSet, pc.Cardinality)

					if pValue > pc.Alpha {
						// X and Y are conditionally independent given condSet
						ug.RemoveEdge(x, y)
						sepSets[x][y] = condSet
						sepSets[y][x] = condSet
						changed = true
						break
					}
				}

				if changed {
					break
				}
			}
		}

		if !changed && condSetSize > 0 {
			break
		}
	}

	// Phase 2: Orient edges using v-structures and Meek rules
	dag := pc.orientEdges(ug, sepSets)

	return dag, nil
}

// orientEdges converts undirected graph to PDAG/DAG using v-structures and Meek's rules
func (pc *PCEstimator) orientEdges(ug *graph.UndirectedGraph, sepSets map[string]map[string][]string) *graph.DAG {
	dag := graph.NewDAG()

	// Add all nodes
	for _, node := range ug.Nodes() {
		dag.AddNode(node)
	}

	// Track oriented and unoriented edges
	oriented := make(map[string]map[string]bool)
	unoriented := make(map[string]map[string]bool)
	for _, v := range pc.Variables {
		oriented[v] = make(map[string]bool)
		unoriented[v] = make(map[string]bool)
	}

	// Initialize all edges as unoriented
	for _, edge := range ug.Edges() {
		node1, node2 := edge[0], edge[1]
		unoriented[node1][node2] = true
		unoriented[node2][node1] = true
	}

	// Rule 0: Find v-structures: X -> Z <- Y where X and Y are not adjacent
	for _, z := range pc.Variables {
		neighbors := ug.Neighbors(z)

		for i := 0; i < len(neighbors); i++ {
			for j := i + 1; j < len(neighbors); j++ {
				x := neighbors[i]
				y := neighbors[j]

				// Check if X and Y are not adjacent
				if !ug.HasEdge(x, y) {
					// Check if Z is not in the separating set of X and Y
					sepSet := sepSets[x][y]
					zInSepSet := false
					for _, v := range sepSet {
						if v == z {
							zInSepSet = true
							break
						}
					}

					if !zInSepSet {
						// Orient X -> Z <- Y
						pc.orientEdge(x, z, oriented, unoriented)
						pc.orientEdge(y, z, oriented, unoriented)
					}
				}
			}
		}
	}

	// Apply Meek's rules iteratively until no more edges can be oriented
	changed := true
	for changed {
		changed = false

		// Rule 1: Orient i - j into i -> j whenever there is k -> i such that k and j are not adjacent
		changed = pc.applyMeekRule1(ug, oriented, unoriented) || changed

		// Rule 2: Orient i - j into i -> j whenever there is a chain i -> k -> j
		changed = pc.applyMeekRule2(oriented, unoriented) || changed

		// Rule 3: Orient i - j into i -> j whenever there are two chains i - k -> j and i - l -> j
		// such that k and l are not adjacent
		changed = pc.applyMeekRule3(ug, oriented, unoriented) || changed

		// Rule 4: Orient i - j into i -> j whenever there are two chains i - k -> l and k -> l -> j
		// such that k and j are not adjacent
		changed = pc.applyMeekRule4(ug, oriented, unoriented) || changed
	}

	// Add oriented edges to DAG
	for parent, children := range oriented {
		for child := range children {
			_ = dag.AddEdge(parent, child) // Ignore error as structure is valid by construction
		}
	}

	// Orient remaining unoriented edges arbitrarily (lexicographic order to ensure consistency)
	for node1, neighbors := range unoriented {
		for node2 := range neighbors {
			if node1 < node2 && unoriented[node2][node1] {
				// This edge is still unoriented
				if !dag.HasEdge(node1, node2) && !dag.HasEdge(node2, node1) {
					_ = dag.AddEdge(node1, node2) // Ignore error as structure is valid by construction
				}
			}
		}
	}

	return dag
}

// orientEdge orients an edge from parent to child
func (pc *PCEstimator) orientEdge(parent, child string, oriented, unoriented map[string]map[string]bool) {
	oriented[parent][child] = true
	delete(unoriented[parent], child)
	delete(unoriented[child], parent)
}

// applyMeekRule1: Orient i - j into i -> j whenever there is k -> i such that k and j are not adjacent
func (pc *PCEstimator) applyMeekRule1(ug *graph.UndirectedGraph, oriented, unoriented map[string]map[string]bool) bool {
	changed := false

	for i, neighbors := range unoriented {
		for j := range neighbors {
			if !unoriented[j][i] {
				continue // Already oriented
			}

			// Look for k -> i where k and j are not adjacent
			for k := range oriented {
				if oriented[k][i] && !ug.HasEdge(k, j) && k != j {
					pc.orientEdge(i, j, oriented, unoriented)
					changed = true
					break
				}
			}
		}
	}

	return changed
}

// applyMeekRule2: Orient i - j into i -> j whenever there is a chain i -> k -> j
func (pc *PCEstimator) applyMeekRule2(oriented, unoriented map[string]map[string]bool) bool {
	changed := false

	for i, neighbors := range unoriented {
		for j := range neighbors {
			if !unoriented[j][i] {
				continue // Already oriented
			}

			// Look for i -> k -> j
			for k := range oriented[i] {
				if oriented[k][j] && k != j {
					pc.orientEdge(i, j, oriented, unoriented)
					changed = true
					break
				}
			}
		}
	}

	return changed
}

// applyMeekRule3: Orient i - j into i -> j whenever there are two chains i - k -> j and i - l -> j
// such that k and l are not adjacent
func (pc *PCEstimator) applyMeekRule3(ug *graph.UndirectedGraph, oriented, unoriented map[string]map[string]bool) bool {
	changed := false

	for i, neighbors := range unoriented {
		for j := range neighbors {
			if !unoriented[j][i] {
				continue // Already oriented
			}

			// Find all k where i - k -> j
			candidates := make([]string, 0)
			for k := range unoriented[i] {
				if oriented[k][j] && k != j {
					candidates = append(candidates, k)
				}
			}

			// Check if any pair of candidates are not adjacent
			for idx1 := 0; idx1 < len(candidates); idx1++ {
				for idx2 := idx1 + 1; idx2 < len(candidates); idx2++ {
					k := candidates[idx1]
					l := candidates[idx2]
					if !ug.HasEdge(k, l) {
						pc.orientEdge(i, j, oriented, unoriented)
						changed = true
						break
					}
				}
				if changed {
					break
				}
			}
		}
	}

	return changed
}

// applyMeekRule4: Orient i - j into i -> j whenever there are two chains i - k -> l and k -> l -> j
// such that k and j are not adjacent
func (pc *PCEstimator) applyMeekRule4(ug *graph.UndirectedGraph, oriented, unoriented map[string]map[string]bool) bool {
	changed := false

	for i, neighbors := range unoriented {
		for j := range neighbors {
			if !unoriented[j][i] {
				continue // Already oriented
			}

			// Look for i - k -> l -> j where k and j are not adjacent
			for k := range unoriented[i] {
				if !ug.HasEdge(k, j) && k != j {
					for l := range oriented[k] {
						if oriented[l][j] && l != j {
							pc.orientEdge(i, j, oriented, unoriented)
							changed = true
							break
						}
					}
					if changed {
						break
					}
				}
			}
		}
	}

	return changed
}

// combinations generates all combinations of size k from elements
func combinations(elements []string, k int) [][]string {
	if k == 0 {
		return [][]string{{}}
	}

	if len(elements) == 0 {
		return [][]string{}
	}

	if len(elements) < k {
		return [][]string{}
	}

	result := make([][]string, 0)

	// Include first element
	withFirst := combinations(elements[1:], k-1)
	for _, combo := range withFirst {
		newCombo := make([]string, 0, k)
		newCombo = append(newCombo, elements[0])
		newCombo = append(newCombo, combo...)
		result = append(result, newCombo)
	}

	// Exclude first element
	withoutFirst := combinations(elements[1:], k)
	result = append(result, withoutFirst...)

	return result
}
