package estimators

import (
	"math/rand"
	"sort"
	"strings"

	"github.com/JohnPierman/bngo/graph"
)

// flip returns value with its bit flipped with the given probability, which is how
// the generators below turn a deterministic relationship into a noisy one.
func flip(value int, probability float64, r *rand.Rand) int {
	if r.Float64() < probability {
		return 1 - value
	}
	return value
}

// chainData samples from A -> B -> C over binary variables. A and C are dependent,
// but independent once B is known, so the true skeleton is A-B and B-C.
func chainData(rows int, seed int64) []map[string]int {
	r := rand.New(rand.NewSource(seed))
	data := make([]map[string]int, rows)

	for i := 0; i < rows; i++ {
		a := r.Intn(2)
		b := flip(a, 0.10, r)
		c := flip(b, 0.10, r)
		data[i] = map[string]int{"A": a, "B": b, "C": c}
	}

	return data
}

// colliderData samples from A -> C <- B over binary variables, where C is a noisy
// OR of its parents. A and B are independent of each other, and both are dependent
// on C, so the true skeleton is A-C and B-C with no edge between A and B.
func colliderData(rows int, seed int64) []map[string]int {
	r := rand.New(rand.NewSource(seed))
	data := make([]map[string]int, rows)

	for i := 0; i < rows; i++ {
		a := r.Intn(2)
		b := r.Intn(2)
		c := 0
		if a == 1 || b == 1 {
			c = 1
		}
		data[i] = map[string]int{"A": a, "B": b, "C": flip(c, 0.05, r)}
	}

	return data
}

// independentData samples three binary variables that have nothing to do with each
// other, so the true structure has no edges at all.
func independentData(rows int, seed int64) []map[string]int {
	r := rand.New(rand.NewSource(seed))
	data := make([]map[string]int, rows)

	for i := 0; i < rows; i++ {
		data[i] = map[string]int{"A": r.Intn(2), "B": r.Intn(2), "C": r.Intn(2)}
	}

	return data
}

// ladderEdges returns the edges of a sparse network on the given number of binary
// variables, where each variable depends on the two before it. It is a stand in for
// a large sparse network: the number of edges grows linearly in the number of
// variables while the number of candidate edges grows quadratically.
func ladderEdges(variables int) [][2]string {
	edges := make([][2]string, 0, 2*variables)

	for i := 1; i < variables; i++ {
		edges = append(edges, [2]string{nodeName(i - 1), nodeName(i)})
		if i >= 2 {
			edges = append(edges, [2]string{nodeName(i - 2), nodeName(i)})
		}
	}

	return edges
}

// ladderData samples from the network ladderEdges describes, giving each variable a
// noisy majority of its parents.
func ladderData(variables, rows int, seed int64) []map[string]int {
	r := rand.New(rand.NewSource(seed))
	data := make([]map[string]int, rows)

	for i := 0; i < rows; i++ {
		row := make(map[string]int, variables)
		row[nodeName(0)] = r.Intn(2)

		for j := 1; j < variables; j++ {
			value := row[nodeName(j-1)]
			if j >= 2 && row[nodeName(j-2)] == value {
				// Both parents agree, so the child follows them closely.
				row[nodeName(j)] = flip(value, 0.05, r)
				continue
			}
			row[nodeName(j)] = flip(value, 0.20, r)
		}

		data[i] = row
	}

	return data
}

// nodeName returns a fixed width name, so sorting names sorts them by index.
func nodeName(index int) string {
	return "V" + pad(index)
}

// pad formats an index to three digits.
func pad(index int) string {
	digits := []byte{
		byte('0' + (index/100)%10),
		byte('0' + (index/10)%10),
		byte('0' + index%10),
	}
	return string(digits)
}

// skeletonOf returns the undirected edges of a DAG as a set of canonical keys, for
// comparing a learned structure against the truth without insisting on a direction
// the data cannot identify.
func skeletonOf(dag *graph.DAG) map[string]bool {
	edges := make(map[string]bool)
	for _, edge := range dag.Edges() {
		edges[undirectedKey(edge[0], edge[1])] = true
	}
	return edges
}

// skeletonOfEdges returns the undirected edges of an edge list.
func skeletonOfEdges(edges [][2]string) map[string]bool {
	set := make(map[string]bool, len(edges))
	for _, edge := range edges {
		set[undirectedKey(edge[0], edge[1])] = true
	}
	return set
}

// undirectedEdgesOf returns the edges of an undirected graph as canonical keys.
func undirectedEdgesOf(g *graph.UndirectedGraph) map[string]bool {
	edges := make(map[string]bool)
	for _, edge := range g.Edges() {
		edges[undirectedKey(edge[0], edge[1])] = true
	}
	return edges
}

// undirectedKey names an edge the same way whichever end comes first.
func undirectedKey(a, b string) string {
	if b < a {
		a, b = b, a
	}
	return a + "-" + b
}

// sortedKeys returns the members of a set in sorted order, for readable failures.
func sortedKeys(set map[string]bool) string {
	keys := make([]string, 0, len(set))
	for key := range set {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return strings.Join(keys, " ")
}

// directedEdgesOf returns the directed edges of a DAG as canonical keys.
func directedEdgesOf(dag *graph.DAG) map[string]bool {
	edges := make(map[string]bool)
	for _, edge := range dag.Edges() {
		edges[edge[0]+"->"+edge[1]] = true
	}
	return edges
}
