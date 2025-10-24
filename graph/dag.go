// Package graph provides graph data structures for Bayesian Networks
package graph

import (
	"fmt"
	"sort"
)

// DAG represents a Directed Acyclic Graph
type DAG struct {
	nodes   map[string]bool
	edges   map[string]map[string]bool // parent -> children
	parents map[string]map[string]bool // child -> parents
}

// NewDAG creates a new empty DAG
func NewDAG() *DAG {
	return &DAG{
		nodes:   make(map[string]bool),
		edges:   make(map[string]map[string]bool),
		parents: make(map[string]map[string]bool),
	}
}

// NewDAGFromEdges creates a DAG from a list of edges
func NewDAGFromEdges(edges [][2]string) (*DAG, error) {
	dag := NewDAG()
	for _, edge := range edges {
		if err := dag.AddEdge(edge[0], edge[1]); err != nil {
			return nil, err
		}
	}
	return dag, nil
}

// AddNode adds a node to the DAG
func (d *DAG) AddNode(node string) {
	if !d.nodes[node] {
		d.nodes[node] = true
		d.edges[node] = make(map[string]bool)
		d.parents[node] = make(map[string]bool)
	}
}

// AddEdge adds a directed edge from parent to child
func (d *DAG) AddEdge(parent, child string) error {
	d.AddNode(parent)
	d.AddNode(child)

	// Check if adding this edge would create a cycle
	if d.wouldCreateCycle(parent, child) {
		return fmt.Errorf("adding edge %s -> %s would create a cycle", parent, child)
	}

	d.edges[parent][child] = true
	d.parents[child][parent] = true
	return nil
}

// RemoveEdge removes a directed edge
func (d *DAG) RemoveEdge(parent, child string) {
	if d.edges[parent] != nil {
		delete(d.edges[parent], child)
	}
	if d.parents[child] != nil {
		delete(d.parents[child], parent)
	}
}

// HasEdge checks if an edge exists
func (d *DAG) HasEdge(parent, child string) bool {
	if d.edges[parent] == nil {
		return false
	}
	return d.edges[parent][child]
}

// Nodes returns all nodes in the DAG
func (d *DAG) Nodes() []string {
	nodes := make([]string, 0, len(d.nodes))
	for node := range d.nodes {
		nodes = append(nodes, node)
	}
	sort.Strings(nodes)
	return nodes
}

// Edges returns all edges in the DAG as [parent, child] pairs
func (d *DAG) Edges() [][2]string {
	edges := make([][2]string, 0)
	for parent, children := range d.edges {
		for child := range children {
			edges = append(edges, [2]string{parent, child})
		}
	}
	return edges
}

// Children returns all children of a node
func (d *DAG) Children(node string) []string {
	children := make([]string, 0)
	if d.edges[node] != nil {
		for child := range d.edges[node] {
			children = append(children, child)
		}
	}
	sort.Strings(children)
	return children
}

// Parents returns all parents of a node
func (d *DAG) Parents(node string) []string {
	parents := make([]string, 0)
	if d.parents[node] != nil {
		for parent := range d.parents[node] {
			parents = append(parents, parent)
		}
	}
	sort.Strings(parents)
	return parents
}

// Ancestors returns all ancestors of a node
func (d *DAG) Ancestors(node string) []string {
	visited := make(map[string]bool)
	d.ancestorsHelper(node, visited)
	delete(visited, node)

	ancestors := make([]string, 0, len(visited))
	for ancestor := range visited {
		ancestors = append(ancestors, ancestor)
	}
	sort.Strings(ancestors)
	return ancestors
}

func (d *DAG) ancestorsHelper(node string, visited map[string]bool) {
	if visited[node] {
		return
	}
	visited[node] = true

	for parent := range d.parents[node] {
		d.ancestorsHelper(parent, visited)
	}
}

// Descendants returns all descendants of a node
func (d *DAG) Descendants(node string) []string {
	visited := make(map[string]bool)
	d.descendantsHelper(node, visited)
	delete(visited, node)

	descendants := make([]string, 0, len(visited))
	for descendant := range visited {
		descendants = append(descendants, descendant)
	}
	sort.Strings(descendants)
	return descendants
}

func (d *DAG) descendantsHelper(node string, visited map[string]bool) {
	if visited[node] {
		return
	}
	visited[node] = true

	for child := range d.edges[node] {
		d.descendantsHelper(child, visited)
	}
}

// wouldCreateCycle checks if adding an edge would create a cycle
func (d *DAG) wouldCreateCycle(parent, child string) bool {
	// If child is an ancestor of parent, adding this edge would create a cycle
	ancestors := d.Ancestors(parent)
	for _, ancestor := range ancestors {
		if ancestor == child {
			return true
		}
	}
	return false
}

// TopologicalSort returns nodes in topological order
func (d *DAG) TopologicalSort() ([]string, error) {
	inDegree := make(map[string]int)
	for node := range d.nodes {
		inDegree[node] = len(d.parents[node])
	}

	queue := make([]string, 0)
	for node, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, node)
		}
	}

	result := make([]string, 0, len(d.nodes))
	for len(queue) > 0 {
		// Sort queue for deterministic ordering
		sort.Strings(queue)
		node := queue[0]
		queue = queue[1:]
		result = append(result, node)

		for child := range d.edges[node] {
			inDegree[child]--
			if inDegree[child] == 0 {
				queue = append(queue, child)
			}
		}
	}

	if len(result) != len(d.nodes) {
		return nil, fmt.Errorf("cycle detected in graph")
	}

	return result, nil
}

// Copy creates a deep copy of the DAG
func (d *DAG) Copy() *DAG {
	newDAG := NewDAG()
	for node := range d.nodes {
		newDAG.AddNode(node)
	}
	for parent, children := range d.edges {
		for child := range children {
			_ = newDAG.AddEdge(parent, child) // Ignore error as structure is valid by construction
		}
	}
	return newDAG
}

// MoralGraph creates a moral graph (undirected) from the DAG
func (d *DAG) MoralGraph() *UndirectedGraph {
	ug := NewUndirectedGraph()

	// Add all nodes
	for node := range d.nodes {
		ug.AddNode(node)
	}

	// Add all edges as undirected
	for parent, children := range d.edges {
		for child := range children {
			ug.AddEdge(parent, child)
		}
	}

	// Marry parents (add edges between parents of each node)
	for child := range d.nodes {
		parents := d.Parents(child)
		for i := 0; i < len(parents); i++ {
			for j := i + 1; j < len(parents); j++ {
				ug.AddEdge(parents[i], parents[j])
			}
		}
	}

	return ug
}
