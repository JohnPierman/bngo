package graph

import "sort"

// UndirectedGraph represents an undirected graph
type UndirectedGraph struct {
	nodes map[string]bool
	edges map[string]map[string]bool
}

// NewUndirectedGraph creates a new empty undirected graph
func NewUndirectedGraph() *UndirectedGraph {
	return &UndirectedGraph{
		nodes: make(map[string]bool),
		edges: make(map[string]map[string]bool),
	}
}

// AddNode adds a node to the graph
func (g *UndirectedGraph) AddNode(node string) {
	if !g.nodes[node] {
		g.nodes[node] = true
		g.edges[node] = make(map[string]bool)
	}
}

// AddEdge adds an undirected edge between two nodes
func (g *UndirectedGraph) AddEdge(node1, node2 string) {
	g.AddNode(node1)
	g.AddNode(node2)
	g.edges[node1][node2] = true
	g.edges[node2][node1] = true
}

// RemoveEdge removes an undirected edge
func (g *UndirectedGraph) RemoveEdge(node1, node2 string) {
	if g.edges[node1] != nil {
		delete(g.edges[node1], node2)
	}
	if g.edges[node2] != nil {
		delete(g.edges[node2], node1)
	}
}

// HasEdge checks if an edge exists
func (g *UndirectedGraph) HasEdge(node1, node2 string) bool {
	if g.edges[node1] == nil {
		return false
	}
	return g.edges[node1][node2]
}

// Nodes returns all nodes in the graph
func (g *UndirectedGraph) Nodes() []string {
	nodes := make([]string, 0, len(g.nodes))
	for node := range g.nodes {
		nodes = append(nodes, node)
	}
	sort.Strings(nodes)
	return nodes
}

// Neighbors returns all neighbors of a node
func (g *UndirectedGraph) Neighbors(node string) []string {
	neighbors := make([]string, 0)
	if g.edges[node] != nil {
		for neighbor := range g.edges[node] {
			neighbors = append(neighbors, neighbor)
		}
	}
	sort.Strings(neighbors)
	return neighbors
}

// Edges returns all edges in the graph
func (g *UndirectedGraph) Edges() [][2]string {
	edges := make([][2]string, 0)
	visited := make(map[string]map[string]bool)

	for node1, neighbors := range g.edges {
		for node2 := range neighbors {
			if visited[node2] == nil || !visited[node2][node1] {
				edges = append(edges, [2]string{node1, node2})
				if visited[node1] == nil {
					visited[node1] = make(map[string]bool)
				}
				visited[node1][node2] = true
			}
		}
	}

	return edges
}

// Copy creates a deep copy of the graph
func (g *UndirectedGraph) Copy() *UndirectedGraph {
	newGraph := NewUndirectedGraph()
	for node := range g.nodes {
		newGraph.AddNode(node)
	}
	for node, neighbors := range g.edges {
		for neighbor := range neighbors {
			if node < neighbor { // Add each edge only once
				newGraph.AddEdge(node, neighbor)
			}
		}
	}
	return newGraph
}
