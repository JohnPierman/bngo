package graph

import (
	"testing"
)

func TestDAGCreation(t *testing.T) {
	dag := NewDAG()
	dag.AddNode("A")
	dag.AddNode("B")
	dag.AddNode("C")

	if len(dag.Nodes()) != 3 {
		t.Errorf("Expected 3 nodes, got %d", len(dag.Nodes()))
	}
}

func TestDAGEdges(t *testing.T) {
	dag := NewDAG()
	err := dag.AddEdge("A", "B")
	if err != nil {
		t.Errorf("Failed to add edge: %v", err)
	}

	if !dag.HasEdge("A", "B") {
		t.Error("Edge A->B should exist")
	}

	if dag.HasEdge("B", "A") {
		t.Error("Edge B->A should not exist")
	}
}

func TestDAGCycleDetection(t *testing.T) {
	dag := NewDAG()
	dag.AddEdge("A", "B")
	dag.AddEdge("B", "C")

	// Try to create a cycle
	err := dag.AddEdge("C", "A")
	if err == nil {
		t.Error("Should have detected cycle")
	}
}

func TestDAGParentsChildren(t *testing.T) {
	dag := NewDAG()
	dag.AddEdge("A", "C")
	dag.AddEdge("B", "C")

	parents := dag.Parents("C")
	if len(parents) != 2 {
		t.Errorf("Expected 2 parents, got %d", len(parents))
	}

	children := dag.Children("A")
	if len(children) != 1 || children[0] != "C" {
		t.Errorf("Expected child C, got %v", children)
	}
}

func TestDAGTopologicalSort(t *testing.T) {
	dag := NewDAG()
	dag.AddEdge("A", "C")
	dag.AddEdge("B", "C")
	dag.AddEdge("C", "D")

	order, err := dag.TopologicalSort()
	if err != nil {
		t.Errorf("Topological sort failed: %v", err)
	}

	// Find positions
	pos := make(map[string]int)
	for i, node := range order {
		pos[node] = i
	}

	// Check that parents come before children
	if pos["A"] >= pos["C"] {
		t.Error("A should come before C")
	}
	if pos["B"] >= pos["C"] {
		t.Error("B should come before C")
	}
	if pos["C"] >= pos["D"] {
		t.Error("C should come before D")
	}
}

func TestDAGAncestorsDescendants(t *testing.T) {
	dag := NewDAG()
	dag.AddEdge("A", "B")
	dag.AddEdge("B", "C")
	dag.AddEdge("C", "D")

	ancestors := dag.Ancestors("D")
	if len(ancestors) != 3 {
		t.Errorf("Expected 3 ancestors, got %d", len(ancestors))
	}

	descendants := dag.Descendants("A")
	if len(descendants) != 3 {
		t.Errorf("Expected 3 descendants, got %d", len(descendants))
	}
}
