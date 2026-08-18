package estimators

import (
	"testing"
)

func TestHillClimbSearch_RecoversTheSkeletonOfAChain(t *testing.T) {
	data := chainData(4000, 31)

	dag, err := NewHillClimb(data, nil).Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	got := skeletonOf(dag)
	want := skeletonOfEdges([][2]string{{"A", "B"}, {"B", "C"}})

	if sortedKeys(got) != sortedKeys(want) {
		t.Errorf("skeleton = [%s], want [%s]", sortedKeys(got), sortedKeys(want))
	}

	if _, err := dag.TopologicalSort(); err != nil {
		t.Errorf("the learned graph is not acyclic: %v", err)
	}
}

func TestHillClimbSearch_LeavesUnrelatedVariablesUnconnected(t *testing.T) {
	data := independentData(4000, 32)

	dag, err := NewHillClimb(data, nil).Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	if edges := dag.Edges(); len(edges) != 0 {
		t.Errorf("learned %v, want no edges between independent variables", edges)
	}
	if nodes := dag.Nodes(); len(nodes) != 3 {
		t.Errorf("learned %d nodes, want 3", len(nodes))
	}
}

func TestHillClimbSearch_RecoversTheSkeletonOfACollider(t *testing.T) {
	data := colliderData(4000, 33)

	dag, err := NewHillClimb(data, nil).Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	got := skeletonOf(dag)
	want := skeletonOfEdges([][2]string{{"A", "C"}, {"B", "C"}})

	if sortedKeys(got) != sortedKeys(want) {
		t.Errorf("skeleton = [%s], want [%s]", sortedKeys(got), sortedKeys(want))
	}
}

func TestHillClimbSearch_OrientsACollider(t *testing.T) {
	data := colliderData(6000, 34)

	dag, err := NewHillClimb(data, nil).Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	// A collider is the one pattern a score can orient on its own, because no
	// other orientation of A -> C <- B encodes the same independencies.
	edges := directedEdgesOf(dag)
	if !edges["A->C"] || !edges["B->C"] {
		t.Errorf("directed edges = [%s], want A->C and B->C", sortedKeys(edges))
	}
}

func TestHillClimbSearch_RespectsMaxIndegree(t *testing.T) {
	data := colliderData(4000, 35)

	search := NewHillClimb(data, nil)
	search.SetMaxIndegree(1)

	dag, err := search.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	for _, node := range dag.Nodes() {
		if parents := dag.Parents(node); len(parents) > 1 {
			t.Errorf("%s has parents %v, want at most 1", node, parents)
		}
	}
}

func TestHillClimbSearch_RespectsAllowedParents(t *testing.T) {
	data := chainData(4000, 36)

	search := NewHillClimb(data, nil)
	// B may not be a parent of C, which forbids the B-C edge in that direction.
	search.SetAllowedParents(map[string][]string{
		"A": {"B"},
		"B": {"A", "C"},
		"C": {},
	})

	dag, err := search.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	if parents := dag.Parents("C"); len(parents) != 0 {
		t.Errorf("C has parents %v, want none to be allowed", parents)
	}
	if edges := skeletonOf(dag); !edges["A-B"] {
		t.Errorf("skeleton = [%s], want the allowed A-B edge", sortedKeys(edges))
	}
}

func TestHillClimbSearch_AllowedParentsCanForbidEverything(t *testing.T) {
	data := chainData(2000, 37)

	search := NewHillClimb(data, nil)
	search.SetAllowedParents(map[string][]string{"A": {}, "B": {}, "C": {}})

	dag, err := search.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	if edges := dag.Edges(); len(edges) != 0 {
		t.Errorf("learned %v, want no edges when every parent is forbidden", edges)
	}
}

func TestHillClimbSearch_SetAllowedParentsNilRemovesTheRestriction(t *testing.T) {
	data := chainData(4000, 38)

	search := NewHillClimb(data, nil)
	search.SetAllowedParents(map[string][]string{"A": {}, "B": {}, "C": {}})
	search.SetAllowedParents(nil)

	dag, err := search.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	if len(dag.Edges()) == 0 {
		t.Error("learned no edges after the restriction was removed")
	}
}

func TestHillClimbSearch_IsDeterministic(t *testing.T) {
	data := chainData(3000, 39)

	first, err := NewHillClimb(data, nil).Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}
	second, err := NewHillClimb(data, nil).Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	if got, want := sortedKeys(directedEdgesOf(first)), sortedKeys(directedEdgesOf(second)); got != want {
		t.Errorf("two runs gave [%s] and [%s]", got, want)
	}
}

func TestHillClimbSearch_StopsAtMaxIterations(t *testing.T) {
	data := chainData(4000, 40)

	search := NewHillClimb(data, nil)
	search.SetMaxIterations(1)

	dag, err := search.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	if got := len(dag.Edges()); got != 1 {
		t.Errorf("learned %d edges after one iteration, want 1", got)
	}
}

func TestHillClimbSearch_AcceptsAnAlternativeScore(t *testing.T) {
	data := chainData(4000, 41)

	bdeu, err := NewBDeu(data, nil, 10)
	if err != nil {
		t.Fatalf("NewBDeu() error = %v", err)
	}

	search := NewHillClimb(data, nil)
	if err := search.SetScore(bdeu); err != nil {
		t.Fatalf("SetScore() error = %v", err)
	}

	dag, err := search.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	got := skeletonOf(dag)
	want := skeletonOfEdges([][2]string{{"A", "B"}, {"B", "C"}})
	if sortedKeys(got) != sortedKeys(want) {
		t.Errorf("skeleton with BDeu = [%s], want [%s]", sortedKeys(got), sortedKeys(want))
	}
}

func TestHillClimbSearch_SetScoreRejectsNil(t *testing.T) {
	if err := NewHillClimb(chainData(10, 42), nil).SetScore(nil); err == nil {
		t.Error("SetScore(nil) error = nil, want error")
	}
}

func TestHillClimbSearch_ImprovesTheScoreAtEveryStep(t *testing.T) {
	data := chainData(3000, 43)
	score := NewBIC(data, nil)

	search := NewHillClimb(data, nil)
	if err := search.SetScore(score); err != nil {
		t.Fatalf("SetScore() error = %v", err)
	}

	previous := 0.0
	for iterations := 1; iterations <= 4; iterations++ {
		search.SetMaxIterations(iterations)

		dag, err := search.Estimate()
		if err != nil {
			t.Fatalf("Estimate() error = %v", err)
		}
		total, err := ScoreDAG(dag, score)
		if err != nil {
			t.Fatalf("ScoreDAG() error = %v", err)
		}

		if iterations > 1 && total < previous {
			t.Errorf("score fell from %.4f to %.4f at iteration %d", previous, total, iterations)
		}
		previous = total
	}
}

func TestHillClimbSearch_MaxIterationsZeroMeansUncapped(t *testing.T) {
	data := chainData(3000, 44)

	uncapped := NewHillClimb(data, nil)
	uncapped.SetMaxIterations(0)

	capped := NewHillClimb(data, nil)
	capped.SetMaxIterations(1)

	full, err := uncapped.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}
	partial, err := capped.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	if len(full.Edges()) <= len(partial.Edges()) {
		t.Errorf("uncapped search learned %d edges, want more than the %d of a single step",
			len(full.Edges()), len(partial.Edges()))
	}
}

func TestHillClimbSearch_SetEpsilon(t *testing.T) {
	data := chainData(3000, 45)

	search := NewHillClimb(data, nil)
	// No single edge can improve BIC by a million, so nothing is worth applying.
	search.SetEpsilon(1e6)

	dag, err := search.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	if edges := dag.Edges(); len(edges) != 0 {
		t.Errorf("learned %v, want no edges when the threshold is unreachable", edges)
	}
}

func TestHillClimbSearch_BreaksTiesDeterministically(t *testing.T) {
	// A and B are the same variable under two names, so adding A -> B and adding
	// B -> A improve the score by exactly the same amount. The tie-break has to
	// settle it rather than leaving it to rounding.
	data := make([]map[string]int, 0, 400)
	for i := 0; i < 400; i++ {
		value := i % 2
		data = append(data, map[string]int{"A": value, "B": value})
	}

	for run := 0; run < 5; run++ {
		dag, err := NewHillClimb(data, nil).Estimate()
		if err != nil {
			t.Fatalf("Estimate() error = %v", err)
		}

		edges := directedEdgesOf(dag)
		if !edges["A->B"] {
			t.Fatalf("directed edges = [%s], want the tie broken towards A->B", sortedKeys(edges))
		}
	}
}
