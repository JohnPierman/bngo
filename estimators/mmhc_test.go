package estimators

import (
	"testing"
)

func TestMMHCEstimator_LearnsTheSkeletonOfAChain(t *testing.T) {
	skeleton := NewMMHC(chainData(4000, 51), nil).LearnSkeleton()

	got := undirectedEdgesOf(skeleton)
	want := skeletonOfEdges([][2]string{{"A", "B"}, {"B", "C"}})

	if sortedKeys(got) != sortedKeys(want) {
		t.Errorf("skeleton = [%s], want [%s]", sortedKeys(got), sortedKeys(want))
	}
}

func TestMMHCEstimator_LearnsTheSkeletonOfACollider(t *testing.T) {
	skeleton := NewMMHC(colliderData(4000, 52), nil).LearnSkeleton()

	got := undirectedEdgesOf(skeleton)
	want := skeletonOfEdges([][2]string{{"A", "C"}, {"B", "C"}})

	if sortedKeys(got) != sortedKeys(want) {
		t.Errorf("skeleton = [%s], want [%s]", sortedKeys(got), sortedKeys(want))
	}
}

func TestMMHCEstimator_LeavesUnrelatedVariablesUnconnected(t *testing.T) {
	skeleton := NewMMHC(independentData(4000, 53), nil).LearnSkeleton()

	if got := undirectedEdgesOf(skeleton); len(got) != 0 {
		t.Errorf("skeleton = [%s], want no edges", sortedKeys(got))
	}
	if got := len(skeleton.Nodes()); got != 3 {
		t.Errorf("skeleton has %d nodes, want 3", got)
	}
}

func TestMMHCEstimator_RecoversTheSkeletonOfAChain(t *testing.T) {
	dag, err := NewMMHC(chainData(4000, 54), nil).Estimate()
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

func TestMMHCEstimator_OrientsACollider(t *testing.T) {
	dag, err := NewMMHC(colliderData(6000, 55), nil).Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	edges := directedEdgesOf(dag)
	if !edges["A->C"] || !edges["B->C"] {
		t.Errorf("directed edges = [%s], want A->C and B->C", sortedKeys(edges))
	}
}

func TestMMHCEstimator_SearchesOnlyWithinTheSkeleton(t *testing.T) {
	estimator := NewMMHC(ladderData(10, 2000, 56), nil)

	skeleton := undirectedEdgesOf(estimator.LearnSkeleton())

	dag, err := estimator.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	for edge := range skeletonOf(dag) {
		if !skeleton[edge] {
			t.Errorf("learned edge %s is outside the skeleton [%s]", edge, sortedKeys(skeleton))
		}
	}
}

func TestMMHCEstimator_IsDeterministic(t *testing.T) {
	data := ladderData(8, 2000, 57)

	first, err := NewMMHC(data, nil).Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}
	second, err := NewMMHC(data, nil).Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	if got, want := sortedKeys(directedEdgesOf(first)), sortedKeys(directedEdgesOf(second)); got != want {
		t.Errorf("two runs gave [%s] and [%s]", got, want)
	}
}

func TestMMHCEstimator_ASmallerAlphaGivesASparserSkeleton(t *testing.T) {
	data := ladderData(10, 1500, 58)

	loose := NewMMHC(data, nil)
	loose.SetAlpha(0.2)

	strict := NewMMHC(data, nil)
	strict.SetAlpha(1e-8)

	looseEdges := len(undirectedEdgesOf(loose.LearnSkeleton()))
	strictEdges := len(undirectedEdgesOf(strict.LearnSkeleton()))

	if strictEdges > looseEdges {
		t.Errorf("alpha 1e-8 gave %d edges and alpha 0.2 gave %d, want the strict test to be no denser",
			strictEdges, looseEdges)
	}
}

func TestMMHCEstimator_RespectsMaxIndegree(t *testing.T) {
	estimator := NewMMHC(ladderData(10, 2000, 59), nil)
	estimator.SetMaxIndegree(1)

	dag, err := estimator.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	for _, node := range dag.Nodes() {
		if parents := dag.Parents(node); len(parents) > 1 {
			t.Errorf("%s has parents %v, want at most 1", node, parents)
		}
	}
}

func TestMMHCEstimator_AcceptsAnAlternativeScore(t *testing.T) {
	data := chainData(4000, 60)

	estimator := NewMMHC(data, nil)
	estimator.SetScore(NewBIC(data, nil))

	dag, err := estimator.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	got := skeletonOf(dag)
	want := skeletonOfEdges([][2]string{{"A", "B"}, {"B", "C"}})
	if sortedKeys(got) != sortedKeys(want) {
		t.Errorf("skeleton with BIC = [%s], want [%s]", sortedKeys(got), sortedKeys(want))
	}
}

func TestMMHCEstimator_LimitsTheConditioningSetSize(t *testing.T) {
	data := ladderData(8, 1500, 61)

	unlimited := NewMMHC(data, nil)
	unlimited.SetMaxConditioningSetSize(0)

	marginalOnly := NewMMHC(data, nil)
	marginalOnly.SetMaxConditioningSetSize(1)

	// Both must produce a skeleton; conditioning on more variables can only remove
	// edges, never add them.
	broad := undirectedEdgesOf(marginalOnly.LearnSkeleton())
	narrow := undirectedEdgesOf(unlimited.LearnSkeleton())

	for edge := range narrow {
		if !broad[edge] {
			t.Errorf("edge %s survives larger conditioning sets but not smaller ones", edge)
		}
	}
}

func TestMMHCEstimator_EmptyData(t *testing.T) {
	dag, err := NewMMHC(nil, nil).Estimate()
	if err != nil {
		t.Fatalf("Estimate() error = %v", err)
	}

	if got := len(dag.Nodes()); got != 0 {
		t.Errorf("learned %d nodes from no data, want 0", got)
	}
}

func TestMMHCEstimator_SatisfiesTheStructureEstimatorInterface(t *testing.T) {
	learners := []StructureEstimator{
		NewMMHC(chainData(500, 62), nil),
		NewHillClimb(chainData(500, 62), nil),
		NewPC(chainData(500, 62)),
	}

	for _, learner := range learners {
		if _, err := learner.Estimate(); err != nil {
			t.Errorf("Estimate() error = %v", err)
		}
	}
}

// TestStructureLearning_OnALargeSparseNetwork is the stress test for large networks:
// it checks that both score based learners still recover most of a 30 variable
// network, and that neither invents many edges.
func TestStructureLearning_OnALargeSparseNetwork(t *testing.T) {
	const variables = 30

	data := ladderData(variables, 1500, 74)
	want := skeletonOfEdges(ladderEdges(variables))

	learners := map[string]StructureEstimator{
		"HillClimb": NewHillClimb(data, nil),
		"MMHC":      NewMMHC(data, nil),
	}

	for name, learner := range learners {
		t.Run(name, func(t *testing.T) {
			dag, err := learner.Estimate()
			if err != nil {
				t.Fatalf("Estimate() error = %v", err)
			}

			got := skeletonOf(dag)
			truePositives := 0
			for edge := range got {
				if want[edge] {
					truePositives++
				}
			}

			precision := 1.0
			if len(got) > 0 {
				precision = float64(truePositives) / float64(len(got))
			}
			recall := float64(truePositives) / float64(len(want))

			if precision < 0.80 {
				t.Errorf("precision = %.2f over %d learned edges, want at least 0.80",
					precision, len(got))
			}
			if recall < 0.70 {
				t.Errorf("recall = %.2f over %d true edges, want at least 0.70",
					recall, len(want))
			}
			if _, err := dag.TopologicalSort(); err != nil {
				t.Errorf("the learned graph is not acyclic: %v", err)
			}
		})
	}
}
