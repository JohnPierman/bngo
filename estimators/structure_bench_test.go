package estimators

import (
	"fmt"
	"testing"
)

// benchmarkSizes are the network sizes the structure learning benchmarks run at.
// They straddle the point where restricting the search starts to pay for itself.
var benchmarkSizes = []int{20, 40, 80}

func BenchmarkHillClimbSearch(b *testing.B) {
	for _, variables := range benchmarkSizes {
		data := ladderData(variables, 2000, 71)

		b.Run(fmt.Sprintf("vars=%d", variables), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := NewHillClimb(data, nil).Estimate(); err != nil {
					b.Fatalf("Estimate() error = %v", err)
				}
			}
		})
	}
}

func BenchmarkMMHCEstimator(b *testing.B) {
	for _, variables := range benchmarkSizes {
		data := ladderData(variables, 2000, 71)

		b.Run(fmt.Sprintf("vars=%d", variables), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := NewMMHC(data, nil).Estimate(); err != nil {
					b.Fatalf("Estimate() error = %v", err)
				}
			}
		})
	}
}

func BenchmarkMMHCEstimator_LearnSkeleton(b *testing.B) {
	for _, variables := range benchmarkSizes {
		data := ladderData(variables, 2000, 71)

		b.Run(fmt.Sprintf("vars=%d", variables), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				NewMMHC(data, nil).LearnSkeleton()
			}
		})
	}
}

func BenchmarkGSquareTest(b *testing.B) {
	data := ladderData(10, 2000, 72)
	cardinality := map[string]int{}
	for i := 0; i < 10; i++ {
		cardinality[nodeName(i)] = 2
	}

	conditioning := [][]string{
		nil,
		{nodeName(2)},
		{nodeName(2), nodeName(3)},
		{nodeName(2), nodeName(3), nodeName(4)},
	}

	for _, z := range conditioning {
		b.Run(fmt.Sprintf("conditioning=%d", len(z)), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				GSquareTest(data, nodeName(0), nodeName(1), z, cardinality)
			}
		})
	}
}

func BenchmarkScore_LocalScore(b *testing.B) {
	data := ladderData(10, 2000, 73)
	parents := []string{nodeName(1), nodeName(2), nodeName(3)}

	b.Run("uncached", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			// A fresh score every time, so every call pays for the counting.
			if _, err := NewBIC(data, nil).LocalScore(nodeName(0), parents); err != nil {
				b.Fatalf("LocalScore() error = %v", err)
			}
		}
	})

	b.Run("cached", func(b *testing.B) {
		score := NewBIC(data, nil)
		if _, err := score.LocalScore(nodeName(0), parents); err != nil {
			b.Fatalf("LocalScore() error = %v", err)
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := score.LocalScore(nodeName(0), parents); err != nil {
				b.Fatalf("LocalScore() error = %v", err)
			}
		}
	})
}
