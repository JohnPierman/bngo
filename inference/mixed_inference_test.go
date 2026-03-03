package inference

import (
	"fmt"
	"math"
	"testing"

	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/models"
)

// buildContinuousXY creates X → Y where X ~ N(0,1), Y|X ~ N(2X+1, 0.5)
func buildContinuousXY(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	edges := [][2]string{{"X", "Y"}}
	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("creating network: %v", err)
	}

	cpdX, err := factors.NewLinearGaussianCPD("X", []string{}, 0.0, map[string]float64{}, 1.0)
	if err != nil {
		t.Fatalf("creating CPD for X: %v", err)
	}
	if err := bn.AddGaussianCPD(cpdX); err != nil {
		t.Fatalf("adding CPD for X: %v", err)
	}

	cpdY, err := factors.NewLinearGaussianCPD("Y", []string{"X"}, 1.0, map[string]float64{"X": 2.0}, 0.5)
	if err != nil {
		t.Fatalf("creating CPD for Y: %v", err)
	}
	if err := bn.AddGaussianCPD(cpdY); err != nil {
		t.Fatalf("adding CPD for Y: %v", err)
	}

	return bn
}

// buildContinuousXYZ creates X → Y → Z
// X ~ N(0,1), Y|X ~ N(2X+1, 0.5), Z|Y ~ N(-0.5Y+1, 0.25)
func buildContinuousXYZ(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	edges := [][2]string{{"X", "Y"}, {"Y", "Z"}}
	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("creating network: %v", err)
	}

	cpdX, err := factors.NewLinearGaussianCPD("X", []string{}, 0.0, map[string]float64{}, 1.0)
	if err != nil {
		t.Fatalf("creating CPD for X: %v", err)
	}
	if err := bn.AddGaussianCPD(cpdX); err != nil {
		t.Fatalf("adding CPD for X: %v", err)
	}

	cpdY, err := factors.NewLinearGaussianCPD("Y", []string{"X"}, 1.0, map[string]float64{"X": 2.0}, 0.5)
	if err != nil {
		t.Fatalf("creating CPD for Y: %v", err)
	}
	if err := bn.AddGaussianCPD(cpdY); err != nil {
		t.Fatalf("adding CPD for Y: %v", err)
	}

	cpdZ, err := factors.NewLinearGaussianCPD("Z", []string{"Y"}, 1.0, map[string]float64{"Y": -0.5}, 0.25)
	if err != nil {
		t.Fatalf("creating CPD for Z: %v", err)
	}
	if err := bn.AddGaussianCPD(cpdZ); err != nil {
		t.Fatalf("adding CPD for Z: %v", err)
	}

	return bn
}

// buildMixedDX creates D → X where D is discrete, X is continuous
// P(D=0)=0.6, P(D=1)=0.4, X|D=0 ~ N(0,1), X|D=1 ~ N(5,1)
func buildMixedDX(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	edges := [][2]string{{"D", "X"}}
	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("creating network: %v", err)
	}

	cpdD, err := factors.NewTabularCPD("D", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)
	if err != nil {
		t.Fatalf("creating CPD for D: %v", err)
	}
	if err := bn.AddCPD(cpdD); err != nil {
		t.Fatalf("adding CPD for D: %v", err)
	}

	statesX := map[string]factors.GaussianParams{
		"0": {Mean: 0.0, Variance: 1.0},
		"1": {Mean: 5.0, Variance: 1.0},
	}
	cpdX, err := factors.NewDiscreteParentGaussianCPD("X", []string{"D"}, map[string]int{"D": 2}, statesX)
	if err != nil {
		t.Fatalf("creating CPD for X: %v", err)
	}
	if err := bn.AddGaussianCPD(cpdX); err != nil {
		t.Fatalf("adding CPD for X: %v", err)
	}

	return bn
}

// buildMixedDXY creates D → X → Y
// D discrete, X continuous with discrete parent, Y continuous with continuous parent
func buildMixedDXY(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	edges := [][2]string{{"D", "X"}, {"X", "Y"}}
	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("creating network: %v", err)
	}

	cpdD, err := factors.NewTabularCPD("D", 2,
		[][]float64{{0.6, 0.4}},
		[]string{},
		map[string]int{},
	)
	if err != nil {
		t.Fatalf("creating CPD for D: %v", err)
	}
	if err := bn.AddCPD(cpdD); err != nil {
		t.Fatalf("adding CPD for D: %v", err)
	}

	statesX := map[string]factors.GaussianParams{
		"0": {Mean: 0.0, Variance: 1.0},
		"1": {Mean: 5.0, Variance: 1.0},
	}
	cpdX, err := factors.NewDiscreteParentGaussianCPD("X", []string{"D"}, map[string]int{"D": 2}, statesX)
	if err != nil {
		t.Fatalf("creating CPD for X: %v", err)
	}
	if err := bn.AddGaussianCPD(cpdX); err != nil {
		t.Fatalf("adding CPD for X: %v", err)
	}

	cpdY, err := factors.NewLinearGaussianCPD("Y", []string{"X"}, 0.0, map[string]float64{"X": 1.0}, 0.1)
	if err != nil {
		t.Fatalf("creating CPD for Y: %v", err)
	}
	if err := bn.AddGaussianCPD(cpdY); err != nil {
		t.Fatalf("adding CPD for Y: %v", err)
	}

	return bn
}

func TestNewMixedVariableElimination(t *testing.T) {
	bn := buildContinuousXY(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}
	if mve.Model != bn {
		t.Error("model reference mismatch")
	}
}

func TestQueryContinuousWithContinuousEvidence(t *testing.T) {
	// X → Y: X ~ N(0,1), Y|X ~ N(2X+1, 0.5)
	// P(Y | X=2) = N(5, 0.5)
	bn := buildContinuousXY(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryContinuous(
		[]string{"Y"},
		MixedEvidence{Continuous: map[string]float64{"X": 2.0}},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	assertFloat(t, "mean[Y]", result.Mean["Y"], 5.0, 1e-9)
	assertFloat(t, "var[Y]", result.Covariance["Y"]["Y"], 0.5, 1e-9)
}

func TestQueryContinuousMarginal(t *testing.T) {
	// X → Y: X ~ N(0,1), Y|X ~ N(2X+1, 0.5)
	// P(Y) = N(1, 4.5)
	bn := buildContinuousXY(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryContinuous(
		[]string{"Y"},
		MixedEvidence{},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	assertFloat(t, "mean[Y]", result.Mean["Y"], 1.0, 1e-9)
	assertFloat(t, "var[Y]", result.Covariance["Y"]["Y"], 4.5, 1e-9)
}

func TestQueryContinuousChain(t *testing.T) {
	// X → Y → Z: query Z given X=2
	// Joint given no evidence:
	//   E[X]=0, E[Y]=1, E[Z]=0.5
	//   Var[X]=1, Var[Y]=4.5, Var[Z]=1.375
	//   Cov[X,Y]=2, Cov[Y,Z]=-2.25, Cov[X,Z]=-1
	// P(Z | X=2): mean=-1.5, var=0.375
	bn := buildContinuousXYZ(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryContinuous(
		[]string{"Z"},
		MixedEvidence{Continuous: map[string]float64{"X": 2.0}},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	assertFloat(t, "mean[Z]", result.Mean["Z"], -1.5, 1e-9)
	assertFloat(t, "var[Z]", result.Covariance["Z"]["Z"], 0.375, 1e-9)
}

func TestQueryContinuousJointMarginal(t *testing.T) {
	// X → Y → Z: query joint [Y,Z] marginal (no evidence)
	bn := buildContinuousXYZ(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryContinuous(
		[]string{"Y", "Z"},
		MixedEvidence{},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	assertFloat(t, "mean[Y]", result.Mean["Y"], 1.0, 1e-9)
	assertFloat(t, "mean[Z]", result.Mean["Z"], 0.5, 1e-9)
	assertFloat(t, "var[Y]", result.Covariance["Y"]["Y"], 4.5, 1e-9)
	assertFloat(t, "var[Z]", result.Covariance["Z"]["Z"], 1.375, 1e-9)
	assertFloat(t, "cov[Y,Z]", result.Covariance["Y"]["Z"], -2.25, 1e-9)
	assertFloat(t, "cov[Z,Y]", result.Covariance["Z"]["Y"], -2.25, 1e-9)
}

func TestQueryContinuousWithDiscreteEvidence(t *testing.T) {
	// D → X: P(D=0)=0.6, X|D=0 ~ N(0,1)
	// P(X | D=0) = N(0, 1)
	bn := buildMixedDX(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryContinuous(
		[]string{"X"},
		MixedEvidence{Discrete: map[string]int{"D": 0}},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	assertFloat(t, "mean[X]", result.Mean["X"], 0.0, 1e-9)
	assertFloat(t, "var[X]", result.Covariance["X"]["X"], 1.0, 1e-9)
}

func TestQueryContinuousWithDiscreteEvidenceState1(t *testing.T) {
	// D → X: P(D=1)=0.4, X|D=1 ~ N(5,1)
	// P(X | D=1) = N(5, 1)
	bn := buildMixedDX(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryContinuous(
		[]string{"X"},
		MixedEvidence{Discrete: map[string]int{"D": 1}},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	assertFloat(t, "mean[X]", result.Mean["X"], 5.0, 1e-9)
	assertFloat(t, "var[X]", result.Covariance["X"]["X"], 1.0, 1e-9)
}

func TestQueryContinuousMixedChain(t *testing.T) {
	// D → X → Y: query Y given D=0
	// X|D=0 ~ N(0,1), Y|X ~ N(X, 0.1)
	// P(Y|D=0) has mean=0, var=1*1+0.1=1.1
	bn := buildMixedDXY(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryContinuous(
		[]string{"Y"},
		MixedEvidence{Discrete: map[string]int{"D": 0}},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	assertFloat(t, "mean[Y]", result.Mean["Y"], 0.0, 1e-9)
	assertFloat(t, "var[Y]", result.Covariance["Y"]["Y"], 1.1, 1e-9)
}

func TestQueryContinuousMarginalMixed(t *testing.T) {
	// D → X: marginal of X with no evidence (hidden D)
	// P(X) = 0.6*N(0,1) + 0.4*N(5,1)
	// Moment-matched: E[X] = 0.6*0 + 0.4*5 = 2.0
	// Var[X] = 0.6*(1+0) + 0.4*(1+25) - 4 = 0.6 + 10.4 - 4 = 7.0
	bn := buildMixedDX(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryContinuous(
		[]string{"X"},
		MixedEvidence{},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	assertFloat(t, "mean[X]", result.Mean["X"], 2.0, 1e-9)
	assertFloat(t, "var[X]", result.Covariance["X"]["X"], 7.0, 1e-9)
}

func TestQueryDiscreteWithContinuousEvidence(t *testing.T) {
	// D → X: query D given X=3.0
	// P(D=0|X=3) ≈ 0.1096, P(D=1|X=3) ≈ 0.8904
	bn := buildMixedDX(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryDiscrete(
		[]string{"D"},
		MixedEvidence{Continuous: map[string]float64{"X": 3.0}},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	// P(D=0|X=3): N(3;0,1)*0.6 / (N(3;0,1)*0.6 + N(3;5,1)*0.4)
	pXgivenD0 := gaussianPDF(3.0, 0.0, 1.0)
	pXgivenD1 := gaussianPDF(3.0, 5.0, 1.0)
	pD0 := pXgivenD0 * 0.6
	pD1 := pXgivenD1 * 0.4
	total := pD0 + pD1
	expectedD0 := pD0 / total
	expectedD1 := pD1 / total

	assertFloat(t, "P(D=0|X=3)", result.Values[0], expectedD0, 1e-6)
	assertFloat(t, "P(D=1|X=3)", result.Values[1], expectedD1, 1e-6)
}

func TestQueryDiscreteMixedChain(t *testing.T) {
	// D → X → Y: query D given Y=3.0
	// Uses marginal P(Y|D=d) to compute posterior
	bn := buildMixedDXY(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryDiscrete(
		[]string{"D"},
		MixedEvidence{Continuous: map[string]float64{"Y": 3.0}},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	// P(Y|D=0): Y marginal given D=0 has mean=0, var=1.1
	// P(Y|D=1): Y marginal given D=1 has mean=5, var=1.1
	pYgivenD0 := gaussianPDF(3.0, 0.0, 1.1)
	pYgivenD1 := gaussianPDF(3.0, 5.0, 1.1)
	pD0 := pYgivenD0 * 0.6
	pD1 := pYgivenD1 * 0.4
	total := pD0 + pD1
	expectedD0 := pD0 / total
	expectedD1 := pD1 / total

	assertFloat(t, "P(D=0|Y=3)", result.Values[0], expectedD0, 1e-6)
	assertFloat(t, "P(D=1|Y=3)", result.Values[1], expectedD1, 1e-6)
}

func TestQueryDiscreteNoEvidence(t *testing.T) {
	// D → X: query D with no evidence, should return prior
	bn := buildMixedDX(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryDiscrete(
		[]string{"D"},
		MixedEvidence{},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	assertFloat(t, "P(D=0)", result.Values[0], 0.6, 1e-6)
	assertFloat(t, "P(D=1)", result.Values[1], 0.4, 1e-6)
}

func TestQueryDiscreteWithDiscreteEvidence(t *testing.T) {
	// D1 → D2: test standard discrete inference path
	edges := [][2]string{{"D1", "D2"}}
	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		t.Fatalf("creating network: %v", err)
	}

	cpdD1, err := factors.NewTabularCPD("D1", 2,
		[][]float64{{0.7, 0.3}},
		[]string{},
		map[string]int{},
	)
	if err != nil {
		t.Fatalf("creating CPD: %v", err)
	}
	if err := bn.AddCPD(cpdD1); err != nil {
		t.Fatalf("adding CPD: %v", err)
	}

	cpdD2, err := factors.NewTabularCPD("D2", 2,
		[][]float64{{0.9, 0.1}, {0.2, 0.8}},
		[]string{"D1"},
		map[string]int{"D1": 2},
	)
	if err != nil {
		t.Fatalf("creating CPD: %v", err)
	}
	if err := bn.AddCPD(cpdD2); err != nil {
		t.Fatalf("adding CPD: %v", err)
	}

	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	result, err := mve.QueryDiscrete(
		[]string{"D2"},
		MixedEvidence{Discrete: map[string]int{"D1": 0}},
	)
	if err != nil {
		t.Fatalf("query failed: %v", err)
	}

	assertFloat(t, "P(D2=0|D1=0)", result.Values[0], 0.9, 1e-6)
	assertFloat(t, "P(D2=1|D1=0)", result.Values[1], 0.1, 1e-6)
}

func TestQueryValidation(t *testing.T) {
	bn := buildMixedDX(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	// Query continuous but variable is discrete
	_, err = mve.QueryContinuous(
		[]string{"D"},
		MixedEvidence{},
	)
	if err == nil {
		t.Error("expected error for discrete variable in continuous query")
	}

	// Query discrete but variable is continuous
	_, err = mve.QueryDiscrete(
		[]string{"X"},
		MixedEvidence{},
	)
	if err == nil {
		t.Error("expected error for continuous variable in discrete query")
	}

	// Empty query
	_, err = mve.QueryContinuous([]string{}, MixedEvidence{})
	if err == nil {
		t.Error("expected error for empty query")
	}

	// Evidence for non-existent variable
	_, err = mve.QueryContinuous(
		[]string{"X"},
		MixedEvidence{Discrete: map[string]int{"NONEXISTENT": 0}},
	)
	if err == nil {
		t.Error("expected error for non-existent evidence variable")
	}

	// Query variable in evidence
	_, err = mve.QueryContinuous(
		[]string{"X"},
		MixedEvidence{Continuous: map[string]float64{"X": 1.0}},
	)
	if err == nil {
		t.Error("expected error for query variable in evidence")
	}
}

func TestEnumerateDiscreteConfigs(t *testing.T) {
	configs := enumerateDiscreteConfigs([]string{}, map[string]int{})
	if len(configs) != 1 {
		t.Errorf("expected 1 config for empty vars, got %d", len(configs))
	}

	configs = enumerateDiscreteConfigs(
		[]string{"A", "B"},
		map[string]int{"A": 2, "B": 3},
	)
	if len(configs) != 6 {
		t.Errorf("expected 6 configs, got %d", len(configs))
	}

	seen := make(map[string]bool)
	for _, c := range configs {
		key := fmt.Sprintf("A=%d,B=%d", c["A"], c["B"])
		if seen[key] {
			t.Errorf("duplicate config: %s", key)
		}
		seen[key] = true
	}
}

func TestBuildJointGaussian(t *testing.T) {
	// X → Y: verify joint distribution parameters directly
	bn := buildContinuousXY(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	joint, err := mve.buildJointGaussian(map[string]int{})
	if err != nil {
		t.Fatalf("building joint: %v", err)
	}

	assertFloat(t, "mean[X]", joint.Mean["X"], 0.0, 1e-9)
	assertFloat(t, "mean[Y]", joint.Mean["Y"], 1.0, 1e-9)
	assertFloat(t, "cov[X][X]", joint.Covariance["X"]["X"], 1.0, 1e-9)
	assertFloat(t, "cov[Y][Y]", joint.Covariance["Y"]["Y"], 4.5, 1e-9)
	assertFloat(t, "cov[X][Y]", joint.Covariance["X"]["Y"], 2.0, 1e-9)
	assertFloat(t, "cov[Y][X]", joint.Covariance["Y"]["X"], 2.0, 1e-9)
}

func TestBuildJointGaussianMixed(t *testing.T) {
	// D → X → Y: verify joint given D=0
	bn := buildMixedDXY(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	joint, err := mve.buildJointGaussian(map[string]int{"D": 0})
	if err != nil {
		t.Fatalf("building joint: %v", err)
	}

	// X|D=0 ~ N(0, 1), Y|X ~ N(X, 0.1)
	assertFloat(t, "mean[X]", joint.Mean["X"], 0.0, 1e-9)
	assertFloat(t, "mean[Y]", joint.Mean["Y"], 0.0, 1e-9)
	assertFloat(t, "cov[X][X]", joint.Covariance["X"]["X"], 1.0, 1e-9)
	assertFloat(t, "cov[Y][Y]", joint.Covariance["Y"]["Y"], 1.1, 1e-9)
	assertFloat(t, "cov[X][Y]", joint.Covariance["X"]["Y"], 1.0, 1e-9)
}

func TestBuildJointGaussianThreeNodeChain(t *testing.T) {
	// X → Y → Z: verify full covariance structure
	bn := buildContinuousXYZ(t)
	mve, err := NewMixedVariableElimination(bn)
	if err != nil {
		t.Fatalf("creating MixedVE: %v", err)
	}

	joint, err := mve.buildJointGaussian(map[string]int{})
	if err != nil {
		t.Fatalf("building joint: %v", err)
	}

	// Z|Y ~ N(-0.5Y+1, 0.25), Y has var=4.5, cov(X,Y)=2
	assertFloat(t, "mean[Z]", joint.Mean["Z"], 0.5, 1e-9)
	assertFloat(t, "cov[Z][Z]", joint.Covariance["Z"]["Z"], 1.375, 1e-9)
	assertFloat(t, "cov[Y][Z]", joint.Covariance["Y"]["Z"], -2.25, 1e-9)
	assertFloat(t, "cov[X][Z]", joint.Covariance["X"]["Z"], -1.0, 1e-9)
}

func gaussianPDF(x, mean, variance float64) float64 {
	diff := x - mean
	return math.Exp(-diff*diff/(2*variance)) / math.Sqrt(2*math.Pi*variance)
}

func assertFloat(t *testing.T, name string, got, expected, tol float64) {
	t.Helper()
	if math.Abs(got-expected) > tol {
		t.Errorf("%s = %.10f, expected %.10f (diff=%.2e, tol=%.2e)",
			name, got, expected, math.Abs(got-expected), tol)
	}
}
