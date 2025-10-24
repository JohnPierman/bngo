package factors

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// GaussianFactor represents a Gaussian (normal) distribution over continuous variables
// Parameterized as: N(x; μ, Σ) = (2π)^(-k/2) |Σ|^(-1/2) exp(-0.5(x-μ)^T Σ^(-1) (x-μ))
// For computational efficiency, we store in canonical form:
// φ(x) = exp(-0.5 x^T K x + h^T x + g)
// where K is the precision matrix (inverse covariance), h is the information vector
type GaussianFactor struct {
	Variables  []string                      // Variable names
	Mean       map[string]float64            // Mean for each variable (μ)
	Covariance map[string]map[string]float64 // Covariance matrix (Σ)
}

// NewGaussianFactor creates a new Gaussian factor
func NewGaussianFactor(variables []string, mean map[string]float64, covariance map[string]map[string]float64) (*GaussianFactor, error) {
	// Validate inputs
	if len(variables) == 0 {
		return nil, fmt.Errorf("variables cannot be empty")
	}

	// Check mean has all variables
	for _, v := range variables {
		if _, ok := mean[v]; !ok {
			return nil, fmt.Errorf("mean missing for variable %s", v)
		}
	}

	// Check covariance matrix
	for _, v1 := range variables {
		if _, ok := covariance[v1]; !ok {
			return nil, fmt.Errorf("covariance missing for variable %s", v1)
		}
		for _, v2 := range variables {
			if _, ok := covariance[v1][v2]; !ok {
				return nil, fmt.Errorf("covariance missing for variables %s, %s", v1, v2)
			}
			// Check symmetry
			if math.Abs(covariance[v1][v2]-covariance[v2][v1]) > 1e-9 {
				return nil, fmt.Errorf("covariance matrix not symmetric")
			}
		}
	}

	return &GaussianFactor{
		Variables:  variables,
		Mean:       mean,
		Covariance: covariance,
	}, nil
}

// Copy creates a deep copy of the Gaussian factor
func (gf *GaussianFactor) Copy() *GaussianFactor {
	varsCopy := make([]string, len(gf.Variables))
	copy(varsCopy, gf.Variables)

	meanCopy := make(map[string]float64)
	for k, v := range gf.Mean {
		meanCopy[k] = v
	}

	covCopy := make(map[string]map[string]float64)
	for k1, v1 := range gf.Covariance {
		covCopy[k1] = make(map[string]float64)
		for k2, v2 := range v1 {
			covCopy[k1][k2] = v2
		}
	}

	return &GaussianFactor{
		Variables:  varsCopy,
		Mean:       meanCopy,
		Covariance: covCopy,
	}
}

// Marginalize marginalizes out specified variables
// For Gaussian: P(X) = ∫ P(X,Y) dY simply drops Y from the distribution
func (gf *GaussianFactor) Marginalize(variables []string) (*GaussianFactor, error) {
	toRemove := make(map[string]bool)
	for _, v := range variables {
		toRemove[v] = true
	}

	// Find remaining variables
	newVars := make([]string, 0)
	for _, v := range gf.Variables {
		if !toRemove[v] {
			newVars = append(newVars, v)
		}
	}

	if len(newVars) == 0 {
		return nil, fmt.Errorf("cannot marginalize out all variables")
	}

	// Extract sub-mean and sub-covariance
	newMean := make(map[string]float64)
	newCov := make(map[string]map[string]float64)

	for _, v1 := range newVars {
		newMean[v1] = gf.Mean[v1]
		newCov[v1] = make(map[string]float64)
		for _, v2 := range newVars {
			newCov[v1][v2] = gf.Covariance[v1][v2]
		}
	}

	return NewGaussianFactor(newVars, newMean, newCov)
}

// Reduce conditions the Gaussian on observed values
// For Gaussian: P(X|Y=y) is also Gaussian with updated parameters
func (gf *GaussianFactor) Reduce(evidence map[string]float64) (*GaussianFactor, error) {
	// Split variables into observed and unobserved
	observed := make([]string, 0)
	unobserved := make([]string, 0)

	for _, v := range gf.Variables {
		if _, ok := evidence[v]; ok {
			observed = append(observed, v)
		} else {
			unobserved = append(unobserved, v)
		}
	}

	if len(observed) == 0 {
		// No evidence, return copy
		return gf.Copy(), nil
	}

	if len(unobserved) == 0 {
		// All variables observed, return point mass
		return nil, fmt.Errorf("all variables observed, cannot reduce to Gaussian")
	}

	// Compute conditional distribution: P(X1 | X2 = x2)
	// X1 = unobserved, X2 = observed
	// μ_1|2 = μ_1 + Σ_12 Σ_22^(-1) (x2 - μ_2)
	// Σ_1|2 = Σ_11 - Σ_12 Σ_22^(-1) Σ_21

	newMean := make(map[string]float64)
	newCov := make(map[string]map[string]float64)

	// Get Σ_22^(-1) (covariance of observed variables inverted)
	sigma22Inv, err := gf.invertSubMatrix(observed)
	if err != nil {
		return nil, fmt.Errorf("failed to invert covariance matrix: %v", err)
	}

	// Compute adjustment: Σ_12 Σ_22^(-1) (x2 - μ_2)
	adjustment := make(map[string]float64)
	for _, v1 := range unobserved {
		sum := 0.0
		for i, v2i := range observed {
			for j, v2j := range observed {
				sum += gf.Covariance[v1][v2i] * sigma22Inv[i][j] * (evidence[v2j] - gf.Mean[v2j])
			}
		}
		adjustment[v1] = sum
	}

	// Compute new mean: μ_1|2 = μ_1 + adjustment
	for _, v := range unobserved {
		newMean[v] = gf.Mean[v] + adjustment[v]
	}

	// Compute new covariance: Σ_1|2 = Σ_11 - Σ_12 Σ_22^(-1) Σ_21
	for _, v1 := range unobserved {
		newCov[v1] = make(map[string]float64)
		for _, v2 := range unobserved {
			val := gf.Covariance[v1][v2]
			// Subtract Σ_12 Σ_22^(-1) Σ_21
			for i, obs1 := range observed {
				for j, obs2 := range observed {
					val -= gf.Covariance[v1][obs1] * sigma22Inv[i][j] * gf.Covariance[obs2][v2]
				}
			}
			newCov[v1][v2] = val
		}
	}

	return NewGaussianFactor(unobserved, newMean, newCov)
}

// invertSubMatrix inverts the covariance submatrix for given variables
func (gf *GaussianFactor) invertSubMatrix(vars []string) ([][]float64, error) {
	n := len(vars)

	// Build matrix
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
		for j := range matrix[i] {
			matrix[i][j] = gf.Covariance[vars[i]][vars[j]]
		}
	}

	// Use Gaussian elimination with partial pivoting
	inv := make([][]float64, n)
	for i := range inv {
		inv[i] = make([]float64, n)
		inv[i][i] = 1.0
	}

	// Forward elimination
	for i := 0; i < n; i++ {
		// Find pivot
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(matrix[k][i]) > math.Abs(matrix[maxRow][i]) {
				maxRow = k
			}
		}

		// Swap rows
		matrix[i], matrix[maxRow] = matrix[maxRow], matrix[i]
		inv[i], inv[maxRow] = inv[maxRow], inv[i]

		// Check for singular matrix
		if math.Abs(matrix[i][i]) < 1e-10 {
			return nil, fmt.Errorf("matrix is singular or nearly singular")
		}

		// Scale pivot row
		pivot := matrix[i][i]
		for j := 0; j < n; j++ {
			matrix[i][j] /= pivot
			inv[i][j] /= pivot
		}

		// Eliminate column
		for k := 0; k < n; k++ {
			if k != i {
				factor := matrix[k][i]
				for j := 0; j < n; j++ {
					matrix[k][j] -= factor * matrix[i][j]
					inv[k][j] -= factor * inv[i][j]
				}
			}
		}
	}

	return inv, nil
}

// Multiply multiplies this Gaussian factor with another
// For independent Gaussians: multiplication gives joint distribution
// For dependent: requires more complex operations (not fully implemented)
func (gf *GaussianFactor) Multiply(other *GaussianFactor) (*GaussianFactor, error) {
	// Find union of variables
	varSet := make(map[string]bool)
	for _, v := range gf.Variables {
		varSet[v] = true
	}
	for _, v := range other.Variables {
		varSet[v] = true
	}

	newVars := make([]string, 0, len(varSet))
	for v := range varSet {
		newVars = append(newVars, v)
	}
	sort.Strings(newVars)

	// Simple case: disjoint variables (independent)
	disjoint := true
	for _, v := range gf.Variables {
		for _, v2 := range other.Variables {
			if v == v2 {
				disjoint = false
				break
			}
		}
		if !disjoint {
			break
		}
	}

	if disjoint {
		// Independent: just concatenate
		newMean := make(map[string]float64)
		newCov := make(map[string]map[string]float64)

		for _, v := range newVars {
			if _, ok := gf.Mean[v]; ok {
				newMean[v] = gf.Mean[v]
			} else {
				newMean[v] = other.Mean[v]
			}

			newCov[v] = make(map[string]float64)
			for _, v2 := range newVars {
				if _, ok1 := gf.Mean[v]; ok1 {
					if _, ok2 := gf.Mean[v2]; ok2 {
						newCov[v][v2] = gf.Covariance[v][v2]
					} else {
						newCov[v][v2] = 0.0
					}
				} else {
					if _, ok2 := other.Mean[v2]; ok2 {
						newCov[v][v2] = other.Covariance[v][v2]
					} else {
						newCov[v][v2] = 0.0
					}
				}
			}
		}

		return NewGaussianFactor(newVars, newMean, newCov)
	}

	// Overlapping variables: use canonical form multiplication
	// This is more complex and requires precision matrix operations
	return nil, fmt.Errorf("multiplication of dependent Gaussian factors not yet fully implemented")
}

// PDF evaluates the probability density at a given point
func (gf *GaussianFactor) PDF(values map[string]float64) (float64, error) {
	// Check all variables present
	for _, v := range gf.Variables {
		if _, ok := values[v]; !ok {
			return 0, fmt.Errorf("missing value for variable %s", v)
		}
	}

	n := len(gf.Variables)

	// Compute (x - μ)
	diff := make([]float64, n)
	for i, v := range gf.Variables {
		diff[i] = values[v] - gf.Mean[v]
	}

	// Compute Σ^(-1) (x - μ)
	sigmaInv, err := gf.invertSubMatrix(gf.Variables)
	if err != nil {
		return 0, err
	}

	// Compute (x - μ)^T Σ^(-1) (x - μ)
	quadForm := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			quadForm += diff[i] * sigmaInv[i][j] * diff[j]
		}
	}

	// Compute determinant of covariance matrix
	det := gf.determinant()
	if det <= 0 {
		return 0, fmt.Errorf("covariance matrix not positive definite")
	}

	// Compute PDF: (2π)^(-k/2) |Σ|^(-1/2) exp(-0.5 * quadForm)
	normConst := math.Pow(2*math.Pi, -float64(n)/2.0) * math.Pow(det, -0.5)
	pdf := normConst * math.Exp(-0.5*quadForm)

	return pdf, nil
}

// determinant computes the determinant of the covariance matrix
func (gf *GaussianFactor) determinant() float64 {
	n := len(gf.Variables)

	// Build matrix
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
		for j := range matrix[i] {
			matrix[i][j] = gf.Covariance[gf.Variables[i]][gf.Variables[j]]
		}
	}

	// Compute determinant using LU decomposition
	det := 1.0
	for i := 0; i < n; i++ {
		// Find pivot
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(matrix[k][i]) > math.Abs(matrix[maxRow][i]) {
				maxRow = k
			}
		}

		if maxRow != i {
			matrix[i], matrix[maxRow] = matrix[maxRow], matrix[i]
			det *= -1
		}

		if math.Abs(matrix[i][i]) < 1e-10 {
			return 0
		}

		det *= matrix[i][i]

		// Eliminate
		for k := i + 1; k < n; k++ {
			factor := matrix[k][i] / matrix[i][i]
			for j := i; j < n; j++ {
				matrix[k][j] -= factor * matrix[i][j]
			}
		}
	}

	return det
}

// String returns a string representation
func (gf *GaussianFactor) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("GaussianFactor(%s)\n", strings.Join(gf.Variables, ", ")))
	sb.WriteString("Mean: ")
	for _, v := range gf.Variables {
		sb.WriteString(fmt.Sprintf("%s=%.4f ", v, gf.Mean[v]))
	}
	sb.WriteString("\nCovariance:\n")
	for _, v1 := range gf.Variables {
		sb.WriteString("  ")
		for _, v2 := range gf.Variables {
			sb.WriteString(fmt.Sprintf("%.4f ", gf.Covariance[v1][v2]))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}
