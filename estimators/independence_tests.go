// Package estimators provides structure and parameter learning algorithms
package estimators

import (
	"math"
)

// ChiSquareTest performs a chi-square test for conditional independence
// Tests if X is independent of Y given Z in the data
func ChiSquareTest(data []map[string]int, x, y string, z []string, cardinality map[string]int) (float64, float64) {
	// Build contingency table
	// Dimensions: [x_states][y_states][z_states_combination]

	zCard := 1
	for _, zVar := range z {
		zCard *= cardinality[zVar]
	}

	xCard := cardinality[x]
	yCard := cardinality[y]

	// Count occurrences
	counts := make([][][]float64, xCard)
	for i := range counts {
		counts[i] = make([][]float64, yCard)
		for j := range counts[i] {
			counts[i][j] = make([]float64, zCard)
		}
	}

	totalCounts := make([]float64, zCard)

	for _, sample := range data {
		xVal, xOk := sample[x]
		yVal, yOk := sample[y]

		if !xOk || !yOk {
			continue
		}

		// Calculate z index
		zIdx := 0
		zStride := 1
		zOk := true
		for i := len(z) - 1; i >= 0; i-- {
			zVar := z[i]
			zVal, ok := sample[zVar]
			if !ok {
				zOk = false
				break
			}
			zIdx += zVal * zStride
			zStride *= cardinality[zVar]
		}

		if !zOk {
			continue
		}

		counts[xVal][yVal][zIdx]++
		totalCounts[zIdx]++
	}

	// Calculate chi-square statistic
	chiSquare := 0.0

	for k := 0; k < zCard; k++ {
		if totalCounts[k] < 5 {
			continue
		}

		// Calculate marginals
		xMarginal := make([]float64, xCard)
		yMarginal := make([]float64, yCard)

		for i := 0; i < xCard; i++ {
			for j := 0; j < yCard; j++ {
				xMarginal[i] += counts[i][j][k]
				yMarginal[j] += counts[i][j][k]
			}
		}

		// Calculate expected counts and chi-square
		for i := 0; i < xCard; i++ {
			for j := 0; j < yCard; j++ {
				expected := xMarginal[i] * yMarginal[j] / totalCounts[k]
				if expected > 0 {
					observed := counts[i][j][k]
					chiSquare += math.Pow(observed-expected, 2) / expected
				}
			}
		}
	}

	// Degrees of freedom
	df := float64((xCard - 1) * (yCard - 1) * zCard)

	// Calculate p-value (approximation using chi-square distribution)
	pValue := chiSquarePValue(chiSquare, df)

	return chiSquare, pValue
}

// chiSquarePValue computes the p-value for chi-square test using incomplete gamma function
func chiSquarePValue(chiSquare, df float64) float64 {
	if df <= 0 {
		return 1.0
	}

	// For very large chi-square values, p-value is essentially 0
	if chiSquare > 1000 {
		return 0.0
	}

	// For very small chi-square values, p-value is essentially 1
	if chiSquare < 0.001 {
		return 1.0
	}

	// P(X > x) = 1 - P(X <= x) = 1 - regularizedGammaP(df/2, x/2)
	k := df / 2
	x := chiSquare / 2

	// Use regularized incomplete gamma function
	pValue := 1.0 - regularizedGammaP(k, x)

	if pValue > 1.0 {
		pValue = 1.0
	}
	if pValue < 0.0 {
		pValue = 0.0
	}

	return pValue
}

// regularizedGammaP computes the regularized incomplete gamma function P(a,x)
// P(a,x) = γ(a,x) / Γ(a) where γ(a,x) is the lower incomplete gamma function
func regularizedGammaP(a, x float64) float64 {
	if x < 0 || a <= 0 {
		return 0.0
	}

	if x == 0 {
		return 0.0
	}

	// Use series expansion for small x or continued fraction for large x
	if x < a+1 {
		return gammaSeriesExpansion(a, x)
	}
	return 1.0 - gammaContinuedFraction(a, x)
}

// gammaSeriesExpansion computes P(a,x) using series expansion
func gammaSeriesExpansion(a, x float64) float64 {
	const maxIter = 200
	const epsilon = 1e-10

	// Series: P(a,x) = e^(-x) * x^a * Σ(Γ(a)/Γ(a+1+n) * x^n)
	ap := a
	sum := 1.0 / a
	del := sum

	for n := 0; n < maxIter; n++ {
		ap++
		del *= x / ap
		sum += del
		if math.Abs(del) < math.Abs(sum)*epsilon {
			break
		}
	}

	return sum * math.Exp(-x+a*math.Log(x)-logGamma(a))
}

// gammaContinuedFraction computes Q(a,x) = 1 - P(a,x) using continued fraction
func gammaContinuedFraction(a, x float64) float64 {
	const maxIter = 200
	const epsilon = 1e-10
	const fpmin = 1e-30

	// Lentz's algorithm for continued fraction
	b := x + 1.0 - a
	c := 1.0 / fpmin
	d := 1.0 / b
	h := d

	for i := 1; i <= maxIter; i++ {
		an := -float64(i) * (float64(i) - a)
		b += 2.0
		d = an*d + b
		if math.Abs(d) < fpmin {
			d = fpmin
		}
		c = b + an/c
		if math.Abs(c) < fpmin {
			c = fpmin
		}
		d = 1.0 / d
		del := d * c
		h *= del
		if math.Abs(del-1.0) < epsilon {
			break
		}
	}

	return math.Exp(-x+a*math.Log(x)-logGamma(a)) * h
}

// logGamma computes the natural logarithm of the gamma function
func logGamma(x float64) float64 {
	// Lanczos approximation
	const g = 7.0
	coef := []float64{
		0.99999999999980993,
		676.5203681218851,
		-1259.1392167224028,
		771.32342877765313,
		-176.61502916214059,
		12.507343278686905,
		-0.13857109526572012,
		9.9843695780195716e-6,
		1.5056327351493116e-7,
	}

	if x < 0.5 {
		// Use reflection formula: Γ(1-x)Γ(x) = π/sin(πx)
		return math.Log(math.Pi) - math.Log(math.Sin(math.Pi*x)) - logGamma(1-x)
	}

	x--
	base := x + g + 0.5
	sum := coef[0]
	for i := 1; i < len(coef); i++ {
		sum += coef[i] / (x + float64(i))
	}

	return math.Log(sum) + math.Log(math.Sqrt(2*math.Pi)) - base + (x+0.5)*math.Log(base)
}

// PearsonCorrelation calculates Pearson correlation coefficient
func PearsonCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}

	n := float64(len(x))

	// Calculate means
	meanX := 0.0
	meanY := 0.0
	for i := range x {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= n
	meanY /= n

	// Calculate correlation
	numerator := 0.0
	denomX := 0.0
	denomY := 0.0

	for i := range x {
		dx := x[i] - meanX
		dy := y[i] - meanY
		numerator += dx * dy
		denomX += dx * dx
		denomY += dy * dy
	}

	if denomX == 0 || denomY == 0 {
		return 0.0
	}

	return numerator / math.Sqrt(denomX*denomY)
}

// PartialCorrelation calculates partial correlation between X and Y given Z
// Uses recursive formula for multiple conditioning variables
func PartialCorrelation(data [][]float64, xIdx, yIdx int, zIdxs []int) float64 {
	if len(zIdxs) == 0 {
		// No conditioning variables, return simple correlation
		xVals := make([]float64, len(data))
		yVals := make([]float64, len(data))
		for i := range data {
			xVals[i] = data[i][xIdx]
			yVals[i] = data[i][yIdx]
		}
		return PearsonCorrelation(xVals, yVals)
	}

	if len(zIdxs) == 1 {
		// First-order partial correlation
		zIdx := zIdxs[0]

		xVals := make([]float64, len(data))
		yVals := make([]float64, len(data))
		zVals := make([]float64, len(data))

		for i := range data {
			xVals[i] = data[i][xIdx]
			yVals[i] = data[i][yIdx]
			zVals[i] = data[i][zIdx]
		}

		rXY := PearsonCorrelation(xVals, yVals)
		rXZ := PearsonCorrelation(xVals, zVals)
		rYZ := PearsonCorrelation(yVals, zVals)

		numerator := rXY - rXZ*rYZ
		denominator := math.Sqrt((1 - rXZ*rXZ) * (1 - rYZ*rYZ))

		if denominator == 0 {
			return 0.0
		}

		return numerator / denominator
	}

	// For multiple conditioning variables, use recursive formula:
	// ρ(X,Y|Z1,...,Zn) = (ρ(X,Y|Z1,...,Zn-1) - ρ(X,Zn|Z1,...,Zn-1)*ρ(Y,Zn|Z1,...,Zn-1)) /
	//                    sqrt((1-ρ²(X,Zn|Z1,...,Zn-1)) * (1-ρ²(Y,Zn|Z1,...,Zn-1)))
	lastZ := zIdxs[len(zIdxs)-1]
	restZ := zIdxs[:len(zIdxs)-1]

	rXY_RestZ := PartialCorrelation(data, xIdx, yIdx, restZ)
	rXLastZ_RestZ := PartialCorrelation(data, xIdx, lastZ, restZ)
	rYLastZ_RestZ := PartialCorrelation(data, yIdx, lastZ, restZ)

	numerator := rXY_RestZ - rXLastZ_RestZ*rYLastZ_RestZ
	denominator := math.Sqrt((1 - rXLastZ_RestZ*rXLastZ_RestZ) * (1 - rYLastZ_RestZ*rYLastZ_RestZ))

	if denominator == 0 || math.IsNaN(denominator) {
		return 0.0
	}

	result := numerator / denominator

	// Handle numerical issues
	if math.IsNaN(result) || math.IsInf(result, 0) {
		return 0.0
	}

	// Clamp to valid correlation range
	if result > 1.0 {
		result = 1.0
	}
	if result < -1.0 {
		result = -1.0
	}

	return result
}

// FisherZ performs Fisher's Z-test for (partial) correlation
func FisherZ(correlation float64, sampleSize int, numCondVars int) float64 {
	// Adjust sample size for conditioning variables
	adjustedN := float64(sampleSize - numCondVars - 3)

	if adjustedN <= 0 {
		return 1.0
	}

	// Fisher Z-transformation
	if correlation >= 1.0 {
		correlation = 0.9999
	}
	if correlation <= -1.0 {
		correlation = -0.9999
	}

	z := 0.5 * math.Log((1+correlation)/(1-correlation))
	testStat := math.Abs(z) * math.Sqrt(adjustedN)

	// Approximate p-value using normal distribution
	pValue := 2 * (1 - normalCDF(testStat))

	return pValue
}

// normalCDF approximates the cumulative distribution function of standard normal
func normalCDF(x float64) float64 {
	// Using error function approximation
	return 0.5 * (1 + math.Erf(x/math.Sqrt(2)))
}
