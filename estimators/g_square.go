package estimators

import (
	"math"
)

// denseTableLimit is the largest contingency table gSquare will allocate as a flat
// array. Below it a dense table is much faster than a map, because counting a row
// costs a few array increments instead of hashing; above it the table would waste
// more memory than the sample could ever fill, so counting switches to the sparse
// path.
const denseTableLimit = 1 << 16

// GSquareTest performs the G-squared test for conditional independence, returning the
// statistic and its p-value. It tests whether X is independent of Y given Z.
//
// G-squared is the likelihood ratio counterpart of the chi-square test and is the
// usual choice inside constraint-based structure learning. Three properties matter
// when it is called hundreds of thousands of times over a large network:
//
//   - The degrees of freedom are adjusted to the states actually observed in each
//     stratum, rather than assumed from the cardinalities. Without that adjustment a
//     thinly populated conditioning set inflates the degrees of freedom and the test
//     stops rejecting anything.
//   - Counting uses a flat table while the table stays small, and a sparse one when
//     it would not, so a high cardinality conditioning set costs memory in proportion
//     to the sample rather than to the product of the cardinalities.
//   - The values are read column by column, so counting a row costs a few slice
//     indexes instead of a map lookup per variable.
//
// A cardinality given for a variable overrides the number of states read off the
// data. A variable of Z that never appears in the data cannot be conditioned on, and
// the p-value is 1.
//
// The statistic is a sum over strata, so it is symmetric in X and Y up to the order
// the terms are added in.
//
// Rows where X, Y or any variable in Z is unobserved are skipped. When the adjusted
// degrees of freedom come out at zero the data cannot speak to the question, and the
// p-value is 1, which callers read as independence.
func GSquareTest(data []map[string]int, x, y string, z []string, cardinality map[string]int) (float64, float64) {
	wanted := make([]string, 0, len(z)+2)
	wanted = append(wanted, x, y)
	wanted = append(wanted, z...)

	return gSquare(newColumnDataFor(data, cardinality, wanted), x, y, z)
}

// gSquare runs the test over an existing columnar view, which is how the estimators
// avoid transposing the data once per test.
func gSquare(d *columnData, x, y string, z []string) (float64, float64) {
	xColumn, xOk := d.column(x)
	yColumn, yOk := d.column(y)
	if !xOk || !yOk {
		return 0, 1.0
	}

	indexer, err := d.indexer(z)
	if err != nil {
		return 0, 1.0
	}

	statistic, degreesOfFreedom := countAndReduce(d, xColumn, yColumn, indexer,
		d.states(x), d.states(y))
	if degreesOfFreedom <= 0 {
		return 0, 1.0
	}

	return statistic, chiSquarePValue(statistic, degreesOfFreedom)
}

// countAndReduce builds the contingency tables and reduces them to the statistic and
// its degrees of freedom, choosing between a dense and a sparse table.
func countAndReduce(d *columnData, xColumn, yColumn []int, indexer *familyIndexer,
	xCard, yCard int) (float64, float64) {

	cells := indexer.configs * float64(xCard) * float64(yCard)

	if xCard > 0 && yCard > 0 && cells <= denseTableLimit {
		table := newDenseTable(int(indexer.configs), xCard, yCard)
		table.fill(d.rows, xColumn, yColumn, indexer)
		return table.reduce()
	}

	return sparseReduce(collectStrata(d.rows, xColumn, yColumn, indexer))
}

// denseTable counts one contingency table per conditioning configuration in flat
// arrays.
type denseTable struct {
	configs int
	xCard   int
	yCard   int
	joint   []float64
	xMargin []float64
	yMargin []float64
	totals  []float64
}

// newDenseTable allocates a table for the given shape.
func newDenseTable(configs, xCard, yCard int) *denseTable {
	return &denseTable{
		configs: configs,
		xCard:   xCard,
		yCard:   yCard,
		joint:   make([]float64, configs*xCard*yCard),
		xMargin: make([]float64, configs*xCard),
		yMargin: make([]float64, configs*yCard),
		totals:  make([]float64, configs),
	}
}

// fill counts every row where all the variables were observed.
func (t *denseTable) fill(rows int, xColumn, yColumn []int, indexer *familyIndexer) {
	for row := 0; row < rows; row++ {
		xValue := xColumn[row]
		yValue := yColumn[row]
		if xValue == missingState || yValue == missingState {
			continue
		}

		config, ok := indexer.at(row)
		if !ok {
			continue
		}

		stratum := int(config)
		t.joint[(stratum*t.xCard+xValue)*t.yCard+yValue]++
		t.xMargin[stratum*t.xCard+xValue]++
		t.yMargin[stratum*t.yCard+yValue]++
		t.totals[stratum]++
	}
}

// reduce turns the table into the G-squared statistic and its adjusted degrees of
// freedom.
func (t *denseTable) reduce() (float64, float64) {
	statistic := 0.0
	degreesOfFreedom := 0.0

	for stratum := 0; stratum < t.configs; stratum++ {
		total := t.totals[stratum]
		if total == 0 {
			continue
		}

		xStates := observedStates(t.xMargin, stratum, t.xCard)
		yStates := observedStates(t.yMargin, stratum, t.yCard)
		degreesOfFreedom += float64((xStates - 1) * (yStates - 1))
		statistic += t.stratumRatio(stratum, total)
	}

	return 2 * statistic, degreesOfFreedom
}

// observedStates counts how many states of one variable occur in a stratum.
func observedStates(margin []float64, stratum, card int) int {
	observed := 0
	for state := 0; state < card; state++ {
		if margin[stratum*card+state] > 0 {
			observed++
		}
	}
	return observed
}

// stratumRatio returns the log likelihood ratio contribution of one stratum.
func (t *denseTable) stratumRatio(stratum int, total float64) float64 {
	contribution := 0.0

	for xState := 0; xState < t.xCard; xState++ {
		xCount := t.xMargin[stratum*t.xCard+xState]
		if xCount == 0 {
			continue
		}

		for yState := 0; yState < t.yCard; yState++ {
			observed := t.joint[(stratum*t.xCard+xState)*t.yCard+yState]
			if observed == 0 {
				continue
			}
			expected := xCount * t.yMargin[stratum*t.yCard+yState] / total
			contribution += observed * math.Log(observed/expected)
		}
	}

	return contribution
}

// cell identifies one combination of an X state and a Y state.
type cell [2]int

// stratum holds the contingency table of one conditioning configuration, storing only
// the combinations that occur. It is the fallback for tables too large to allocate
// densely, which is where high cardinality data ends up.
type stratum struct {
	joint   map[cell]float64
	xCounts map[int]float64
	yCounts map[int]float64
	total   float64
}

// newStratum creates an empty stratum.
func newStratum() *stratum {
	return &stratum{
		joint:   make(map[cell]float64),
		xCounts: make(map[int]float64),
		yCounts: make(map[int]float64),
	}
}

// observe records one row.
func (s *stratum) observe(xValue, yValue int) {
	s.joint[cell{xValue, yValue}]++
	s.xCounts[xValue]++
	s.yCounts[yValue]++
	s.total++
}

// collectStrata builds one sparse contingency table per conditioning configuration.
func collectStrata(rows int, xColumn, yColumn []int, indexer *familyIndexer) map[int64]*stratum {
	strata := make(map[int64]*stratum)

	for row := 0; row < rows; row++ {
		xValue := xColumn[row]
		yValue := yColumn[row]
		if xValue == missingState || yValue == missingState {
			continue
		}

		config, ok := indexer.at(row)
		if !ok {
			continue
		}

		current, seen := strata[config]
		if !seen {
			current = newStratum()
			strata[config] = current
		}
		current.observe(xValue, yValue)
	}

	return strata
}

// sparseReduce turns sparse strata into the statistic and its adjusted degrees of
// freedom.
func sparseReduce(strata map[int64]*stratum) (float64, float64) {
	statistic := 0.0
	degreesOfFreedom := 0.0

	for _, current := range strata {
		if current.total == 0 {
			continue
		}
		statistic += current.logLikelihoodRatio()
		degreesOfFreedom += current.degreesOfFreedom()
	}

	return 2 * statistic, degreesOfFreedom
}

// logLikelihoodRatio returns the contribution of one stratum to half the G-squared
// statistic.
func (s *stratum) logLikelihoodRatio() float64 {
	total := 0.0

	for combination, observed := range s.joint {
		expected := s.xCounts[combination[0]] * s.yCounts[combination[1]] / s.total
		if expected > 0 && observed > 0 {
			total += observed * math.Log(observed/expected)
		}
	}

	return total
}

// degreesOfFreedom returns the degrees of freedom of one stratum, counting only the
// states that actually occur in it.
func (s *stratum) degreesOfFreedom() float64 {
	return float64((len(s.xCounts) - 1) * (len(s.yCounts) - 1))
}
