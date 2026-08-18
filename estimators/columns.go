package estimators

import (
	"fmt"
	"math"
	"sort"
)

// missingState marks a value that was not observed, or that falls outside the
// declared states of its variable. Normalising both cases to one marker when the
// data is loaded keeps the counting loops free of range checks.
const missingState = -1

// columnData holds observations column by column rather than row by row.
//
// Structure learning reads every value of a variable thousands of times: once per
// independence test and once per family it scores. Reaching those values through a
// map for each row makes hashing, not counting, the dominant cost on a large
// network. Laid out as one slice per variable, a value costs an index.
type columnData struct {
	variables   []string
	cardinality map[string]int
	columns     map[string][]int
	rows        int
}

// newColumnData transposes rows into columns, taking each variable's number of
// states from the data unless a non-empty entry in cardinality overrides it. The
// arguments are not modified.
func newColumnData(rows []map[string]int, cardinality map[string]int) *columnData {
	variables, states := scanVariables(rows)

	for variable, card := range cardinality {
		if card > 0 {
			states[variable] = card
		}
	}

	return buildColumns(rows, variables, states)
}

// newColumnDataFor transposes only the variables asked for, for callers that need a
// handful of columns out of a wide data set.
func newColumnDataFor(rows []map[string]int, cardinality map[string]int, wanted []string) *columnData {
	_, observed := scanVariables(rows)

	states := make(map[string]int, len(wanted))
	variables := make([]string, 0, len(wanted))
	seen := make(map[string]bool, len(wanted))

	for _, variable := range wanted {
		if seen[variable] {
			continue
		}
		seen[variable] = true
		variables = append(variables, variable)

		if card, ok := cardinality[variable]; ok && card > 0 {
			states[variable] = card
			continue
		}
		states[variable] = observed[variable]
	}
	sort.Strings(variables)

	return buildColumns(rows, variables, states)
}

// buildColumns fills one column per variable, writing missingState wherever a value
// is absent or outside the declared states.
func buildColumns(rows []map[string]int, variables []string, states map[string]int) *columnData {
	data := &columnData{
		variables:   variables,
		cardinality: states,
		columns:     make(map[string][]int, len(variables)),
		rows:        len(rows),
	}

	for _, variable := range variables {
		column := make([]int, len(rows))
		card := states[variable]

		for i, row := range rows {
			value, ok := row[variable]
			if !ok || value < 0 || value >= card {
				column[i] = missingState
				continue
			}
			column[i] = value
		}

		data.columns[variable] = column
	}

	return data
}

// scanVariables returns the variables of the data in sorted order, and the number of
// states each one shows, as one more than the largest state seen.
func scanVariables(rows []map[string]int) ([]string, map[string]int) {
	cardinality := make(map[string]int)

	for _, row := range rows {
		for variable, value := range row {
			if value+1 > cardinality[variable] {
				cardinality[variable] = value + 1
			}
		}
	}

	variables := make([]string, 0, len(cardinality))
	for variable := range cardinality {
		variables = append(variables, variable)
	}
	sort.Strings(variables)

	return variables, cardinality
}

// column returns the values of one variable.
func (d *columnData) column(variable string) ([]int, bool) {
	column, ok := d.columns[variable]
	return column, ok
}

// states returns how many states a variable has, or zero when it is unknown.
func (d *columnData) states(variable string) int {
	return d.cardinality[variable]
}

// familyIndexer maps the parent values of one row onto a single configuration index.
type familyIndexer struct {
	columns [][]int
	strides []int64
	configs float64
}

// indexer precomputes the strides of a parent set, so indexing a row is one pass
// over the parent columns.
func (d *columnData) indexer(parents []string) (*familyIndexer, error) {
	indexer := &familyIndexer{
		columns: make([][]int, len(parents)),
		strides: make([]int64, len(parents)),
		configs: 1,
	}

	stride := int64(1)
	for i := len(parents) - 1; i >= 0; i-- {
		card := d.states(parents[i])
		if card <= 0 {
			return nil, fmt.Errorf("variable %s has no observed states", parents[i])
		}

		column, ok := d.column(parents[i])
		if !ok {
			return nil, fmt.Errorf("variable %s is not in the data", parents[i])
		}
		if stride > math.MaxInt64/int64(card) {
			return nil, fmt.Errorf("parent set %v has too many configurations to index", parents)
		}

		indexer.columns[i] = column
		indexer.strides[i] = stride
		stride *= int64(card)
		indexer.configs *= float64(card)
	}

	return indexer, nil
}

// at returns the parent configuration of one row, and false when a parent is
// unobserved there.
func (fi *familyIndexer) at(row int) (int64, bool) {
	index := int64(0)

	for i, column := range fi.columns {
		value := column[row]
		if value == missingState {
			return 0, false
		}
		index += int64(value) * fi.strides[i]
	}

	return index, true
}
