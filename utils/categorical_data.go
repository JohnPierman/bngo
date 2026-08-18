package utils

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/JohnPierman/bngo/categorical"
)

// CategoricalFrame holds label valued rows, the shape data has before it is
// encoded into the integer states the models work with. A variable missing from
// a row is an unobserved value, which the parameter learners and the estimators
// skip rather than guess.
type CategoricalFrame struct {
	Columns []string
	Rows    []map[string]string
}

// CSVOptions controls how a label valued CSV file is read.
type CSVOptions struct {
	// Delimiter separates fields. Zero means comma.
	Delimiter rune
	// MissingMarkers lists the raw field values that mean "not observed". The
	// default is the empty field only: markers such as "NA" and "?" are not
	// assumed, because they are real labels in some data sets (NA is a country
	// code, for one). Add them explicitly when the data uses them.
	MissingMarkers []string
}

// DefaultCSVOptions returns the conservative defaults: comma separated, and only
// an empty field counts as missing.
func DefaultCSVOptions() CSVOptions {
	return CSVOptions{Delimiter: ',', MissingMarkers: []string{""}}
}

// NewCategoricalFrame creates an empty frame with the given columns.
func NewCategoricalFrame(columns []string) *CategoricalFrame {
	copied := make([]string, len(columns))
	copy(copied, columns)

	return &CategoricalFrame{
		Columns: copied,
		Rows:    make([]map[string]string, 0),
	}
}

// AddRow appends a row to the frame.
func (f *CategoricalFrame) AddRow(row map[string]string) {
	f.Rows = append(f.Rows, row)
}

// Len returns the number of rows.
func (f *CategoricalFrame) Len() int {
	return len(f.Rows)
}

// Column returns the labels of one column in row order, using the empty label
// for rows where the value was not observed.
func (f *CategoricalFrame) Column(name string) []string {
	values := make([]string, len(f.Rows))
	for i, row := range f.Rows {
		values[i] = row[name]
	}
	return values
}

// LoadCategoricalCSV reads a CSV file whose fields are labels rather than
// integers, using the default options. Unlike LoadCSV it does not require the
// data to be numeric, so it is the entry point for categorical and binary
// fields.
func LoadCategoricalCSV(filename string) (*CategoricalFrame, error) {
	return LoadCategoricalCSVWithOptions(filename, DefaultCSVOptions())
}

// LoadCategoricalCSVWithOptions reads a label valued CSV file with explicit
// options.
func LoadCategoricalCSVWithOptions(filename string, options CSVOptions) (*CategoricalFrame, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("opening %s: %w", filename, err)
	}
	defer func() { _ = file.Close() }()

	reader := csv.NewReader(file)
	if options.Delimiter != 0 {
		reader.Comma = options.Delimiter
	}

	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("reading header of %s: %w", filename, err)
	}

	frame := NewCategoricalFrame(header)
	missing := missingMarkerSet(options)

	for lineNumber := 2; ; lineNumber++ {
		record, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("reading %s line %d: %w", filename, lineNumber, err)
		}

		frame.AddRow(rowFromRecord(header, record, missing))
	}

	return frame, nil
}

// missingMarkerSet turns the configured markers into a set, defaulting to the
// empty field when none are configured.
func missingMarkerSet(options CSVOptions) map[string]bool {
	markers := options.MissingMarkers
	if markers == nil {
		markers = []string{""}
	}

	set := make(map[string]bool, len(markers))
	for _, marker := range markers {
		set[marker] = true
	}
	return set
}

// rowFromRecord maps one CSV record onto its header, leaving missing values out
// of the row entirely.
func rowFromRecord(header, record []string, missing map[string]bool) map[string]string {
	row := make(map[string]string, len(header))

	for i, value := range record {
		if i >= len(header) || missing[value] {
			continue
		}
		row[header[i]] = value
	}

	return row
}

// SaveCSV writes the frame back out, using an empty field for values that were
// never observed.
func (f *CategoricalFrame) SaveCSV(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("creating %s: %w", filename, err)
	}
	defer func() { _ = file.Close() }()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if err := writer.Write(f.Columns); err != nil {
		return fmt.Errorf("writing header of %s: %w", filename, err)
	}

	for i, row := range f.Rows {
		record := make([]string, len(f.Columns))
		for j, column := range f.Columns {
			record[j] = row[column]
		}
		if err := writer.Write(record); err != nil {
			return fmt.Errorf("writing %s row %d: %w", filename, i, err)
		}
	}

	writer.Flush()
	return writer.Error()
}

// Encode infers a codebook from the labels present in the frame and returns the
// integer rows the models consume alongside it. Keep the codebook: it is what
// turns query results back into labels.
func (f *CategoricalFrame) Encode() ([]map[string]int, *categorical.Codebook, error) {
	codebook, err := categorical.InferCodebook(f.Rows)
	if err != nil {
		return nil, nil, err
	}

	encoded, err := f.EncodeWith(codebook)
	if err != nil {
		return nil, nil, err
	}

	return encoded, codebook, nil
}

// EncodeWith encodes the frame against an existing codebook, which is what a
// test or scoring set needs so that its states line up with the training data.
func (f *CategoricalFrame) EncodeWith(codebook *categorical.Codebook) ([]map[string]int, error) {
	if codebook == nil {
		return nil, fmt.Errorf("encode: no codebook given")
	}
	return codebook.EncodeRows(f.Rows)
}

// NewCategoricalFrameFromRows decodes integer rows back into a label valued
// frame, for writing simulated or predicted data out in the vocabulary the
// caller started with.
func NewCategoricalFrameFromRows(rows []map[string]int, codebook *categorical.Codebook, columns []string) (*CategoricalFrame, error) {
	if codebook == nil {
		return nil, fmt.Errorf("decode: no codebook given")
	}

	decoded, err := codebook.DecodeRows(rows)
	if err != nil {
		return nil, err
	}

	if columns == nil {
		columns = codebook.Variables()
	}

	frame := NewCategoricalFrame(columns)
	frame.Rows = decoded
	return frame, nil
}
