// Package utils provides utility functions for data handling
package utils

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

// DataFrame represents a simple data frame
type DataFrame struct {
	Columns []string
	Data    []map[string]int
}

// NewDataFrame creates a new data frame
func NewDataFrame(columns []string) *DataFrame {
	return &DataFrame{
		Columns: columns,
		Data:    make([]map[string]int, 0),
	}
}

// AddRow adds a row to the data frame
func (df *DataFrame) AddRow(row map[string]int) {
	df.Data = append(df.Data, row)
}

// GetColumn returns all values for a column
func (df *DataFrame) GetColumn(column string) []int {
	values := make([]int, len(df.Data))
	for i, row := range df.Data {
		values[i] = row[column]
	}
	return values
}

// Len returns the number of rows
func (df *DataFrame) Len() int {
	return len(df.Data)
}

// LoadCSV loads a CSV file into a DataFrame
func LoadCSV(filename string) (*DataFrame, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	// Read header
	header, err := reader.Read()
	if err != nil {
		return nil, err
	}

	df := NewDataFrame(header)

	// Read data
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}

		row := make(map[string]int)
		for i, value := range record {
			intValue, err := strconv.Atoi(value)
			if err != nil {
				return nil, fmt.Errorf("invalid integer value %s in column %s", value, header[i])
			}
			row[header[i]] = intValue
		}

		df.AddRow(row)
	}

	return df, nil
}

// SaveCSV saves a DataFrame to a CSV file
func (df *DataFrame) SaveCSV(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	if err := writer.Write(df.Columns); err != nil {
		return err
	}

	// Write data
	for _, row := range df.Data {
		record := make([]string, len(df.Columns))
		for i, col := range df.Columns {
			record[i] = strconv.Itoa(row[col])
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}

// DataFrameFromSamples creates a DataFrame from sample data
func DataFrameFromSamples(samples []map[string]int, columns []string) *DataFrame {
	df := NewDataFrame(columns)
	df.Data = samples
	return df
}

// ToSamples converts a DataFrame to sample format
func (df *DataFrame) ToSamples() []map[string]int {
	return df.Data
}
