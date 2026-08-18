package examples

import (
	"fmt"

	"github.com/JohnPierman/bngo/models"
)

// GetWeatherModel returns a Bayesian Network whose variables are labelled rather
// than numbered: Weather is a three state categorical field and Sprinkler and
// WetGrass are binary fields.
//
//	Weather -> Sprinkler -> WetGrass
//	Weather ------------- > WetGrass
//
// Because the states are declared, every query and every result can be stated in
// the vocabulary of the data instead of in state indices.
func GetWeatherModel() (*models.BayesianNetwork, error) {
	edges := [][2]string{
		{"Weather", "Sprinkler"},
		{"Weather", "WetGrass"},
		{"Sprinkler", "WetGrass"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		return nil, err
	}

	states := map[string][]string{
		"Weather":   {"cloudy", "rainy", "sunny"},
		"Sprinkler": {"off", "on"},
		"WetGrass":  {"no", "yes"},
	}
	for variable, labels := range states {
		if err := bn.DeclareStates(variable, labels); err != nil {
			return nil, err
		}
	}

	// P(Weather)
	if err := bn.AddCategoricalCPD("Weather", nil, [][]float64{
		{0.3, 0.2, 0.5},
	}); err != nil {
		return nil, err
	}

	// P(Sprinkler | Weather), one row per weather state
	if err := bn.AddCategoricalCPD("Sprinkler", []string{"Weather"}, [][]float64{
		{0.60, 0.40}, // cloudy
		{0.95, 0.05}, // rainy
		{0.30, 0.70}, // sunny
	}); err != nil {
		return nil, err
	}

	// P(WetGrass | Sprinkler, Weather), with Weather varying fastest
	if err := bn.AddCategoricalCPD("WetGrass", []string{"Sprinkler", "Weather"}, [][]float64{
		{0.90, 0.10}, // off, cloudy
		{0.20, 0.80}, // off, rainy
		{0.95, 0.05}, // off, sunny
		{0.10, 0.90}, // on,  cloudy
		{0.01, 0.99}, // on,  rainy
		{0.10, 0.90}, // on,  sunny
	}); err != nil {
		return nil, err
	}

	if err := bn.CheckModel(); err != nil {
		return nil, err
	}

	return bn, nil
}

// DemonstrateCategoricalNetwork prints a short tour of the labelled workflow:
// simulate labelled rows, refit the parameters from them, and predict a missing
// field.
func DemonstrateCategoricalNetwork() {
	bn, err := GetWeatherModel()
	if err != nil {
		fmt.Printf("Error building model: %v\n", err)
		return
	}

	fmt.Println("=== Categorical Weather Network ===")
	for _, variable := range bn.Codebook().Variables() {
		states, _ := bn.StateNames(variable)
		fmt.Printf("  %-10s %v\n", variable, states.Labels())
	}

	samples, err := bn.SimulateCategorical(5000, 42)
	if err != nil {
		fmt.Printf("Error simulating: %v\n", err)
		return
	}

	fmt.Printf("\nSimulated %d labelled rows, first three:\n", len(samples))
	for _, sample := range samples[:3] {
		fmt.Printf("  Weather=%s Sprinkler=%s WetGrass=%s\n",
			sample["Weather"], sample["Sprinkler"], sample["WetGrass"])
	}

	relearned, err := GetWeatherModel()
	if err != nil {
		fmt.Printf("Error building model: %v\n", err)
		return
	}
	if err := relearned.FitCategorical(samples); err != nil {
		fmt.Printf("Error fitting: %v\n", err)
		return
	}

	cpd, err := relearned.GetCPD("Weather")
	if err != nil {
		fmt.Printf("Error reading CPD: %v\n", err)
		return
	}
	fmt.Printf("\nP(Weather) refitted from the labels: %.3f %.3f %.3f\n",
		cpd.Values[0][0], cpd.Values[0][1], cpd.Values[0][2])

	predictions, err := bn.PredictCategorical([]map[string]string{
		{"Weather": "rainy", "Sprinkler": "off"},
		{"Weather": "sunny", "Sprinkler": "off"},
	})
	if err != nil {
		fmt.Printf("Error predicting: %v\n", err)
		return
	}
	fmt.Printf("\nMost likely WetGrass when rainy: %s, when sunny: %s\n",
		predictions["WetGrass"][0], predictions["WetGrass"][1])
}
