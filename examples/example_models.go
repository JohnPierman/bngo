// Package examples provides example Bayesian Network models
package examples

import (
	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/models"
)

// GetStudentModel creates the classic Student Bayesian Network
// Variables: Difficulty (D), Intelligence (I), Grade (G), SAT (S), Letter (L)
func GetStudentModel() (*models.BayesianNetwork, error) {
	edges := [][2]string{
		{"Difficulty", "Grade"},
		{"Intelligence", "Grade"},
		{"Intelligence", "SAT"},
		{"Grade", "Letter"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		return nil, err
	}

	// P(Difficulty) - 0: easy, 1: hard
	cpdD, _ := factors.NewTabularCPD("Difficulty", 2,
		[][]float64{
			{0.6, 0.4},
		},
		[]string{},
		map[string]int{},
	)

	// P(Intelligence) - 0: low, 1: high
	cpdI, _ := factors.NewTabularCPD("Intelligence", 2,
		[][]float64{
			{0.7, 0.3},
		},
		[]string{},
		map[string]int{},
	)

	// P(Grade | Difficulty, Intelligence) - 0: A, 1: B, 2: C
	cpdG, _ := factors.NewTabularCPD("Grade", 3,
		[][]float64{
			{0.3, 0.4, 0.3},   // D=0, I=0 (easy, low)
			{0.05, 0.25, 0.7}, // D=0, I=1 (easy, high)
			{0.9, 0.08, 0.02}, // D=1, I=0 (hard, low)
			{0.5, 0.3, 0.2},   // D=1, I=1 (hard, high)
		},
		[]string{"Difficulty", "Intelligence"},
		map[string]int{"Difficulty": 2, "Intelligence": 2},
	)

	// P(SAT | Intelligence) - 0: low, 1: high
	cpdS, _ := factors.NewTabularCPD("SAT", 2,
		[][]float64{
			{0.95, 0.05}, // I=0 (low intelligence)
			{0.2, 0.8},   // I=1 (high intelligence)
		},
		[]string{"Intelligence"},
		map[string]int{"Intelligence": 2},
	)

	// P(Letter | Grade) - 0: weak, 1: strong
	cpdL, _ := factors.NewTabularCPD("Letter", 2,
		[][]float64{
			{0.1, 0.9},   // G=0 (A)
			{0.4, 0.6},   // G=1 (B)
			{0.99, 0.01}, // G=2 (C)
		},
		[]string{"Grade"},
		map[string]int{"Grade": 3},
	)

	bn.AddCPD(cpdD)
	bn.AddCPD(cpdI)
	bn.AddCPD(cpdG)
	bn.AddCPD(cpdS)
	bn.AddCPD(cpdL)

	return bn, nil
}

// GetAlarmModel creates a simplified version of the ALARM network
// Variables: Burglary, Earthquake, Alarm, JohnCalls, MaryCalls
func GetAlarmModel() (*models.BayesianNetwork, error) {
	edges := [][2]string{
		{"Burglary", "Alarm"},
		{"Earthquake", "Alarm"},
		{"Alarm", "JohnCalls"},
		{"Alarm", "MaryCalls"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		return nil, err
	}

	// P(Burglary) - 0: no, 1: yes
	cpdB, _ := factors.NewTabularCPD("Burglary", 2,
		[][]float64{
			{0.999, 0.001},
		},
		[]string{},
		map[string]int{},
	)

	// P(Earthquake) - 0: no, 1: yes
	cpdE, _ := factors.NewTabularCPD("Earthquake", 2,
		[][]float64{
			{0.998, 0.002},
		},
		[]string{},
		map[string]int{},
	)

	// P(Alarm | Burglary, Earthquake) - 0: no, 1: yes
	cpdA, _ := factors.NewTabularCPD("Alarm", 2,
		[][]float64{
			{0.999, 0.001}, // B=0, E=0
			{0.71, 0.29},   // B=0, E=1
			{0.06, 0.94},   // B=1, E=0
			{0.05, 0.95},   // B=1, E=1
		},
		[]string{"Burglary", "Earthquake"},
		map[string]int{"Burglary": 2, "Earthquake": 2},
	)

	// P(JohnCalls | Alarm) - 0: no, 1: yes
	cpdJ, _ := factors.NewTabularCPD("JohnCalls", 2,
		[][]float64{
			{0.95, 0.05}, // A=0
			{0.1, 0.9},   // A=1
		},
		[]string{"Alarm"},
		map[string]int{"Alarm": 2},
	)

	// P(MaryCalls | Alarm) - 0: no, 1: yes
	cpdM, _ := factors.NewTabularCPD("MaryCalls", 2,
		[][]float64{
			{0.99, 0.01}, // A=0
			{0.3, 0.7},   // A=1
		},
		[]string{"Alarm"},
		map[string]int{"Alarm": 2},
	)

	bn.AddCPD(cpdB)
	bn.AddCPD(cpdE)
	bn.AddCPD(cpdA)
	bn.AddCPD(cpdJ)
	bn.AddCPD(cpdM)

	return bn, nil
}

// GetCancerModel creates a simple cancer diagnosis network
// Variables: Pollution, Smoker, Cancer, XRay, Dyspnoea
func GetCancerModel() (*models.BayesianNetwork, error) {
	edges := [][2]string{
		{"Pollution", "Cancer"},
		{"Smoker", "Cancer"},
		{"Cancer", "XRay"},
		{"Cancer", "Dyspnoea"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		return nil, err
	}

	// P(Pollution) - 0: low, 1: high
	cpdP, _ := factors.NewTabularCPD("Pollution", 2,
		[][]float64{
			{0.9, 0.1},
		},
		[]string{},
		map[string]int{},
	)

	// P(Smoker) - 0: no, 1: yes
	cpdS, _ := factors.NewTabularCPD("Smoker", 2,
		[][]float64{
			{0.7, 0.3},
		},
		[]string{},
		map[string]int{},
	)

	// P(Cancer | Pollution, Smoker) - 0: no, 1: yes
	cpdC, _ := factors.NewTabularCPD("Cancer", 2,
		[][]float64{
			{0.99, 0.01}, // P=0, S=0
			{0.97, 0.03}, // P=0, S=1
			{0.98, 0.02}, // P=1, S=0
			{0.95, 0.05}, // P=1, S=1
		},
		[]string{"Pollution", "Smoker"},
		map[string]int{"Pollution": 2, "Smoker": 2},
	)

	// P(XRay | Cancer) - 0: negative, 1: positive
	cpdX, _ := factors.NewTabularCPD("XRay", 2,
		[][]float64{
			{0.9, 0.1}, // C=0
			{0.2, 0.8}, // C=1
		},
		[]string{"Cancer"},
		map[string]int{"Cancer": 2},
	)

	// P(Dyspnoea | Cancer) - 0: no, 1: yes
	cpdD, _ := factors.NewTabularCPD("Dyspnoea", 2,
		[][]float64{
			{0.7, 0.3},   // C=0
			{0.35, 0.65}, // C=1
		},
		[]string{"Cancer"},
		map[string]int{"Cancer": 2},
	)

	bn.AddCPD(cpdP)
	bn.AddCPD(cpdS)
	bn.AddCPD(cpdC)
	bn.AddCPD(cpdX)
	bn.AddCPD(cpdD)

	return bn, nil
}

// GetSprinklerModel creates the classic Sprinkler/Rain network
// Variables: Cloudy, Sprinkler, Rain, WetGrass
func GetSprinklerModel() (*models.BayesianNetwork, error) {
	edges := [][2]string{
		{"Cloudy", "Sprinkler"},
		{"Cloudy", "Rain"},
		{"Sprinkler", "WetGrass"},
		{"Rain", "WetGrass"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		return nil, err
	}

	// P(Cloudy) - 0: no, 1: yes
	cpdC, _ := factors.NewTabularCPD("Cloudy", 2,
		[][]float64{
			{0.5, 0.5},
		},
		[]string{},
		map[string]int{},
	)

	// P(Sprinkler | Cloudy) - 0: off, 1: on
	cpdS, _ := factors.NewTabularCPD("Sprinkler", 2,
		[][]float64{
			{0.5, 0.5}, // C=0
			{0.9, 0.1}, // C=1
		},
		[]string{"Cloudy"},
		map[string]int{"Cloudy": 2},
	)

	// P(Rain | Cloudy) - 0: no, 1: yes
	cpdR, _ := factors.NewTabularCPD("Rain", 2,
		[][]float64{
			{0.8, 0.2}, // C=0
			{0.2, 0.8}, // C=1
		},
		[]string{"Cloudy"},
		map[string]int{"Cloudy": 2},
	)

	// P(WetGrass | Sprinkler, Rain) - 0: no, 1: yes
	cpdW, _ := factors.NewTabularCPD("WetGrass", 2,
		[][]float64{
			{1.0, 0.0},   // S=0, R=0
			{0.1, 0.9},   // S=0, R=1
			{0.1, 0.9},   // S=1, R=0
			{0.01, 0.99}, // S=1, R=1
		},
		[]string{"Sprinkler", "Rain"},
		map[string]int{"Sprinkler": 2, "Rain": 2},
	)

	bn.AddCPD(cpdC)
	bn.AddCPD(cpdS)
	bn.AddCPD(cpdR)
	bn.AddCPD(cpdW)

	return bn, nil
}
