package examples

import (
	"fmt"

	"github.com/JohnPierman/bngo/factors"
	"github.com/JohnPierman/bngo/models"
)

// GetHeightWeightModel returns a Bayesian Network modeling the relationship
// between gender (discrete), height (continuous), and weight (continuous)
//
// Network structure:
//
//	Gender -> Height
//	Gender -> Weight
//	Height -> Weight
//
// Gender: 0 = Female, 1 = Male
// Height: in inches, modeled as Gaussian
// Weight: in pounds, modeled as linear function of height
func GetHeightWeightModel() (*models.BayesianNetwork, error) {
	// Define structure
	edges := [][2]string{
		{"Gender", "Height"},
		{"Gender", "Weight"},
		{"Height", "Weight"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		return nil, err
	}

	// CPD for Gender
	// P(Gender=Female) = 0.5, P(Gender=Male) = 0.5
	cpdGender, err := factors.NewTabularCPD("Gender", 2,
		[][]float64{{0.5, 0.5}},
		[]string{},
		map[string]int{},
	)
	if err != nil {
		return nil, err
	}
	bn.AddCPD(cpdGender)

	// CPD for Height given Gender
	// Height | Female ~ N(64, 9)  (mean 64 inches, std 3 inches)
	// Height | Male ~ N(70, 16)   (mean 70 inches, std 4 inches)
	heightStates := map[string]factors.GaussianParams{
		"0": {Mean: 64.0, Variance: 9.0},  // Female
		"1": {Mean: 70.0, Variance: 16.0}, // Male
	}
	cpdHeight, err := factors.NewDiscreteParentGaussianCPD(
		"Height",
		[]string{"Gender"},
		map[string]int{"Gender": 2},
		heightStates,
	)
	if err != nil {
		return nil, err
	}
	if err := bn.AddGaussianCPD(cpdHeight); err != nil {
		return nil, err
	}

	// CPD for Weight given Gender and Height
	// This would ideally depend on both, but for simplicity we'll make it
	// depend primarily on Height with different intercepts for Gender
	// For now, simplified: Weight = 3*Height - 100 + noise
	// In a full implementation, we'd want different parameters per gender
	cpdWeight, err := factors.NewLinearGaussianCPD(
		"Weight",
		[]string{"Height"},
		-100.0,                            // intercept
		map[string]float64{"Height": 3.0}, // 3 lbs per inch
		25.0,                              // variance
	)
	if err != nil {
		return nil, err
	}
	if err := bn.AddGaussianCPD(cpdWeight); err != nil {
		return nil, err
	}

	return bn, nil
}

// GetTemperatureModel returns a simple weather model with continuous variables
//
// Network structure:
//
//	Season -> Temperature
//	Temperature -> IceCreamSales
//
// Season: 0=Winter, 1=Spring, 2=Summer, 3=Fall
// Temperature: in Fahrenheit
// IceCreamSales: daily sales in dollars
func GetTemperatureModel() (*models.BayesianNetwork, error) {
	edges := [][2]string{
		{"Season", "Temperature"},
		{"Temperature", "IceCreamSales"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		return nil, err
	}

	// CPD for Season (uniform)
	cpdSeason, err := factors.NewTabularCPD("Season", 4,
		[][]float64{{0.25, 0.25, 0.25, 0.25}},
		[]string{},
		map[string]int{},
	)
	if err != nil {
		return nil, err
	}
	bn.AddCPD(cpdSeason)

	// CPD for Temperature given Season
	tempStates := map[string]factors.GaussianParams{
		"0": {Mean: 30.0, Variance: 100.0}, // Winter: 30°F ± 10°
		"1": {Mean: 60.0, Variance: 100.0}, // Spring: 60°F ± 10°
		"2": {Mean: 85.0, Variance: 64.0},  // Summer: 85°F ± 8°
		"3": {Mean: 55.0, Variance: 100.0}, // Fall: 55°F ± 10°
	}
	cpdTemp, err := factors.NewDiscreteParentGaussianCPD(
		"Temperature",
		[]string{"Season"},
		map[string]int{"Season": 4},
		tempStates,
	)
	if err != nil {
		return nil, err
	}
	if err := bn.AddGaussianCPD(cpdTemp); err != nil {
		return nil, err
	}

	// CPD for Ice Cream Sales given Temperature
	// Sales = 10*Temperature - 200 + noise
	// (roughly $10 more per degree, starting at $200 base)
	cpdSales, err := factors.NewLinearGaussianCPD(
		"IceCreamSales",
		[]string{"Temperature"},
		-200.0,
		map[string]float64{"Temperature": 10.0},
		2500.0, // high variance
	)
	if err != nil {
		return nil, err
	}
	bn.AddGaussianCPD(cpdSales)

	return bn, nil
}

// GetLinearChainModel returns a simple linear chain of continuous variables
//
// Network: X1 -> X2 -> X3
// Each variable is continuous and linearly related to its parent
func GetLinearChainModel() (*models.BayesianNetwork, error) {
	edges := [][2]string{
		{"X1", "X2"},
		{"X2", "X3"},
	}

	bn, err := models.NewBayesianNetwork(edges)
	if err != nil {
		return nil, err
	}

	// X1 ~ N(0, 1)
	cpd1, err := factors.NewLinearGaussianCPD(
		"X1",
		[]string{},
		0.0,
		map[string]float64{},
		1.0,
	)
	if err != nil {
		return nil, err
	}
	bn.AddGaussianCPD(cpd1)

	// X2 = 0.8*X1 + 0.5 + ε, ε ~ N(0, 0.5)
	cpd2, err := factors.NewLinearGaussianCPD(
		"X2",
		[]string{"X1"},
		0.5,
		map[string]float64{"X1": 0.8},
		0.5,
	)
	if err != nil {
		return nil, err
	}
	bn.AddGaussianCPD(cpd2)

	// X3 = -0.5*X2 + 1.0 + ε, ε ~ N(0, 0.25)
	cpd3, err := factors.NewLinearGaussianCPD(
		"X3",
		[]string{"X2"},
		1.0,
		map[string]float64{"X2": -0.5},
		0.25,
	)
	if err != nil {
		return nil, err
	}
	bn.AddGaussianCPD(cpd3)

	return bn, nil
}

// DemonstrateContinuousNetwork shows how to use continuous Bayesian Networks
func DemonstrateContinuousNetwork() {
	fmt.Println("=== Continuous Bayesian Network Example ===")

	// Create the temperature model
	bn, err := GetTemperatureModel()
	if err != nil {
		fmt.Printf("Error creating model: %v\n", err)
		return
	}

	fmt.Println("Created Temperature/Ice Cream Sales model")
	fmt.Printf("Nodes: %v\n", bn.Nodes())
	fmt.Printf("Edges: %v\n", bn.Edges())
	fmt.Println()

	// Simulate data
	fmt.Println("Simulating 10 samples...")
	samples, err := bn.SimulateMixed(10, 42)
	if err != nil {
		fmt.Printf("Error simulating: %v\n", err)
		return
	}

	fmt.Println("\nSample data:")
	fmt.Println("Season | Temperature | Ice Cream Sales")
	fmt.Println("-------|-------------|----------------")
	seasonNames := []string{"Winter", "Spring", "Summer", "Fall"}
	for i, sample := range samples {
		season := sample.Discrete["Season"]
		temp := sample.Continuous["Temperature"]
		sales := sample.Continuous["IceCreamSales"]
		fmt.Printf("%2d. %-6s | %6.1f°F    | $%.2f\n",
			i+1, seasonNames[season], temp, sales)
	}

	// Demonstrate parameter learning
	fmt.Println()
	fmt.Println("=== Parameter Learning Example ===")

	// Create a new model to learn
	edges := [][2]string{{"X", "Y"}}
	learnBN, _ := models.NewBayesianNetwork(edges)

	// Generate training data: Y = 2*X + 5
	fmt.Println("Generating training data: Y = 2*X + 5")
	trainingData := make([]models.Sample, 100)
	for i := 0; i < 100; i++ {
		x := float64(i-50) / 10.0 // X from -5 to 5
		y := 2*x + 5
		trainingData[i] = models.Sample{
			Discrete:   make(map[string]int),
			Continuous: map[string]float64{"X": x, "Y": y},
		}
	}

	// Learn parameters
	err = learnBN.FitMixed(trainingData)
	if err != nil {
		fmt.Printf("Error learning parameters: %v\n", err)
		return
	}

	// Display learned parameters
	cpdX, _ := learnBN.GetGaussianCPD("X")
	cpdY, _ := learnBN.GetGaussianCPD("Y")

	fmt.Printf("\nLearned parameters for X:\n")
	fmt.Printf("  Mean: %.4f (actual: 0.0)\n", cpdX.Intercept)
	fmt.Printf("  Variance: %.4f\n", cpdX.Variance)

	fmt.Printf("\nLearned parameters for Y|X:\n")
	fmt.Printf("  Intercept: %.4f (actual: 5.0)\n", cpdY.Intercept)
	fmt.Printf("  Coefficient for X: %.4f (actual: 2.0)\n", cpdY.Coefficients["X"])
	fmt.Printf("  Variance: %.4f\n", cpdY.Variance)
}
