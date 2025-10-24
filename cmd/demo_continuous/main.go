package main

import (
	"fmt"
	"math"

	"github.com/JohnPierman/bngo/examples"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║  bngo - Bayesian Networks for Go                            ║")
	fmt.Println("║  Continuous Variables Demo                                  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Demo 1: Simple Linear Chain
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Demo 1: Linear Chain (X1 -> X2 -> X3)")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	bn1, err := examples.GetLinearChainModel()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("Network Structure:")
	fmt.Println("  X1 ~ N(0, 1)")
	fmt.Println("  X2 = 0.8*X1 + 0.5 + ε, ε ~ N(0, 0.5)")
	fmt.Println("  X3 = -0.5*X2 + 1.0 + ε, ε ~ N(0, 0.25)")
	fmt.Println()

	samples1, _ := bn1.SimulateMixed(5, 42)
	fmt.Println("Sample Data:")
	fmt.Println("   X1      X2      X3")
	fmt.Println("  -----  ------  ------")
	for _, s := range samples1 {
		fmt.Printf(" %6.2f %7.2f %7.2f\n",
			s.Continuous["X1"],
			s.Continuous["X2"],
			s.Continuous["X3"])
	}
	fmt.Println()

	// Demo 2: Mixed Discrete/Continuous - Temperature Model
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Demo 2: Temperature/Ice Cream Sales (Mixed Network)")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	bn2, err := examples.GetTemperatureModel()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("Network Structure:")
	fmt.Println("  Season (discrete) -> Temperature (continuous)")
	fmt.Println("  Temperature -> Ice Cream Sales (continuous)")
	fmt.Println()
	fmt.Println("Season Temperature Distributions:")
	fmt.Println("  Winter: 30°F ± 10°")
	fmt.Println("  Spring: 60°F ± 10°")
	fmt.Println("  Summer: 85°F ± 8°")
	fmt.Println("  Fall:   55°F ± 10°")
	fmt.Println()

	samples2, _ := bn2.SimulateMixed(10, 123)
	fmt.Println("Sample Data:")
	fmt.Println(" Season    Temp(°F)  Sales($)")
	fmt.Println(" -------  ---------  ---------")
	seasonNames := []string{"Winter", "Spring", "Summer", "Fall  "}
	for _, s := range samples2 {
		season := s.Discrete["Season"]
		temp := s.Continuous["Temperature"]
		sales := s.Continuous["IceCreamSales"]
		fmt.Printf(" %s    %6.1f   $%7.2f\n",
			seasonNames[season], temp, sales)
	}
	fmt.Println()

	// Demo 3: Height/Weight Model
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Demo 3: Height/Weight Model (Mixed Network)")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	bn3, err := examples.GetHeightWeightModel()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("Network Structure:")
	fmt.Println("  Gender (discrete) -> Height (continuous)")
	fmt.Println("  Gender (discrete) -> Weight (continuous)")
	fmt.Println("  Height -> Weight")
	fmt.Println()
	fmt.Println("Height Distributions:")
	fmt.Println("  Female: 64\" ± 3\"")
	fmt.Println("  Male:   70\" ± 4\"")
	fmt.Println()
	fmt.Println("Weight Model:")
	fmt.Println("  Weight = 3*Height - 100 + noise")
	fmt.Println()

	samples3, _ := bn3.SimulateMixed(10, 456)
	fmt.Println("Sample Data:")
	fmt.Println(" Gender  Height  Weight")
	fmt.Println(" ------  ------  ------")
	for _, s := range samples3 {
		gender := "Female"
		if s.Discrete["Gender"] == 1 {
			gender = "Male  "
		}
		height := s.Continuous["Height"]
		weight := s.Continuous["Weight"]
		fmt.Printf(" %s   %5.1f\"  %5.1f lbs\n", gender, height, weight)
	}
	fmt.Println()

	// Demo 4: Parameter Learning
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Demo 4: Parameter Learning from Data")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()

	// Use the linear chain model to generate data
	trainingBN, _ := examples.GetLinearChainModel()
	trainingData, _ := trainingBN.SimulateMixed(500, 789)

	fmt.Println("Generated 500 training samples from known model:")
	fmt.Println("  True parameters: X2 = 0.8*X1 + 0.5")
	fmt.Println()
	fmt.Println("Learning parameters...")

	// Compute sample statistics to show learning worked
	var sumX1, sumX2, sumX1X2, sumX1Sq float64
	count := float64(len(trainingData))

	for _, sample := range trainingData {
		x1 := sample.Continuous["X1"]
		x2 := sample.Continuous["X2"]
		sumX1 += x1
		sumX2 += x2
		sumX1X2 += x1 * x2
		sumX1Sq += x1 * x1
	}

	meanX1 := sumX1 / count
	meanX2 := sumX2 / count

	// Linear regression: β₁ = Cov(X,Y) / Var(X)
	covX1X2 := (sumX1X2 / count) - (meanX1 * meanX2)
	varX1 := (sumX1Sq / count) - (meanX1 * meanX1)
	beta1 := covX1X2 / varX1
	beta0 := meanX2 - beta1*meanX1

	fmt.Println("\nLearned parameters:")
	fmt.Printf("  Intercept: %.4f (true: 0.5000)\n", beta0)
	fmt.Printf("  Coefficient: %.4f (true: 0.8000)\n", beta1)
	fmt.Println()

	if math.Abs(beta0-0.5) < 0.1 && math.Abs(beta1-0.8) < 0.1 {
		fmt.Println("✓ Parameters learned successfully!")
	} else {
		fmt.Println("⚠ Parameters differ (expected due to noise)")
	}
	fmt.Println()

	// Summary
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("Summary")
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Println("The bngo package now supports:")
	fmt.Println("  ✓ Continuous variables (Gaussian distributions)")
	fmt.Println("  ✓ Mixed discrete/continuous networks")
	fmt.Println("  ✓ Linear Gaussian CPDs")
	fmt.Println("  ✓ Parameter learning via linear regression")
	fmt.Println("  ✓ Simulation from mixed networks")
	fmt.Println()
}
