.PHONY: all build test test-coverage test-verbose bench clean fmt vet lint install run-demo help

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod
GOFMT=$(GOCMD) fmt
GOVET=$(GOCMD) vet

# Build parameters
BINARY_NAME=bngo-demo
BINARY_PATH=./bin/$(BINARY_NAME)
CMD_PATH=./cmd/demo

# Test parameters
TEST_FLAGS=-v -race
COVERAGE_FILE=coverage.out
COVERAGE_HTML=coverage.html

all: test build

## build: Build the demo application
build:
	@echo "Building..."
	@mkdir -p bin
	@$(GOBUILD) -o $(BINARY_PATH) $(CMD_PATH)
	@echo "Built $(BINARY_PATH)"

## test: Run all tests
test:
	@echo "Running tests..."
	@$(GOTEST) -vet=off ./...

## test-verbose: Run tests with verbose output
test-verbose:
	@echo "Running tests (verbose)..."
	@$(GOTEST) $(TEST_FLAGS) ./...

## test-coverage: Run tests with coverage report
test-coverage:
	@echo "Running tests with coverage..."
	@$(GOTEST) -coverprofile=$(COVERAGE_FILE) -covermode=atomic ./...
	@$(GOCMD) tool cover -html=$(COVERAGE_FILE) -o $(COVERAGE_HTML)
	@echo "Coverage report generated: $(COVERAGE_HTML)"

## bench: Run benchmarks
bench:
	@echo "Running benchmarks..."
	@$(GOTEST) -bench=. -benchmem ./...

## clean: Remove build artifacts and test files
clean:
	@echo "Cleaning..."
	@$(GOCLEAN)
	@rm -rf bin/
	@rm -f $(COVERAGE_FILE) $(COVERAGE_HTML)
	@rm -f *.csv
	@echo "Cleaned"

## fmt: Format all Go files
fmt:
	@echo "Formatting code..."
	@$(GOFMT) ./...

## vet: Run go vet
vet:
	@echo "Running go vet..."
	@$(GOVET) ./...

## lint: Run linter (requires golangci-lint)
lint:
	@echo "Running linter..."
	@which golangci-lint > /dev/null || (echo "golangci-lint not installed. Install from https://golangci-lint.run/usage/install/" && exit 1)
	@golangci-lint run --timeout=5m

## mod-download: Download Go module dependencies
mod-download:
	@echo "Downloading dependencies..."
	@$(GOMOD) download

## mod-tidy: Tidy Go module dependencies
mod-tidy:
	@echo "Tidying dependencies..."
	@$(GOMOD) tidy

## mod-verify: Verify Go module dependencies
mod-verify:
	@echo "Verifying dependencies..."
	@$(GOMOD) verify

## install: Install the demo application
install: build
	@echo "Installing..."
	@cp $(BINARY_PATH) $(GOPATH)/bin/

## run-demo: Run the demo application
run-demo:
	@echo "Running demo..."
	@$(GOCMD) run $(CMD_PATH)

## run: Alias for run-demo
run: run-demo

## check: Run all checks (fmt, vet, lint, test)
check: fmt vet test
	@echo "All checks passed!"

## ci: Run CI checks
ci: mod-tidy fmt vet test-coverage
	@echo "CI checks complete!"

## help: Display this help message
help:
	@echo "bngo Makefile commands:"
	@echo ""
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed -e 's/^/ /'
	@echo ""

