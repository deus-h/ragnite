.PHONY: setup-conda setup-micromamba setup-env setup-dev-env install-poetry install-deps dev-env stop-dev-env validate test run-basic-rag install-ollama-model docs clean help

# Default target
all: setup-env

#-------------------
# Environment Setup
#-------------------

# Setup Conda environment
setup-conda:
	@echo "Setting up Conda environment..."
	conda env create -f environment.yml
	@echo "Conda environment 'ragnite' created. Activate with: conda activate ragnite"

# Setup Micromamba environment (faster alternative to Conda)
setup-micromamba:
	@echo "Setting up Micromamba environment..."
	micromamba create -n ragnite -f environment.yml
	@echo "Micromamba environment 'ragnite' created. Activate with: micromamba activate ragnite"

# Setup environment (.env file)
setup-env:
	@echo "Setting up environment variables..."
	cp -n .env.example .env || true
	@echo "Checking for API keys or Ollama..."
	python utils/setup_env_check.py
	@echo "Environment setup complete. Check the .env file if you need to customize settings."

# Install Poetry
install-poetry:
	@echo "Installing Poetry..."
	pip install poetry
	@echo "Poetry installed"

# Install dependencies with Poetry
install-deps:
	@echo "Installing dependencies with Poetry..."
	poetry install
	@echo "Dependencies installed"

# Set up development environment with both Conda and Poetry
setup-dev-env: setup-conda setup-env install-poetry install-deps
	@echo "Development environment fully set up!"

#-------------------
# Docker Management
#-------------------
# Start development environment
dev-env:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d

# Stop development environment
stop-dev-env:
	@echo "Stopping development environment..."
	docker-compose -f docker-compose.dev.yml down

#-------------------
# Testing & Validation
#-------------------
# Validate environment setup
validate:
	@echo "Validating environment setup..."
	python utils/setup_test_env.py

# Run tests
test:
	@echo "Running tests..."
	pytest -xvs

# Run basic RAG example
run-basic-rag:
	@echo "Running basic RAG example..."
	python examples/cached_rag_example.py

#-------------------
# Model Management
#-------------------
# Install Ollama model
install-ollama-model:
	@echo "Installing Ollama model..."
	ollama pull llama3

#-------------------
# Documentation
#-------------------
# Create documentation
docs:
	@echo "Generating documentation..."
	@echo "Not implemented yet"

#-------------------
# Dependency Management
#-------------------
# Add a dependency with Poetry
add-dep:
	@echo "Adding dependency: $(pkg)"
	poetry add $(pkg)

# Add a dev dependency with Poetry
add-dev-dep:
	@echo "Adding dev dependency: $(pkg)"
	poetry add --group dev $(pkg)

# Export dependencies
export-deps:
	@echo "Exporting dependencies..."
	poetry export -f requirements.txt > requirements.txt
	@echo "Dependencies exported to requirements.txt"

#-------------------
# Maintenance
#-------------------
# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name ".cache" -exec rm -rf {} +
	rm -rf dist/ build/ .eggs/

# Help text
help:
	@echo "RAGNITE Makefile"
	@echo "----------------"
	@echo "Environment:"
	@echo "  setup-conda      - Create Conda environment from environment.yml"
	@echo "  setup-micromamba - Create Micromamba environment (faster alternative)"
	@echo "  setup-env        - Create .env file from template"
	@echo "  install-poetry   - Install Poetry package manager"
	@echo "  install-deps     - Install dependencies with Poetry"
	@echo "  setup-dev-env    - Full setup: Conda + .env + Poetry + dependencies"
	@echo ""
	@echo "Docker:"
	@echo "  dev-env          - Start development environment with Docker"
	@echo "  stop-dev-env     - Stop development environment"
	@echo ""
	@echo "Testing:"
	@echo "  validate         - Validate environment setup"
	@echo "  test             - Run tests"
	@echo "  run-basic-rag    - Run basic RAG example"
	@echo ""
	@echo "Dependencies:"
	@echo "  add-dep          - Add dependency with Poetry (usage: make add-dep pkg=numpy)"
	@echo "  add-dev-dep      - Add dev dependency with Poetry"
	@echo "  export-deps      - Export dependencies to requirements.txt"
	@echo ""
	@echo "Models:"
	@echo "  install-ollama-model - Install Ollama model"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             - Generate documentation"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean            - Clean up temporary files" 