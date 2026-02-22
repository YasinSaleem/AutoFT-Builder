.PHONY: install install-dev sync test coverage lint typecheck format run clean help

# Default arguments for run command
ARGS ?=

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies with uv
	uv sync --no-dev

install-dev:  ## Install all dependencies including dev with uv
	uv sync

sync:  ## Sync dependencies (alias for install-dev)
	uv sync

test:  ## Run tests
	uv run pytest

test-unit:  ## Run unit tests only (exclude integration tests)
	uv run pytest -m "not integration and not real"

test-integration:  ## Run integration tests
	uv run pytest -m "integration"

coverage:  ## Run tests with coverage report
	uv run pytest --cov=autoft --cov-report=html --cov-report=term-missing

lint:  ## Run linter (ruff)
	uv run ruff check src tests

lint-fix:  ## Run linter and auto-fix issues
	uv run ruff check --fix src tests

typecheck:  ## Run type checker (mypy)
	uv run mypy src/autoft

format:  ## Format code (ruff)
	uv run ruff format src tests

format-check:  ## Check code formatting without making changes
	uv run ruff format --check src tests

run:  ## Run the CLI (use ARGS="..." to pass arguments)
	uv run autoft $(ARGS)

clean:  ## Clean build artifacts and caches
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

check: lint typecheck test  ## Run all checks (lint, typecheck, test)

lock:  ## Update uv.lock file
	uv lock

upgrade:  ## Upgrade all dependencies
	uv lock --upgrade
