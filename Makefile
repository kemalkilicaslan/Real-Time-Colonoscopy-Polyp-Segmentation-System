.PHONY: install test lint reproduce

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

reproduce:
	bash scripts/reproduce.sh
