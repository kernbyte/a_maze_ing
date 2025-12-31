PY=python3
CONFIG?=config.txt
VENV?=.venv
PIP=$(VENV)/bin/pip

install:
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

run:
	$(PY) a_maze_ing.py $(CONFIG)

debug:
	$(PY) -m pdb a_maze_ing.py $(CONFIG)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache build dist *.egg-info
	find . -name '*.pyc' -delete

lint:
	PATH="$(VENV)/bin:$$PATH" flake8 .
	PATH="$(VENV)/bin:$$PATH" mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	PATH="$(VENV)/bin:$$PATH" flake8 .
	PATH="$(VENV)/bin:$$PATH" mypy . --strict

.PHONY: install run debug clean lint lint-strict
