PY=python3
CONFIG?=config.txt
VENV?=.venv
PIP=$(VENV)/bin/pip
FLAKE8=$(VENV)/bin/flake8
MYPY=$(VENV)/bin/mypy

.PHONY: install run debug clean lint lint-strict

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
	$(FLAKE8) .
	$(MYPY) . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

