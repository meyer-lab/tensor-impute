.PHONY: clean test

all: test

test: .venv
	rye run pytest -s -v -x

.venv: pyproject.toml
	rye sync

coverage.xml:
	rye run pytest --junitxml=junit.xml --cov=timpute --cov-report xml:coverage.xml

clean:
	rm -rf coverage.xml
