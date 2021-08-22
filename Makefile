FLAKE8_CONFIG_FILE = .flake8

.PHONY: dep
dep:
	@poetry install --no-root

.PHONY: format
format:
	@poetry run black .
	@poetry run isort .

.PHONY: lint
lint:
	@poetry run black --check .
	@poetry run flake8 --config $(FLAKE8_CONFIG_FILE) .
	@poetry run mypy .

.PHONY: test
test:
	@PYTHONPATH=src poetry run pytest --cov=src tests/
