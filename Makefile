SOURCE_DIR = src

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
	@poetry run flake8 .
	@poetry run mypy .

.PHONY: test
test:
	@PYTHONPATH=$(SOURCE_DIR) poetry run pytest --cov=$(SOURCE_DIR) tests/
