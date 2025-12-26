all: lint test-smoke


# Linting & formatting
lint:
	uv run ruff check .

format:
	uv run ruff format .


# Build package
build:
	uv build --clear


# CI-safe smoke tests
test-smoke:
	rm -rf .venv_smoke
	uv venv .venv_smoke
	. .venv_smoke/bin/activate && \
		uv sync --active && \
		uv pip install -e . && \
		uv run --active pytest -m smoke -q ; \
	deactivate
	rm -rf .venv_smoke


# Local integration tests (with torch)
test-integration:
	rm -rf .venv_test
	uv venv .venv_test
	. .venv_test/bin/activate && \
		uv sync && \
		uv sync --group torch && \
		uv sync --group dev && \
		uv pip install dist/*.whl pytest && \
		uv run --active pytest -m integration -v ; \
	deactivate
	rm -rf .venv_test


# Wheel sanity check
test-build: build
	rm -rf .venv_test
	uv venv .venv_test
	. .venv_test/bin/activate && \
		uv pip install dist/*.whl && \
		python -c "import vaas; print('Imported VAAS version:', vaas.__version__)" ; \
	deactivate
	rm -rf .venv_test
