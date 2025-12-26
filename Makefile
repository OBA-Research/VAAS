all: format lint test

format:
	uv run black .
	uv run isort .

lint:
	uv run ruff check .

build:
	uv build --clear


test:
	rm -rf .venv_test
	uv venv .venv_test
	uv build --clear
	. .venv_test/bin/activate && \
		uv sync && \
		uv sync --group torch && \
		uv pip install dist/*.whl pytest && \
		uv run --active --group torch pytest -v \
			&& deactivate
		rm -rf .venv_test


test-build: build
	rm -rf .venv_test
	uv venv .venv_test
	. .venv_test/bin/activate && uv pip install dist/*.whl && python -c "import vaas; print('Imported VAAS version:', vaas.__version__)" && deactivate
	rm -rf .venv_test
