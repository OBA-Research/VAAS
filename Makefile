all: format lint test

format:
	uv run black .
	uv run isort .

lint:
	uv run ruff check .

test:
	uv run pytest -q

build:
	uv build --clear

test-build: build
	rm -rf .venv_test
	uv venv .venv_test
	. .venv_test/bin/activate && uv pip install dist/*.whl && python -c "import vaas; print('Imported VAAS version:', vaas.__version__)" && deactivate
	rm -rf .venv_test
