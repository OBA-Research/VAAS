# Contributing to VAAS

Thank you for your interest in improving **VAAS (Vision-Attention Anomaly Scoring)**.
This document defines the **rules and expectations** for contributing to the VAAS codebase.

---

## Important Design Constraints (Read First)

- **PyTorch must NOT be imported at import-time of public APIs**
  - Public entry points (e.g. `vaas.inference.pipeline` and other modules imported by users)
  must remain importable without torch.
- **PyTorch may be imported freely inside internal logic**
  - Internal functions, methods, and execution paths may require torch.
- Lazy-loading is a **hard requirement**, not a preference.
- Any change that modifies the public API or inference behavior must include
at least one passing test demonstrating correctness.
- Changes and PR should be CI-safe. If a change breaks CI-safe imports, it will not be accepted.

---

## 1. Development Environment

VAAS uses **uv** for environment and dependency management.

Create and activate the development environment:

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

Verify installation:

```bash
python -c "import vaas; print(vaas.__version__)"
```

---

## 2. Common Project Tasks

Run the full local workflow:

```bash
make
```

Individual commands:

```bash
make format        # code formatting (ruff)
make lint          # linting (ruff)
make test-smoke    # CI-safe tests (no torch)
make test-integration  # local tests with torch
make build         # build wheel + sdist
```

---

## 3. Code Style & Quality

Formatting and linting are enforced with **ruff**.

Before submitting any change, ensure:

```bash
make format
make lint
```

Pre-commit hooks are required:

```bash
pre-commit install
pre-commit run --all-files
```

---

## 4. Testing Requirements

All contributions must pass:

```bash
make test-smoke
```

If your change affects inference behavior, also run:

```bash
make test-integration
```

New features should include corresponding tests where applicable.

---

## 5. Pull Request Guidelines

Before opening a PR:

1. Tests pass locally
2. Code is formatted and linted
3. Changes are focused and minimal
4. Documentation is updated if behavior changes

Open pull requests **against the `main` branch**.

---

## 6. Issue Reporting

When reporting a bug or issue, please include:

- Python version
- Operating system
- VAAS version
- Installation method (pip / editable / wheel)
- Steps to reproduce
- Full error traceback if available

Incomplete reports may be closed.

---

## 7. Example Notebooks (Optional Contributions)

Notebook contributions are welcome and encouraged mostly around how you are using VAAS in your projects.

**Towards notebook PR:**

- Place under `examples/notebooks/<vaas version>`
- Must run top-to-bottom without manual edits
- No training code
- Lightweight runtime
- Prefer public or sample images
- Colab-friendly where possible

---

## Final Note

VAAS prioritizes **stability, clarity, and reproducibility** over rapid feature growth.
Thank you for contributing.
