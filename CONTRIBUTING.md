# Contributing to VAAS

Thank you for your interest in improving **VAAS (Vision-Attention Anomaly Scoring)**.  
This guide explains how to set up the development environment and how to contribute changes.

---

## 1. Development Environment

VAAS uses **uv** for dependency and environment management.

Create and activate the environment:

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .
```

Run project tasks:

```bash
make          # runs format + lint + test
make format
make lint
make test
make build
```

---

## 2. Code Style

Formatting:

```bash
make format
```

Linting:

```bash
make lint
```

All submitted code must pass both.

---

## 3. Tests

Run tests locally:

```bash
make test
```

All new functionality should include or update relevant tests.

---

## 4. Pull Requests

Before opening a PR:

1. Ensure tests pass  
2. Run `make format` and `make lint`  
3. Keep changes focused and well-scoped  
4. Update documentation where needed  

Open the pull request against the **main** branch.

---

## 5. Issues

When reporting an issue, include:

- Python version  
- OS  
- VAAS version  
- Steps to reproduce  
- Relevant logs or error messages  

---

## 6. Example Notebooks (Colab / Jupyter)

We welcome notebook contributions that help users understand and apply VAAS.

Suggested topics:

- Basic inference walkthrough  
- Visualising anomaly maps with Matplotlib  
- Running VAAS on entire folders or batches  
- Exploring α and score behaviour  
- Device configuration (CPU / CUDA / MPS)  
- Using Hugging Face `from_pretrained`  

**Notebook requirements:**

- Place all notebooks under `examples/notebooks/`
- Must run top-to-bottom without manual intervention
- Prefer publicly accessible images / HF dataset samples
- Keep runtime lightweight (no training)
- Colab-friendly where possible

---

Thank you for contributing to VAAS.
