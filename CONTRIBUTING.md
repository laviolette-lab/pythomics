# Contributing

Thank you for your interest in contributing! This document provides guidelines for working with this project.

## Development Setup

1. **Clone the repository:**

   ```console
   git clone https://github.com/laviolette-lab/taps.git
   cd taps
   ```

2. **Install Hatch** (build & environment manager):

   ```console
   pip install hatch
   ```

3. **Install pre-commit hooks:**

   ```console
   pip install pre-commit
   pre-commit install
   ```

## Common Tasks

All project tasks are managed through [Hatch environments](https://hatch.pypa.io/latest/environment/):

| Task                    | Command                      |
|-------------------------|------------------------------|
| Run tests               | `hatch run test:test`        |
| Run tests with coverage | `hatch run test:cov`         |
| Lint code               | `hatch run lint:check`       |
| Format code             | `hatch run lint:format`      |
| Auto-fix lint issues    | `hatch run lint:fix`         |
| Run all lint checks     | `hatch run lint:all`         |
| Type check              | `hatch run types:check`      |
| Build docs              | `hatch run docs:build-docs`  |
| Serve docs locally      | `hatch run docs:serve-docs`  |
| Build wheel             | `hatch build`                |

## Code Style

- **Formatter & Linter:** [Ruff](https://docs.astral.sh/ruff/) handles both formatting and linting. Configuration lives in `pyproject.toml` under `[tool.ruff]`.
- **Line length:** 88 characters (consistent with Black's default).
- **Docstrings:** Use [Sphinx-style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) docstrings (`:param:`, `:type:`, `:return:`, `:rtype:`).
- **Type hints:** Encouraged. The project includes a `py.typed` marker for PEP 561 compliance.

## Project Architecture

This project follows a **library-first** design:

1. **Library modules** (`src/taps/`) contain all business logic as importable functions and classes.
2. **CLI** (`src/taps/cli.py`) is a thin wrapper that parses arguments and delegates to library functions.
3. **Tests** (`tests/`) import directly from the library — never from the CLI.

When adding new functionality:

- Write the logic as a function or class in a library module.
- Add tests that call the function directly.
- If CLI access is needed, add a subcommand in `cli.py` that wraps the library function.

## Writing Tests

- Tests live in `tests/` and use [pytest](https://docs.pytest.org/).
- Shared fixtures belong in `tests/conftest.py`.
- Name test files `test_<module>.py` and test functions `test_<behavior>`.
- Aim for descriptive test names that explain what is being verified.

```python
def test_resolve_checkpoint_uses_bundled_model_by_default():
    """Verify resolve_checkpoint falls back to the packaged model."""
    assert resolve_checkpoint(None).name == "model_v1.pth"
```

## Submitting Changes

1. **Create a branch** from `main`:

   ```console
   git checkout -b feature/my-change
   ```

2. **Make your changes** and ensure all checks pass:

   ```console
   hatch run lint:all
   hatch run test:cov
   hatch run types:check
   ```

3. **Commit** with a clear, descriptive message. Pre-commit hooks will run Ruff automatically.

4. **Open a Pull Request** against `main`. The CI pipeline will run linting, tests, and build checks.

## Docker

The project includes multi-stage Docker builds:

| Target   | Purpose                                    |
|----------|--------------------------------------------|
| `base`   | Base image with source and non-root user   |
| `hatch`  | Runs Hatch commands (used in CI)           |
| `dev`    | Full development environment               |
| `prod`   | Minimal production image with only the wheel installed |

```console
# Build and run tests via Docker
docker build --target hatch -t myapp:hatch .
docker run --rm -e HATCH_ENV=test taps:hatch cov

# Build production image
docker build --target prod -t taps:prod .
```

## Questions?

Open an issue on GitHub if you have questions or run into problems.