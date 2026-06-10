# Python Package Management with uv

Use uv exclusively for Python package management in this project.

## Package Management Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync environment: `uv sync`
- Lock dependencies: `uv lock`

## Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools with `uv run <tool>` (e.g. `uv run pytest`, `uv run ruff`, `uv run mypy`, `uv run pre-commit`)
- Launch a Python REPL with `uv run python`


## Linting and formatting

This project uses Ruff for both linting and formatting. Do not call Black,
flake8, isort, or pylint.

- Lint: `uv run ruff check .`
- Lint and auto-fix: `uv run ruff check --fix .`
- Format: `uv run ruff format .`
- Check formatting without writing: `uv run ruff format --check .`
- Always invoke Ruff through `uv run` so it resolves to the project's
  virtual environment.

Ruff configuration lives in `pyproject.toml` under `[tool.ruff]`. Do not
add a separate `ruff.toml` or `.ruff.toml`. Do not add inline `# noqa`
comments without a rule code.

## Git commit
Create a detailed git commit after accomplishing a unique feature.

## Unit tests
Create a unit tests.

