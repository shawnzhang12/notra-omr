.PHONY: lint fmt typecheck test check precommit

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy .

test:
	uv run pytest

check: lint typecheck test

precommit:
	uv run pre-commit run --all-files
