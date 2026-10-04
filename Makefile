.PHONY: install lint fmt test e2e demo train gen clean

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run black --check src tests

fmt:
	uv run black src tests
	uv run ruff check --fix src tests

test:
	uv run pytest -q

e2e:
	JEV_E2E=1 uv run pytest -q -k e2e

demo:
	uv run python -m jev.cli demo --task intent --limit 100 --show 3

train:
	uv run python -m jev.cli train --limit 100

gen:
	uv run python -m jev.cli gen-schema --out schema/jev.schema.json

clean:
	rm -rf .pytest_cache .ruff_cache out mlruns mlartifacts mlflow.db
