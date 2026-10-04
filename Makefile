.PHONY: install lint fmt test e2e demo gen clean

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
	uv run python -m jev.cli demo --task banking77 --limit 100 --show 3

gen:
	uv run python -m jev.cli gen-schema --out schema/jev.schema.json

clean:
	rm -rf .pytest_cache .ruff_cache out
