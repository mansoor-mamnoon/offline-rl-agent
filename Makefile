.PHONY: test lint format train-smoke reproduce-small type-check reproduce-traffic generate-gifs

test:
	.venv/bin/pytest tests/ -x --timeout=60

lint:
	.venv/bin/black --check offline_rl/
	.venv/bin/ruff check offline_rl/

format:
	.venv/bin/black offline_rl/
	.venv/bin/ruff check --fix offline_rl/

train-smoke:
	.venv/bin/python scripts/train.py --algo bc --env traffic --n-dataset-episodes 50 --n-train-epochs 3 --seed 0

reproduce-small:
	.venv/bin/python scripts/reproduce_table.py --env traffic --seeds 0 1 --n-epochs 10

type-check:
	.venv/bin/mypy offline_rl/ --ignore-missing-imports --no-strict-optional 2>&1 | tail -20

reproduce-traffic:
	.venv/bin/python scripts/reproduce_table.py --env traffic --seeds 0 1 2 --n-epochs 30

generate-gifs:
	mkdir -p artifacts/gifs
	.venv/bin/python scripts/generate_gifs.py 2>&1 | tail -20
