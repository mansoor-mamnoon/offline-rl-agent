.PHONY: test lint format train-smoke reproduce-small

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
