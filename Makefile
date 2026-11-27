.PHONY: install install-pretrained data run pretrained test lint clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install ruff pytest

install-pretrained:
	pip install -r requirements-pretrained.txt

data:
	python -c "from demand_forecasting import data; data.get_daily_demand(rebuild=True)"

run:
	python run.py

pretrained:
	python run_pretrained.py

test:
	pytest -q

lint:
	ruff check .

clean:
	rm -rf data/raw data/processed docs/*.png reports/*.json reports/*.md docs/pretrained_results.md
