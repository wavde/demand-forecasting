.PHONY: install data run test lint clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install ruff pytest

data:
	python -c "from demand_forecasting import data; data.get_daily_demand(rebuild=True)"

run:
	python run.py

test:
	pytest -q

lint:
	ruff check .

clean:
	rm -rf data/raw data/processed docs/*.png reports/*.json reports/*.md
