# data/

This directory is intentionally (almost) empty. Raw and processed data are **not**
committed — they are downloaded and built at runtime.

Running `python run.py` (or `make data`) will:

1. Download **UCI Online Retail II** from the UCI ML repository into `data/raw/`
   (public, no authentication required).
2. Clean and aggregate it into a cached daily series at
   `data/processed/daily_demand.csv`.

Everything here except this README and `.gitkeep` is gitignored.
