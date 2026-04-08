.PHONY: install data models eval full

install:
	pip install -r requirements.txt

data:
	python scripts/run_data_pipeline.py

models:
	python scripts/run_models.py

eval:
	python scripts/run_evaluation.py

full:
	bash scripts/run_full_pipeline.sh
