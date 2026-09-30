run:
	uv run python3 -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -name ".DS_Store" -delete

debug:
	pdb