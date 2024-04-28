lint:
	@find . -type d -name "Lab*" -exec sh -c 'pylint "$$0"/*.py' {} \;

format:
	@echo "Running black..."
	@black .
	@echo "Running isort..."
	@isort .
	@echo "Formatting complete!"

.PHONY: lint format
