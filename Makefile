# Makefile for creating a new release of the package and uploading it to PyPI

PYTHON = python3

.PHONY: $(PACKAGES:test)

help:
	@echo "Use 'make test-upload' to upload the package to testPyPi"
	@echo "Use 'make upload' to upload the package to PyPi"


upload:
	rm -r dist | true
	python -m build --sdist --wheel
	twine upload --skip-existing dist/*

# For testing:
test-upload:
	rm -r dist | true
	python -m build --sdist --wheel
	twine upload --repository testpypi --skip-existing dist/*
