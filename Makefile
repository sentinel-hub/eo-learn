# Makefile for creating a new release of the package and uploading it to PyPI

PYTHON = python3
PACKAGES = core coregistration features geometry io mask ml_tools

.PHONY: $(PACKAGES:test)

help:
	@echo "Use 'make upload-<package>' to upload the package to PyPi"

.ONESHELL:
build-core:
	cd core
	rm -r dist
	$(PYTHON) setup.py sdist

.ONESHELL:
build-coregistration:
	cd coregistration
	rm -r dist
	$(PYTHON) setup.py sdist

.ONESHELL:
build-features:
	cd features
	rm -r dist
	$(PYTHON) setup.py sdist

.ONESHELL:
build-geometry:
	cd geometry
	rm -r dist
	$(PYTHON) setup.py sdist

.ONESHELL:
build-io:
	cd io
	rm -r dist
	$(PYTHON) setup.py sdist

.ONESHELL:
build-mask:
	cd mask
	rm -r dist
	$(PYTHON) setup.py sdist

.ONESHELL:
build-ml-tools:
	cd ml_tools
	rm -r dist
	$(PYTHON) setup.py sdist

.ONESHELL:
build-abstract-package:
	rm -r dist
	$(PYTHON) setup.py sdist

upload-core: build-core
	twine upload core/dist/*

upload-coregistration: build-coregistration
	twine upload coregistration/dist/*

upload-features: build-features
	twine upload features/dist/*

upload-geometry: build-geometry
	twine upload geometry/dist/*

upload-io: build-io
	twine upload io/dist/*

upload-mask: build-mask
	twine upload mask/dist/*

upload-ml-tools: build-ml-tools
	twine upload ml_tools/dist/*

upload-abstract-package: build-abstract-package
	twine upload dist/*

# For testing:

test-upload-core: build-core
	twine upload --repository testpypi core/dist/*

test-upload-coregistration: build-coregistration
	twine upload --repository testpypi coregistration/dist/*

test-upload-features: build-features
	twine upload --repository testpypi features/dist/*

test-upload-geometry: build-geometry
	twine upload --repository testpypi geometry/dist/*

test-upload-io: build-io
	twine upload --repository testpypi io/dist/*

test-upload-mask: build-mask
	twine upload --repository testpypi mask/dist/*

test-upload-ml-tools: build-ml-tools
	twine upload --repository testpypi ml_tools/dist/*

test-upload-abstract-package: build-abstract-package
	twine upload --repository testpypi dist/*
