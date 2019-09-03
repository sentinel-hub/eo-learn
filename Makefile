# Makefile for creating a new release of the package and uploading it to PyPI

PYTHON = python3
PACKAGES = core coregistration features geometry io mask ml_tools
PYLINT = pylint

.PHONY: $(PACKAGES:test)

help:
	@echo "Use 'make upload-<package>' to upload the package to PyPi"
	@echo "Use 'make pylint' to run pylint on the code of all subpackages"

pylint:
	$(PYLINT) core/eolearn/core/*.py
	$(PYLINT) coregistration/eolearn/coregistration/*.py
	$(PYLINT) features/eolearn/features/*.py
	$(PYLINT) geometry/eolearn/geometry/*.py
	$(PYLINT) io/eolearn/io/*.py
	$(PYLINT) mask/eolearn/mask/*.py
	$(PYLINT) ml_tools/eolearn/ml_tools/*.py
	$(PYLINT) visualization/eolearn/visualization/*.py

.ONESHELL:
build-core:
	cd core
	cp ../LICENSE LICENSE
	rm -r dist
	$(PYTHON) setup.py sdist
	rm LICENSE

.ONESHELL:
build-coregistration:
	cd coregistration
	cp ../LICENSE LICENSE
	rm -r dist
	$(PYTHON) setup.py sdist
	rm LICENSE

.ONESHELL:
build-features:
	cd features
	cp ../LICENSE LICENSE
	rm -r dist
	$(PYTHON) setup.py sdist
	rm LICENSE

.ONESHELL:
build-geometry:
	cd geometry
	cp ../LICENSE LICENSE
	rm -r dist
	$(PYTHON) setup.py sdist
	rm LICENSE

.ONESHELL:
build-io:
	cd io
	cp ../LICENSE LICENSE
	rm -r dist
	$(PYTHON) setup.py sdist
	rm LICENSE

.ONESHELL:
build-mask:
	cd mask
	cp ../LICENSE LICENSE
	rm -r dist
	$(PYTHON) setup.py sdist
	rm LICENSE

.ONESHELL:
build-ml-tools:
	cd ml_tools
	cp ../LICENSE LICENSE
	rm -r dist
	$(PYTHON) setup.py sdist
	rm LICENSE

.ONESHELL:
build-visualization:
	cd visualization
	cp ../LICENSE LICENSE
	rm -r dist
	$(PYTHON) setup.py sdist
	rm LICENSE

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

upload-visualization: build-visualization
	twine upload visualization/dist/*

upload-abstract-package: build-abstract-package
	twine upload dist/*

upload-all: \
 	upload-core \
	upload-coregistration \
	upload-features \
	upload-geometry \
	upload-io \
	upload-mask \
	upload-ml-tools \
	upload-visualization \
	upload-abstract-package

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

test-upload-visualization: build-visualization
	twine upload --repository testpypi visualization/dist/*

test-upload-abstract-package: build-abstract-package
	twine upload --repository testpypi dist/*

test-upload-all: \
 	test-upload-core \
	test-upload-coregistration \
	test-upload-features \
	test-upload-geometry \
	test-upload-io \
	test-upload-mask \
	test-upload-ml-tools \
	test-upload-visualization \
	test-upload-abstract-package
