## Package management

# Create a Conda environment for this folder only
env:
	conda env create --file environment.yml
	pip install -r requirements_test.txt
	pip install -r requirements_build.txt

# pip upgrade
pip_upgrade:
	python -m pip install --upgrade pip

# pip upgrade and install build requirements
pip_build: pip_upgrade
	pip install -r requirements_build.txt

# pip upgrade and install test requirements
pip_test: pip_upgrade
	pip install -r requirements_test.txt

pip_publish: pip_upgrade
	pip install --upgrade build twine

# pip install local package in editable mode for development and testing
pip_local:
	pip install -e .

# pip install everything for local development and testing
pip: pip_build pip_test pip_local

## Actions

# clean build artifacts
clean:
	for folder in .mypy_cache __pycache__ logs outputs; do \
		rm -rf `find . -type d -name $$folder`; \
	done
	rm -rf ./.pytest_cache ./build ./dist ./htmlcov ./*.egg-info
	rm -f ./coverage ./coverage.txt ./coverage.xml ./latest_version.txt ./most_recent_run.txt ./package_name.txt

# build package, assuming build requirements already installed
build:
	python setup.py sdist bdist_wheel

# run flake8, assuming test requirements already installed
flake8:
	flake8 --count --statistics .

# run mypy, assuming test requirements already installed
mypy:
	mypy --install-types --show-error-codes --non-interactive setup.py
	mypy --install-types --show-error-codes --non-interactive src
	mypy --install-types --show-error-codes --non-interactive test_multimodal

# run pytest on package, assuming test requirements already installed
pytest:
	pytest test_multimodal

# run pytest with coverage on package
pytest_coverage:
	pytest \
		--cov=health_multimodal \
		--cov-branch \
		--cov-report=html \
		--cov-report=xml \
		--cov-report=term-missing \
		--cov-config=.coveragerc

# Install pyright
pyright_install:
	npm install -g pyright

# conda install test requirements and run pyright
pyright:
	pyright

# run basic checks
check: flake8 mypy pyright

# bump version from specified part (major, minor or patch) (dry run). Usage:
# $ make bump_version_dry part=patch
bump_version_dry:
	bump2version $(part) --verbose --dry-run --allow-dirty

# bump version from specified part (major, minor or patch). Usage:
# $ make bump_version part=patch
bump_version:
	bump2version $(part) --verbose

# build Python package
build: pip_publish
	python -m build

# publish package to TestPyPI
publish_test: build
	python -m twine upload --repository testpypi dist/*

# publish package to PyPI
publish: build
	python -m twine upload dist/*
