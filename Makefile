# Make commands for the toolbox users

# Create a Conda environment for use with both the hi-ml and hi-ml-azure folder
env:
	conda env create --file hi-ml/environment.yml

# Make commands that are used in the build pipeline

# call make for each sub package
define call_packages
	cd hi-ml-azure && $(MAKE) $(1)
	cd hi-ml && $(MAKE) $(1)
endef

## Package management

# pip upgrade
pip_upgrade:
	python -m pip install --upgrade pip

# pip upgrade and install build requirements
pip_build: pip_upgrade
	pip install -r build_requirements.txt

# pip upgrade and install test requirements
pip_test: pip_upgrade
	pip install -r test_requirements.txt

# pip install local packages in editable mode for development and testing
call_pip_local:
	$(call call_packages,call_pip_local)

# pip upgrade and install local packages in editable mode
pip_local: pip_upgrade call_pip_local

# pip install everything for local development and testing
pip: pip_build pip_test call_pip_local


# update current conda environment
conda_update:
	conda env update -n $(CONDA_DEFAULT_ENV) --file hi-ml/environment.yml

# Set the conda environment for local development work, that contains all packages need for hi-ml, hi-ml-azure
# and hi-ml-cpath with hi-ml and hi-ml-azure installed in editable mode
conda: conda_update call_pip_local

## Actions

# clean build artifacts
clean:
	rm -rf ./.mypy_cache ./.pytest_cache ./coverage ./logs ./outputs
	rm -f ./coverage.txt ./coverage.xml ./most_recent_run.txt
	$(call call_packages,clean)

# build package, assuming build requirements already installed
call_build:
	$(call call_packages,call_build)

# pip install build requirements and build package
build: pip_build call_build

# run flake8, assuming test requirements already installed
flake8:
	$(call call_packages,flake8)

# run mypy, assuming test requirements already installed
mypy:
	$(call call_packages,mypy)

# run pyright, assuming test requirements already installed
call_pyright:
	npm install -g pyright
	pyright

# conda install test requirements and run pyright
pyright: conda call_pyright

# run basic checks, assuming test requirements already installed
check: flake8 mypy

# run pytest on package, assuming test requirements already installed
pytest:
	$(call call_packages,pytest)

# run pytest fast subset on package, assuming test requirements already installed
pytest_fast:
	$(call call_packages,pytest_fast)

# run pytest with coverage on package, and format coverage output as a text file, assuming test requirements already installed
call_pytest_and_coverage:
	$(call call_packages,call_pytest_and_coverage)

# install test requirements and run pytest coverage
pytest_and_coverage: pip_test call_pytest_and_coverage

# install test requirements and run all tests
test_all: pip_test call_flake8 call_mypy call_pytest_and_coverage

# build the github format_coverage action
action:
	cd .github/actions/format_coverage && ncc build index.js --license licenses.txt

combine: pip_test
	mkdir -p coverage
	cp hi-ml/.coverage coverage/hi-ml-coverage
	cp hi-ml-azure/.coverage coverage/hi-ml-azure-coverage
	cp hi-ml-cpath/.coverage coverage/hi-ml-cpath-coverage
	cp .coveragerc coverage/
	cd coverage && \
		coverage combine hi-ml-coverage hi-ml-azure-coverage hi-ml-cpath-coverage &&  \
		coverage html && \
		coverage xml && \
		pycobertura show --format text --output coverage.txt coverage.xml
