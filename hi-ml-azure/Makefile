# call make for parent
define call_parent
	cd .. && $(MAKE) $(1)
endef

## Package management

# pip upgrade
pip_upgrade:
	$(call call_parent,pip_upgrade)

# pip upgrade and install build requirements
pip_build:
	$(call call_parent,pip_build)

# pip upgrade and install test requirements
pip_test:
	$(call call_parent,pip_test)

# pip install local package in editable mode for development and testing
call_pip_local:
	pip install -e .

# pip upgrade and install local package in editable mode
pip_local: pip_upgrade call_pip_local

# pip install everything for local development and testing
pip: pip_build pip_test call_pip_local

## Actions

# clean build artifacts
clean:
	for folder in .mypy_cache __pycache__ logs outputs; do \
		rm -rf `find . -type d -name $$folder`; \
	done
	rm -rf ./.pytest_cache ./build ./dist ./htmlcov ./src/*.egg-info
	rm -rf ./testazure/testazure/test_data/test_make_tests
	rm -f ./coverage ./coverage.txt ./coverage.xml ./latest_version.txt ./most_recent_run.txt ./package_name.txt

# build package, assuming build requirements already installed
call_build:
	python setup.py sdist bdist_wheel

# pip install build requirements and build package
build: pip_build call_build

# run flake8, assuming test requirements already installed
flake8:
	flake8 --count --statistics .

# run mypy, assuming test requirements already installed
mypy:
	mypy --install-types --show-error-codes --non-interactive setup.py
	mypy --install-types --show-error-codes --non-interactive --package health_azure
	mypy --install-types --show-error-codes --non-interactive --package testazure

# run black check
black:
	black . --check --diff

# run basic checks, assuming test requirements already installed
check: flake8 mypy black

# run pytest on package, assuming test requirements already installed
pytest:
	pytest testazure

# run pytest fast subset on package, assuming test requirements already installed
pytest_fast:
	pytest -m fast testazure

# run pytest with coverage on package
# Output the slowest tests via the --durations flag.
# For diagnostics, use this extra flag to output all captured stdout:
# pytest -rP testazure/testazure/test_azure_util.py::test_download_run_file_during_run
call_pytest_and_coverage:
	pytest --durations=50 --cov=health_azure  --cov-branch --cov-report=html --cov-report=xml --cov-report=term-missing --cov-config=.coveragerc testazure

# install test requirements and run pytest coverage
pytest_and_coverage: pip_test call_pytest_and_coverage

# install test requirements and run all tests
test_all: pip_test call_flake8 call_mypy call_pytest_and_coverage

# run example
example: pip_local
	echo 'edit src/health/azure/examples/elevate_this.py to reference your compute_cluster_name'
	cd src/health/azure/examples; python elevate_this.py --azureml --message 'running example from makefile'

# Create a local Conda environment
env:
	conda env create --file environment.yml

# Install Conda from scratch
conda:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh -b
	rm Miniconda3-latest-Linux-x86_64.sh
	conda update -y -n base conda
	conda install -y -n base conda-libmamba-solver
	conda config --set solver libmamba

env_hello_world_lock:
	../create_and_lock_environment.sh primary_deps_hello_world.yml environment_hello_world.yml

env_hello_world:
	conda env create --file environment_hello_world.yml
