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
	rm -rf Lightning checkpoints lightning_logs
	rm -f ./coverage ./coverage.txt ./coverage.xml ./latest_version.txt ./most_recent_run.txt ./package_name.txt

# build package, assuming build requirements already installed
call_build:
	python setup.py sdist bdist_wheel

# pip install build requirements and build package
build: pip_build call_build

# run flake8, assuming test requirements already installed
flake8:
	flake8 --count --statistics

# run mypy, assuming test requirements already installed
mypy:
	mypy --install-types --show-error-codes --non-interactive setup.py
	mypy --install-types --show-error-codes --non-interactive --package health_ml
	mypy --install-types --show-error-codes --non-interactive --package testhiml

# run basic checks, assuming test requirements already installed
check: flake8 mypy

# run pytest on package, assuming test requirements already installed
pytest:
	pytest testhiml

# run pytest fast subset on package, assuming test requirements already installed
pytest_fast:
	pytest -m fast testhiml

# Run pytest with coverage on package.
# Output the slowest tests via the --durations flag.
call_pytest_and_coverage:
	pytest --durations=20 --cov=health_ml --cov-branch --cov-report=html --cov-report=xml --cov-report=term-missing --cov-config=.coveragerc testhiml

# install test requirements and run pytest coverage
pytest_and_coverage: pip_test call_pytest_and_coverage

# install test requirements and run all tests
test_all: pip_test flake8 mypy call_pytest_and_coverage

# Smoke tests should all wait for completion and run for 15 minutes maximum.
# Tests run with the cpath Conda environment because of some odd problems with the hi-ml environment on GPU machines.
# Some of that can be fixed by pinning torch==1.10.0, but then that triggers problems with distutils.
define SHARED_ARGS
--conda_env ../hi-ml-cpath/environment.yml --cluster=pr-gpu --max_run_duration=15m --wait_for_completion
endef

# HelloWorld model training on a single node, submitted using the v1 SDK
smoke_helloworld_v1:
	python src/health_ml/runner.py --model=health_ml.HelloWorld --strictly_aml_v1=True --tag smoke_helloworld_v1 ${SHARED_ARGS}

# HelloWorld model training on a single node, submitted using the v2 SDK
smoke_helloworld_v2:
	python src/health_ml/runner.py --model=health_ml.HelloWorld --strictly_aml_v1=False --tag smoke_helloworld_v2 ${SHARED_ARGS}

# HelloWorld model training on 2 nodes, submitted using the v1 SDK
smoke_helloworld_v1_2nodes:
	python src/health_ml/runner.py --model=health_ml.HelloWorld --strictly_aml_v1=True --num_nodes=2 --tag smoke_helloworld_v1_2nodes ${SHARED_ARGS}

# HelloWorld model training on 2 nodes, submitted using the v2 SDK
smoke_helloworld_v2_2nodes:
	python src/health_ml/runner.py --model=health_ml.HelloWorld --strictly_aml_v1=False --num_nodes=2 --tag smoke_helloworld_v2_2nodes ${SHARED_ARGS}
