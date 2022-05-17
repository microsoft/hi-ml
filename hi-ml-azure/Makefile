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
	rm -vrf `find . -type d -name __pycache__`
	rm -vrf `find . -type d -name logs`
	rm -vrf `find . -type d -name outputs`
	rm -vrf ./.mypy_cache ./.pytest_cache ./build ./dist ./htmlcov ./src/*.egg-info
	rm -vrf ./testazure/testazure/test_data/test_make_tests
	rm -vf ./coverage ./coverage.txt ./coverage.xml ./latest_version.txt ./most_recent_run.txt ./package_name.txt

# build package, assuming build requirements already installed
call_build:
	python setup.py sdist bdist_wheel

# pip install build requirements and build package
build: pip_build call_build

# run flake8, assuming test requirements already installed
call_flake8:
	flake8 --count --statistics .

# pip install test requirements and run flake8
flake8: pip_test call_flake8

# run mypy, assuming test requirements already installed
call_mypy:
	mypy --install-types --non-interactive setup.py
	mypy --install-types --non-interactive -p health_azure
	mypy --install-types --non-interactive -p testazure

# pip install test requirements and run mypy
mypy: pip_test call_mypy

# run basic checks
call_check: call_flake8 call_mypy

# install test requirements and run basic checks
check: pip_test call_check

# run pytest on package, assuming test requirements already installed
call_pytest:
	pytest testazure

# install test requirements and run tests
pytest: pip_test call_pytest

# run pytest fast subset on package, assuming test requirements already installed
call_pytest_fast:
	pytest -m fast testazure

# install test requirements and run pytest fast subset
pytest_fast: pip_test call_pytest_fast

# run pytest with coverage on package
call_pytest_and_coverage:
	pytest --cov=health_azure  --cov-branch --cov-report=html --cov-report=xml --cov-report=term-missing --cov-config=.coveragerc testazure

# install test requirements and run pytest coverage
pytest_and_coverage: pip_test call_pytest_and_coverage

# install test requirements and run all tests
test_all: pip_test call_flake8 call_mypy call_pytest_and_coverage

# run example
example: pip_local
	echo 'edit src/health/azure/examples/elevate_this.py to reference your compute_cluster_name'
	cd src/health/azure/examples; python elevate_this.py --azureml --message 'running example from makefile'
