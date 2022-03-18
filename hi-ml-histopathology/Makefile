# Make commands for the toolbox users

# Create a Conda environment for this folder only
env:
	conda env create --file environment.yml
	pip install -r ../test_requirements.txt

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

# pip install all requirements for histo, read off the Conda file. This is somewhat hacky,
# we could also build a full Conda before starting the tests. Unclear about the performance
# impact of that.
call_pip_local:
	sed -e '1,/pip:/ d' environment.yml | cut -d "-" -f 2- > temp_requirements.txt
	pip install -r temp_requirements.txt

# pip upgrade and install local package in editable mode
pip_local: pip_upgrade call_pip_local

# pip install everything for local development and testing
pip: pip_build pip_test


## Actions
# clean build artifacts
clean:
	rm -vrf `find . -type d -name __pycache__`
	rm -vrf `find . -type d -name logs`
	rm -vrf `find . -type d -name outputs`
	rm -vrf ./.mypy_cache ./.pytest_cache
	rm -vrf ./testhisto/testhisto/test_outputs ./testhistotestSSL/test_ouputs
	rm -vf ./coverage ./coverage.txt ./coverage.xml

# run flake8, assuming test requirements already installed
call_flake8:
	flake8 --count --statistics --config=../.flake8 .

# pip install test requirements and run flake8
flake8: pip_test call_flake8

# run mypy for hi-ml and hi-ml-azure, assuming test requirements already installed
call_mypy:
	mypy --config=../mypy.ini --install-types --non-interactive -p src
	mypy --config=../mypy.ini --install-types --non-interactive -p testhisto
	mypy --config=../mypy.ini --install-types --non-interactive -p testSSL

# pip install test requirements and run mypy
mypy: pip_test call_mypy

# run pyright, assuming test requirements already installed
call_pyright:
	npm install -g pyright
	pyright

# conda install test requirements and run pyright
pyright: conda call_pyright

# run basic checks
call_check: call_flake8 call_mypy

# install test requirements and run basic checks
check: pip_test call_check

# run pytest on package, assuming test requirements already installed
call_pytest:
	pytest testhisto testSSL

# install test requirements and run tests
pytest: pip_test testhisto testSSL

# run pytest fast subset on package, assuming test requirements already installed
call_pytest_fast:
	pytest -m fast testhisto testSSL

# install test requirements and run pytest fast subset
pytest_fast: pip_test call_pytest_fast

# run pytest with coverage on package, and format coverage output as a text file, assuming test requirements already installed
call_pytest_and_coverage:
	pytest --cov=histopathology  --cov-branch --cov-report=html --cov-report=xml --cov-report=term-missing --cov-config=.coveragerc testhisto
	pytest --cov=SSL  --cov-branch --cov-report=html --cov-report=xml --cov-report=term-missing --cov-config=.coveragerc testSSL
	pycobertura show --format text --output coverage.txt coverage.xml

# install test requirements and run pytest coverage
pytest_and_coverage: pip_test call_pytest_and_coverage

# install test requirements and run all tests
test_all: pip_test call_flake8 call_mypy call_pytest_and_coverage
