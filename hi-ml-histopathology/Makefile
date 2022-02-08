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


## Actions

# run flake8, assuming test requirements already installed
call_flake8:
	flake8 --count --statistics --config=../.flake8 .

# pip install test requirements and run flake8
flake8: pip_test call_flake8

# run mypy, assuming test requirements already installed
call_mypy:
	mypy --config=../mypy.ini --install-types --non-interactive setup.py
	mypy --config=../mypy.ini --install-types --non-interactive -p src
	mypy --config=../mypy.ini --install-types --non-interactive -p testhisto
	mypy --config=../mypy.ini --install-types --non-interactive -p testSSL

# pip install test requirements and run mypy
mypy: pip_test call_mypy

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