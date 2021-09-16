# pip upgrade
pip_upgrade:
	cd .. && $(MAKE) pip_upgrade

# pip install build requirements
pip_build:
	cd .. && $(MAKE) pip_build

# pip install test requirements
pip_test:
	cd .. && $(MAKE) pip_test

# pip install local package in editable mode for development and testing
pip_local: pip_upgrade
	pip install -e .

# pip install everything for local development and testing
pip: pip_build pip_test pip_local

# run flake8
flake8: pip_test
	flake8 --count --statistics --config=../.flake8 .

# run mypy
mypy: pip_test
	mypy --config=../mypy.ini setup.py 
	mypy --config=../mypy.ini -p src
	mypy --config=../mypy.ini -p testhiml

# run basic checks
check: flake8 mypy

# run pytest on local package
pytest: pip_local pip_test
	pytest testhiml

# run coverage on local package
coverage: pip_local pip_test
	pytest --quiet --log-cli-level=critical --cov=health --cov-branch --cov-report=term-missing testhiml

# run pytest with coverage on health package, however it may be installed, and format coverage output as a text file
pytest_and_coverage: pip_test
	pytest --cov=health --cov-branch --cov-report=html --cov-report=term-missing --cov-report=xml testhiml
	pycobertura show --format text --output coverage.txt coverage.xml

# run pytest fast subset on health package, however it may be installed
testfast_no_install: pip_test
	pytest -m fast testhiml

# run pytest fast subset on local package
testfast: pip_local testfast_no_install

# run all tests
test: flake8 mypy pytest

# build package
build: pip_build
	python setup.py sdist bdist_wheel

# clean build artifacts
clean:
	rm -vrf ./build ./dist ./src/*.egg-info
