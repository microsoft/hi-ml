# pip upgrade
pip_upgrade:
	python -m pip install --upgrade pip

# pip install build requirements
pip_build: pip_upgrade
	pip install -r build_requirements.txt

# pip install test requirements
pip_test: pip_upgrade
	pip install -r test_requirements.txt

# pip install local package in editable mode for development and testing
pip_local: pip_upgrade
	pip install -e .

# pip install everything for local development and testing
pip: pip_build pip_test pip_local

# set the conda environment
conda:
	conda env update --file environment.yml

# run flake8
flake8: pip_test
	flake8 --count --statistics --config=../.flake8 .

# run mypy
mypy: pip_test
	python mypy_runner.py --config=../mypy.ini

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

# run example
example: pip_local
	echo 'edit src/health/azure/examples/elevate_this.py to reference your compute_cluster_name'
	cd src/health/azure/examples; python elevate_this.py --azureml --message 'running example from makefile'
