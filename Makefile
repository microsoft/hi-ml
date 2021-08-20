pip_upgrade:
	python -m pip install --upgrade pip

pip_build: pip_upgrade
	pip install -r build_requirements.txt

pip_test: pip_upgrade
	pip install -r test_requirements.txt

pip_editable: pip_upgrade
	pip install -e .

pip: pip_build pip_test pip_editable

conda:
	conda env update --file environment.yml

flake8:
	flake8 --count --statistics --config=.flake8 .

mypy:
	python mypy_runner.py

check: flake8 mypy

pytest: pip_editable pip_test
	pytest testhiml

coverage: pip_editable pip_test
	pytest --quiet --log-cli-level=critical --cov=src/health --cov-branch --cov-report=term-missing testhiml

pytest_and_coverage: pip_test
	pytest --cov=health --cov-branch --cov-report=html --cov-report=term-missing --cov-report=xml testhiml
	pycobertura show --format text --output coverage.txt coverage.xml

testfast_no_install: pip_test
	pytest -m fast testhiml

testfast: pip_editable testfast_no_install

test: flake8 mypy pytest

build: pip_build
	python setup.py sdist bdist_wheel

clean:
	rm -vrf ./build ./dist ./src/*.egg-info

example: pip_editable
	echo 'edit src/health/azure/examples/elevate_this.py to reference your compute_cluster_name'
	cd src/health/azure/examples; python elevate_this.py --azureml --message 'running example from makefile'
