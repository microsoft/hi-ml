pip:
	python -m pip install --upgrade pip
	pip install -r build_requirements.txt
	pip install -r run_requirements.txt
	pip install -r test_requirements.txt
	pip install -e .

conda:
	conda env update --file environment.yml

flake8:
	flake8 . --statistics

mypy:
	cd src; mypy --install-types --non-interactive --config=../mypy.ini -p health
	mypy --install-types --non-interactive --config=mypy.ini -p testhiml
	mypy --install-types --non-interactive --config=mypy.ini setup.py

check: flake8 mypy

pytest:
	pip install -e .
	pytest testhiml

coverage:
	pip install -e .
	pytest --quiet --log-cli-level=critical --cov=src/health --cov-branch --cov-report=term-missing testhiml

testfast:
	pip install -e .
	pytest -m fast testhiml

test: flake8 mypy pytest

build:
	python setup.py sdist bdist_wheel

clean:
	rm -vrf ./build ./dist ./src/*.egg-info

example:
	echo 'edit src/health/azure/examples/elevate_this.py to reference your compute_cluster_name'
	pip install -e .
	cd src/health/azure/examples; python elevate_this.py --azureml --message 'running example from makefile'
