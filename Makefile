pip:
	python -m pip install --upgrade pip
	pip install -r build_requirements.txt
	pip install -r run_requirements.txt
	pip install -r test_requirements.txt
	pip install -e .

conda:
	conda env update --file environment.yml
	conda activate himl

flake8:
	flake8 . --statistics

mypy:
	mypy setup.py
	mypy -p src
	mypy -p testhiml

check: flake8 mypy

pytest:
	pip install -e .
	pytest testhiml

test: flake8 mypy pytest

build:
	python setup.py sdist bdist_wheel

clean:
	rm -vrf ./build ./dist ./src/*.egg-info
