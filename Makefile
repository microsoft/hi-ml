init:
	python -m pip install --upgrade pip
	pip install -r build_requirements.txt
	pip install -r run_requirements.txt
	pip install -r test_requirements.txt

test_flake8:
	flake8 . --statistics

test_mypy:
	mypy setup.py
	mypy -p src
	mypy -p testhiml

test_pytest:
	pytest testhiml

test: test_flake8 test_mypy test_pytest

build:
	python setup.py sdist bdist_wheel

clean:
	rm -vrf ./build ./dist ./src/*.egg-info