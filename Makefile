init:
	pip install -r requirements.txt

test:
	pytest tests

build:
	python setup.py sdist bdist_wheel