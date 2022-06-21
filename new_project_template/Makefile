## Package management

# pip upgrade
pip_upgrade:
	python -m pip install --upgrade pip

# pip upgrade and install build requirements
pip_build: pip_upgrade
	pip install -r requirements_build.txt

# pip upgrade and install test requirements
pip_test: pip_upgrade
	pip install -r requirements_test.txt

# pip install local package in editable mode for development and testing
pip_local:
	pip install -e .

# pip install everything for local development and testing
pip: pip_build pip_test pip_local

## Actions

# clean build artifacts
clean:
	rm -vrf `find . -type d -name __pycache__`
	rm -vrf `find . -type d -name logs`
	rm -vrf `find . -type d -name outputs`
	rm -vrf ./.mypy_cache ./.pytest_cache ./build ./dist ./htmlcov ./*.egg-info
	rm -vf ./coverage ./coverage.txt ./coverage.xml ./latest_version.txt ./most_recent_run.txt ./package_name.txt

# build package, assuming build requirements already installed
build:
	python setup.py sdist bdist_wheel

# run flake8, assuming test requirements already installed
flake8:
	flake8 --count --statistics .

# run mypy, assuming test requirements already installed
mypy:
	mypy --install-types --show-error-codes --non-interactive setup.py
	mypy --install-types --show-error-codes --non-interactive --package health_newproject
	mypy --install-types --show-error-codes --non-interactive --package test_newproject

# run pytest on package, assuming test requirements already installed
pytest:
	pytest test_newproject

# Install pyright
pyright_install:
	npm install -g pyright

# conda install test requirements and run pyright
pyright:
	pyright

# run basic checks
check: flake8 mypy pyright
