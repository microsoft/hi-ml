# pip upgrade
pip_upgrade:
	python -m pip install --upgrade pip

# pip install build requirements
pip_build: pip_upgrade
	pip install -r build_requirements.txt

# pip install test requirements
pip_test: pip_upgrade
	pip install -r test_requirements.txt

# pip install local packages in editable mode for development and testing
pip_local:
	cd hi-ml-azure && $(MAKE) pip_local
	cd hi-ml && $(MAKE) pip_local

# pip install everything for local development and testing
pip: pip_build pip_test pip_local

# run flake8
flake8:
	cd hi-ml-azure && $(MAKE) flake8
	cd hi-ml && $(MAKE) flake8

# run mypy
mypy:
	cd hi-ml-azure && $(MAKE) mypy
	cd hi-ml && $(MAKE) mypy

# run basic checks
check: flake8 mypy

# set the conda environment
conda:
	conda env update --file environment.yml
