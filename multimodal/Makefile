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

# pip install local package in editable mode for development and testing
pip_local:
	pip install -e .

# pip install everything for local development and testing
pip: pip_upgrade pip_build pip_test pip_local

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
	mypy --install-types --non-interactive setup.py
	mypy --install-types --non-interactive -p health_multimodal
	mypy --install-types --non-interactive -p test_multimodal

# run basic checks
check: flake8 mypy

# install test requirements and run basic checks
check: call_check

# run pytest on package, assuming test requirements already installed
pytest:
	pytest test_multimodal
