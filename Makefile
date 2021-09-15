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
