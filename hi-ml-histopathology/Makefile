# Make commands for the toolbox users

# Create a Conda environment for this folder only
env:
	conda env create --file environment.yml
	pip install -r requirements_test.txt
	pip install -r requirements_build.txt

# Package management

# pip upgrade
pip_upgrade:
	python -m pip install --upgrade pip

# pip upgrade and install build requirements
pip_build: pip_upgrade
	pip install -r requirements_build.txt

# pip upgrade and install test requirements
pip_test: pip_upgrade
	pip install -r requirements_test.txt

# pip install all requirements for histo, read off the Conda file. This is somewhat hacky,
# we could also build a full Conda before starting the tests. Unclear about the performance
# impact of that.
pip_from_conda:
	sed -e '1,/pip:/ d' environment.yml | grep -v "#" | cut -d "-" -f 2- > temp_requirements.txt
	pip install -r temp_requirements.txt

# clean build artifacts
clean:
	rm -vrf `find . -type d -name __pycache__`
	rm -vrf `find . -type d -name logs`
	rm -vrf `find . -type d -name outputs`
	rm -vrf ./.mypy_cache ./.pytest_cache
	rm -vrf ./testhisto/testhisto/test_outputs ./testhistotestSSL/test_ouputs
	rm -vf ./coverage ./coverage.txt ./coverage.xml

# run flake8, assuming test requirements already installed
flake8:
	flake8 --count --statistics .

# run mypy, assuming test requirements already installed
mypy:
	mypy --install-types --non-interactive -p histopathology
	mypy --install-types --non-interactive -p SSL
	mypy --install-types --non-interactive -p testhisto
	mypy --install-types --non-interactive -p testSSL

# run basic checks
check: flake8 mypy

# run pytest on package, assuming test requirements already installed
pytest:
	pytest

# run pytest with coverage on package
pytest_coverage:
	pytest --cov=histopathology --cov SSL --cov-branch --cov-report=html --cov-report=xml --cov-report=term-missing --cov-config=.coveragerc

# Run regression tests and compare performance
run_regression_test:
	cd ../; python hi-ml/src/health_ml/runner.py --model=cpath.TilesPandaImageNetMIL --azureml --tag panda_transformer_newdataset --cluster innereye4cl --mount_in_azureml --additional_env_files hi-ml/hi-ml-histopathology/environment.yml --is_finetune --regression_test_folder=hi-ml-histopathology/RegressionTestResults/HD_4ab0d833-fe55-44e8-aa04-cbaadbcc2733_0/OUTPUT --regression_test_csv_tolerance=0.05
