# Make commands for the toolbox users

# Create a Conda environment for this folder only
env:
	conda env create --file environment.yml
	pip install -r requirements_test.txt
	pip install -r requirements_build.txt

# Update the Conda environment
conda_update:
	conda env update -n $(CONDA_DEFAULT_ENV) --file environment.yml --prune

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
	mypy --install-types --show-error-codes --non-interactive --package health_cpath
	mypy --install-types --show-error-codes --non-interactive --package SSL
	mypy --install-types --show-error-codes --non-interactive --package testhisto
	mypy --install-types --show-error-codes --non-interactive --package testSSL

# run basic checks
check: flake8 mypy

# run pytest on package, assuming test requirements already installed
pytest:
	pytest

# run pytest with coverage on package
pytest_coverage:
	pytest --cov=health_cpath --cov SSL --cov-branch --cov-report=html --cov-report=xml --cov-report=term-missing --cov-config=.coveragerc

# Run regression tests and compare performance
regression_test_tilespandaimagenetmil:
	{ \
	cd ../ ;\
	python hi-ml/src/health_ml/runner.py --model=health_cpath.TilesPandaImageNetMIL --batch_size=2 \
	--cluster pr-gpu --mount_in_azureml --conda_env hi-ml-health_cpath/environment.yml \
	--regression_test_folder=testhisto/RegressionTestResults/TilesPANDAImageNetMIL/\
	HD_4ab0d833-fe55-44e8-aa04-cbaadbcc2733_0 --regression_test_csv_tolerance=0.5 --is_finetune \
	--regression_metrics='test/accuracy,test/macro_accuracy,test/weighted_accuracy,test/auroc,test/ISUP 0,test/ISUP 1,\
	test/ISUP 2,test/ISUP 3,test/ISUP4, test/ISUP5,test/loss_epoch';\
	}

regression_test_slidespandaimagenetmil:
	{ \
	cd ../ ;\
	python hi-ml/src/health_ml/runner.py --model=health_cpath.SlidesPandaImageNetMILBenchmark \
	--cluster pr-gpu --mount_in_azureml --conda_env hi-ml-cpath/environment.yml \
	--regression_test_folder=testhisto/RegressionTestResults/SlidesPANDAImageNetMIL/\
	HD_0e805b91-319d-4fde-8bc3-1cea3a6d08dd_0 --regression_test_csv_tolerance=0.5 --is_finetune \
	--regression_metrics='test/accuracy,test/macro_accuracy,test/weighted_accuracy,test/auroc,test/ISUP 0,test/ISUP 1,\
	test/ISUP 2,test/ISUP 3,test/ISUP4, test/ISUP5,test/loss_epoch';\
	}

regression_test_tcgacrcksslmil:
	{ \
	cd ../ ;\
	python hi-ml/src/health_ml/runner.py --model=health_cpath.TcgaCrckSSLMIL \
	--cluster pr-gpu --conda_env hi-ml-cpath/environment.yml --max_epochs=50 \
	--regression_test_folder=hi-ml-cpath/testhisto/RegressionTestResults/\
	TcgaCrckSSLMIL/HD_d76ef6cd-0403-4923-b8fa-dfd2827c5d74 --regression_test_csv_tolerance=0.5 \
	--regression_metrics=test/accuracy,test/auroc,test/f1score,test/precision,test/recall;\
	}

regression_test_crck_simclr:
	{ \
	cd ../; \
	python hi-ml/src/health_ml/runner.py --model=health_cpath.CRCK_SimCLR \
	--cluster pr-gpu --conda_env hi-ml-cpath/environment.yml \
	--regression_test_folder=hi-ml-cpath/testhisto/RegressionTestResults/CRCK_SimCLR/\
	CRCK_SimCLR_1653673515_42d53d78 --regression_test_csv_tolerance=0.5 \
	 --regression_metrics=ssl_online_evaluator/val/AreaUnderRocCurve,\
	ssl_online_evaluator/val/AreaUnderPRCurve,ssl_online_evaluator/val/AccuracyAtThreshold05 --max_epochs=200;\
	}

regression tests: regression_test_tilespandaimagenetmil regression_test_slidespandaimagenetmil regression_test_tcgacrcksslmil regression_test_crck_simclr

# Smoke tests (smaller tests that run end to end to check integration)
# The following test takes around 26 minutes due to saving hyperparameters
smoke_test_slidespandaimagenetmil:
	{ \
	cd ../ ;\
	python hi-ml/src/health_ml/runner.py --model=health_cpath.SlidesPandaImageNetMILBenchmark \
	--mount_in_azureml --conda_env hi-ml-cpath/environment.yml --is_finetune \
	--conda_env hi-ml-cpath/environment.yml --crossval_count=0 --num_top_slides=2 --num_top_tiles=2 \
	--pl_limit_train_batches=2 --pl_limit_val_batches=2 --pl_limit_test_batches=2 \
	--max_bag_size=3 --max_bag_size_inf=3 --max_epochs=2;\
	}

# The following test takes about 6 minutes
smoke_test_tilespandaimagenetmil:
	{ \
	cd ../ ;\
	python hi-ml/src/health_ml/runner.py --model=health_cpath.TilesPandaImageNetMIL \
	--mount_in_azureml --conda_env hi-ml-cpath/environment.yml \
	--is_finetune --batch_size=2 --crossval_count=0 --num_top_slides=2 --num_top_tiles=2 \
	--pl_limit_train_batches=2 --pl_limit_val_batches=2 --pl_limit_test_batches=2 \
	--max_bag_size=3 --max_bag_size_inf=3 --max_epochs=2;\
	}

# The following test takes about 30 seconds
smoke_test_tcgacrcksslmil:
	{ \
	cd ../ ;\
	python hi-ml/src/health_ml/runner.py --model=health_cpath.TcgaCrckSSLMIL \
	--conda_env hi-ml-cpath/environment.yml --crossval_count=1 \
	--pl_limit_train_batches=2 --pl_limit_val_batches=2 --pl_limit_test_batches=2 \
	--max_bag_size=3 --max_bag_size_inf=3 --max_epochs=2;\
	}

# The following test takes about 3 minutes
smoke_test_crck_simclr:
	{ \
	cd ../; \
	python hi-ml/src/health_ml/runner.py --model=health_cpath.CRCK_SimCLR \
	--conda_env hi-ml-cpath/environment.yml \
	--pl_limit_train_batches=2 --pl_limit_val_batches=2 --pl_limit_test_batches=2 \
	--is_debug_model=True --num_workers=0 --max_epochs=2; \
	}

smoke tests: smoke_test_slidespandaimagenetmil smoke_test_tilespandaimagenetmil smoke_test_tcgacrcksslmil smoke_test_crck_simclr
