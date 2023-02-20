# Make commands for the toolbox users

# Create a Conda environment for this folder only
env:
	conda env create --file environment.yml

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

# Lock the current Conda environment secondary dependencies versions
lock_env:
	../create_and_lock_environment.sh

# clean build artifacts
clean:
	for folder in .mypy_cache __pycache__ logs outputs; do \
		rm -rf `find . -type d -name $$folder`; \
	done
	rm -rf ./.pytest_cache
	rm -rf ./testhisto/test_outputs ./testSSL/test_outputs
	rm -f ./coverage ./coverage.txt ./coverage.xml
	rm -rf Lightning checkpoints lightning_logs
	rm -f ./latest_version.txt ./package_name.txt
	rm -rf build dist hi_ml_cpath.egg-info
	rm cifar-10-python.tar.gz
	rm -rf cifar-10-batches-py

# build package, assuming build requirements already installed
build:
	python setup.py sdist bdist_wheel

# Brief smoke-test of the package, by just importing one of the essential modules.
test_wheel:
	pip uninstall -y hi-ml hi-ml-azure
	pip install -e ../hi-ml/
	pip install -e ../hi-ml-azure/
	pip install `ls dist/*.whl`
	python -c "from health_cpath.configs.classification.BaseMIL import BaseMIL"

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

# Run pytest with coverage on package
# Output the slowest tests via the --durations flag.
pytest_coverage:
	pytest --durations=20 --cov=health_cpath --cov SSL --cov-branch --cov-report=html --cov-report=xml --cov-report=term-missing --cov-config=.coveragerc

# run pytest using GPUs via an AzureML job
pytest_gpu:
	python ../hi-ml-azure/run_pytest.py \
		--mark=gpu --cluster=pr-gpu \
		--conda_env=environment.yml \
		--folder=hi-ml-cpath \
		--coverage_module=health_cpath \
		--strictly_aml_v1=True

SSL_CKPT_RUN_ID_CRCK := CRCK_SimCLR_1655731022_85790606
SRC_CKPT_RUN_ID_CRCK := TcgaCrckSSLMIL_1667832811_4e855804

# Run regression tests and compare performance
define BASE_CPATH_RUNNER_COMMAND
cd ../ ; \
python hi-ml/src/health_ml/runner.py --mount_in_azureml --conda_env=hi-ml-cpath/environment.yml --datastore=himldatasets
endef

define DEEPSMILEPANDASLIDES_ARGS
--model=health_cpath.SlidesPandaImageNetMILBenchmark --tune_encoder
endef

define DEEPSMILEPANDATILES_ARGS
--model=health_cpath.TilesPandaImageNetMIL --tune_encoder --batch_size=2
endef

define TCGACRCKSSLMIL_ARGS
--model=health_cpath.TcgaCrckSSLMIL --ssl_checkpoint=${SSL_CKPT_RUN_ID_CRCK}
endef

define TCGACRCKIMANEGETMIL_ARGS
--model=health_cpath.TcgaCrckImageNetMIL --is_caching=False --batch_size=2 --max_num_gpus=1
endef

define CRCKSIMCLR_ARGS
--model=SSL.CRCK_SimCLR
endef

define REGRESSION_TEST_ARGS
--cluster dedicated-nc24s-v2 --regression_test_csv_tolerance=0.5 --strictly_aml_v1=True
endef

define PANDA_REGRESSION_METRICS
--regression_metrics='test/accuracy,test/macro_accuracy,test/weighted_accuracy,test/auroc,test/ISUP 0,test/ISUP 1,\
test/ISUP 2,test/ISUP 3,test/ISUP 4, test/ISUP 5,test/loss_epoch'
endef

regression_test_tilespandaimagenetmil:
	{ \
    ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDATILES_ARGS} ${REGRESSION_TEST_ARGS} \
	--regression_test_folder=testhisto/RegressionTestResults/TilesPANDAImageNetMIL/\
	HD_4ab0d833-fe55-44e8-aa04-cbaadbcc2733_0\
	${PANDA_REGRESSION_METRICS};\
	}

regression_test_slidespandaimagenetmil:
	{ \
	${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDASLIDES_ARGS} ${REGRESSION_TEST_ARGS} \
	--regression_test_folder=testhisto/RegressionTestResults/SlidesPANDAImageNetMIL/\
	HD_0e805b91-319d-4fde-8bc3-1cea3a6d08dd_0 \
	${PANDA_REGRESSION_METRICS};\
	}

regression_test_tcgacrcksslmil:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${TCGACRCKSSLMIL_ARGS} ${REGRESSION_TEST_ARGS} --max_epochs=50 \
	--regression_test_folder=hi-ml-cpath/testhisto/RegressionTestResults/\
	TcgaCrckSSLMIL/HD_d76ef6cd-0403-4923-b8fa-dfd2827c5d74 \
	--regression_metrics=test/accuracy,test/auroc,test/f1score,test/precision,test/recall;\
	}

regression_test_crck_simclr:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${CRCKSIMCLR_ARGS}\
	${REGRESSION_TEST_ARGS}  --max_epochs=200 \
	--regression_test_folder=hi-ml-cpath/testhisto/RegressionTestResults/CRCK_SimCLR/\
	CRCK_SimCLR_1653673515_42d53d78 --regression_test_csv_tolerance=0.5 \
	 --regression_metrics=ssl_online_evaluator/val/AreaUnderRocCurve,\
	ssl_online_evaluator/val/AreaUnderPRCurve,ssl_online_evaluator/val/AccuracyAtThreshold05;\
	}

regression tests: regression_test_tilespandaimagenetmil regression_test_slidespandaimagenetmil regression_test_tcgacrcksslmil regression_test_crck_simclr

# Smoke tests (smaller tests that run end to end to check integration)
define DEFAULT_SMOKE_TEST_ARGS
--pl_limit_train_batches=2 --pl_limit_val_batches=2 --pl_limit_test_batches=2 \
--max_epochs=2
endef

define AML_ONE_DEVICE_ARGS
--cluster=testing-nc6 --wait_for_completion --num_nodes=1 --max_num_gpus=1 --strictly_aml_v1=True --max_run_duration=1h
endef

define AML_MULTIPLE_DEVICE_ARGS
--cluster=dedicated-nc24s-v2 --wait_for_completion --strictly_aml_v1=True --max_run_duration=1h
endef

define DEEPSMILEDEFAULT_SMOKE_TEST_ARGS
--crossval_count=0 --num_top_slides=2 --num_top_tiles=2 --max_bag_size=3 \
--max_bag_size_inf=3
endef

define TCGACRCKSSLMIL_SMOKE_TEST_ARGS
--crossval_count=1 --max_bag_size=3 --max_bag_size_inf=3
endef

define CRCKSIMCLR_SMOKE_TEST_ARGS
 --is_debug_model=True --num_workers=0
endef

define CRCK_TUNING_ARGS
--tune_encoder=False --tune_pooling=True --tune_classifier=True --pretrained_encoder=True \
--pretrained_pooling=True --pretrained_classifier=True --src_checkpoint=${SRC_CKPT_RUN_ID_CRCK}
endef

define LOSS_ANALYSIS_ARGS
--analyse_loss=True --loss_analysis_patience=0 --loss_analysis_epochs_interval=1 \
--num_slides_scatter=2 --num_slides_heatmap=2 --save_tile_ids=True
endef

define DDP_SAMPLER_ARGS
--pl_replace_sampler_ddp=False
endef

define OPENSLIDE_BACKEND_ARGS
--backend=OpenSlide
endef

# The following test takes around 5 minutes
smoke_test_cucim_slidespandaimagenetmil_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDASLIDES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS};}

# Once running in AML the following test takes around 6 minutes
smoke_test_cucim_slidespandaimagenetmil_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDASLIDES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS} ${AML_MULTIPLE_DEVICE_ARGS} --tag smoke_test_cucim_slidespandaimagenetmil_aml;}

smoke_test_openslide_slidespandaimagenetmil_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDASLIDES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS} ${OPENSLIDE_BACKEND_ARGS};}

smoke_test_openslide_slidespandaimagenetmil_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDASLIDES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS} ${OPENSLIDE_BACKEND_ARGS} ${AML_MULTIPLE_DEVICE_ARGS} --tag smoke_test_openslide_slidespandaimagenetmil_aml;}

# The following test takes about 6 minutes
smoke_test_tilespandaimagenetmil_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDATILES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS};}

smoke_test_tilespandaimagenetmil_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDATILES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS}  ${AML_MULTIPLE_DEVICE_ARGS} --tag smoke_test_tilespandaimagenetmil_aml;}

# The following test takes about 30 seconds
smoke_test_tcgacrcksslmil_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${TCGACRCKSSLMIL_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${TCGACRCKSSLMIL_SMOKE_TEST_ARGS};}

smoke_test_tcgacrcksslmil_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${TCGACRCKSSLMIL_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${TCGACRCKSSLMIL_SMOKE_TEST_ARGS} ${AML_MULTIPLE_DEVICE_ARGS} --tag smoke_test_tcgacrcksslmil_aml;}

smoke_test_tcgacrckimagenetmil_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${TCGACRCKIMANEGETMIL_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS};}

smoke_test_tcgacrckimagenetmil_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${TCGACRCKIMANEGETMIL_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS} ${AML_MULTIPLE_DEVICE_ARGS} --tag smoke_test_tcgacrckimagenetmil_aml;}

# The following test takes about 3 minutes
smoke_test_crck_simclr_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${CRCKSIMCLR_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${CRCKSIMCLR_SMOKE_TEST_ARGS};}

smoke_test_crck_simclr_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${CRCKSIMCLR_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${CRCKSIMCLR_SMOKE_TEST_ARGS} ${AML_MULTIPLE_DEVICE_ARGS} --tag smoke_test_crck_simclr_aml;}

smoke_test_crck_flexible_finetuning_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${TCGACRCKSSLMIL_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${TCGACRCKSSLMIL_SMOKE_TEST_ARGS} ${CRCK_TUNING_ARGS};}

smoke_test_crck_flexible_finetuning_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${TCGACRCKSSLMIL_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${TCGACRCKSSLMIL_SMOKE_TEST_ARGS} ${AML_MULTIPLE_DEVICE_ARGS} ${CRCK_TUNING_ARGS} --tag smoke_test_crck_flexible_finetuning_aml;}

smoke_test_crck_loss_analysis_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${TCGACRCKSSLMIL_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${TCGACRCKSSLMIL_SMOKE_TEST_ARGS} ${LOSS_ANALYSIS_ARGS};}

smoke_test_crck_loss_analysis_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${TCGACRCKSSLMIL_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${TCGACRCKSSLMIL_SMOKE_TEST_ARGS} ${AML_MULTIPLE_DEVICE_ARGS} ${LOSS_ANALYSIS_ARGS} --tag smoke_test_crck_loss_analysis_aml;}

smoke_test_slides_panda_loss_analysis_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDASLIDES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS} ${LOSS_ANALYSIS_ARGS};}

smoke_test_slides_panda_loss_analysis_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDASLIDES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS} ${LOSS_ANALYSIS_ARGS} ${AML_MULTIPLE_DEVICE_ARGS} --tag smoke_test_slides_panda_loss_analysis_aml;}

smoke_test_slides_panda_no_ddp_sampler_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDASLIDES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS} ${DDP_SAMPLER_ARGS};}

smoke_test_slides_panda_no_ddp_sampler_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDASLIDES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS} ${DDP_SAMPLER_ARGS} ${AML_MULTIPLE_DEVICE_ARGS} --tag smoke_test_slides_panda_no_ddp_sampler_aml;}

smoke_test_tiles_panda_no_ddp_sampler_local:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDATILES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS} ${DDP_SAMPLER_ARGS};}

smoke_test_tiles_panda_no_ddp_sampler_aml:
	{ ${BASE_CPATH_RUNNER_COMMAND} ${DEEPSMILEPANDATILES_ARGS} ${DEFAULT_SMOKE_TEST_ARGS} \
	${DEEPSMILEDEFAULT_SMOKE_TEST_ARGS} ${DDP_SAMPLER_ARGS} ${AML_MULTIPLE_DEVICE_ARGS} --tag smoke_test_tiles_panda_no_ddp_sampler_aml;}

smoke_test_montage_creation:
	python src/health_cpath/scripts/create_montage.py --dataset test_montage --width 1000 --cluster lite-testing-ds2 --datastore himldatasets --conda_env environment.yml --wait_for_completion

smoke tests local: smoke_test_cucim_slidespandaimagenetmil_local smoke_test_openslide_slidespandaimagenetmil_local smoke_test_tilespandaimagenetmil_local smoke_test_tcgacrcksslmil_local smoke_test_crck_simclr_local smoke_test_crck_flexible_finetuning_local smoke_test_tcgacrckimagenetmil_local smoke_test_crck_loss_analysis_local smoke_test_slides_panda_loss_analysis_local smoke_test_slides_panda_no_ddp_sampler_local smoke_test_tiles_panda_no_ddp_sampler_local

smoke tests AML: smoke_test_cucim_slidespandaimagenetmil_aml smoke_test_openslide_slidespandaimagenetmil_aml smoke_test_tilespandaimagenetmil_aml smoke_test_tcgacrcksslmil_aml smoke_test_crck_simclr_aml smoke_test_crck_flexible_finetuning_aml smoke_test_tcgacrckimagenetmil_aml smoke_test_crck_loss_analysis_aml smoke_test_slides_panda_loss_analysis_aml smoke_test_slides_panda_no_ddp_sampler_aml smoke_test_tiles_panda_no_ddp_sampler_aml
