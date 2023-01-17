#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import sys
import logging
import torch

from pathlib import Path
from typing import Dict, List, Optional

from azureml.core import Run
from pytorch_lightning import Trainer, seed_everything

from health_azure import AzureRunInfo
from health_azure.logging import logging_section
from health_azure.utils import (create_run_recovery_id, ENV_OMPI_COMM_WORLD_RANK,
                                is_running_in_azure_ml, PARENT_RUN_CONTEXT, RUN_CONTEXT,
                                aggregate_hyperdrive_metrics, get_metrics_for_childless_run,
                                ENV_GLOBAL_RANK, ENV_LOCAL_RANK, ENV_NODE_RANK,
                                is_local_rank_zero, is_global_rank_zero, create_aml_run_object)
from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
from health_ml.model_trainer import create_lightning_trainer, write_experiment_summary_file
from health_ml.utils import fixed_paths
from health_ml.utils.checkpoint_utils import cleanup_checkpoints
from health_ml.utils.checkpoint_handler import CheckpointHandler
from health_ml.utils.common_utils import (
    EFFECTIVE_RANDOM_SEED_KEY_NAME,
    change_working_directory,
    RUN_RECOVERY_ID_KEY,
    RUN_RECOVERY_FROM_ID_KEY_NAME,
    df_to_json,
    seed_monai_if_available,
)
from health_ml.utils.lightning_loggers import HimlMLFlowLogger, StoringLogger
from health_ml.utils.regression_test_utils import REGRESSION_TEST_METRICS_FILENAME, compare_folders_and_run_outputs
from health_ml.utils.type_annotations import PathOrString


def get_mlflow_run_id_from_previous_loggers(trainer: Optional[Trainer]) -> Optional[str]:
    """
    If self.trainer has already been intialised with loggers, attempt to retrieve a HimlMLFLowLogger and
    return the mlflow run_id associated with it, to allow continued logging to the same run. Otherwise, return None

    :return: The mlflow run id from the existing HimlMLFlowLogger
    """
    if trainer is None:
        return None
    try:
        mlflow_logger = [logger for logger in trainer.loggers if isinstance(logger, HimlMLFlowLogger)][0]
        return mlflow_logger.run_id
    except IndexError:
        return None


def check_dataset_folder_exists(local_dataset: PathOrString) -> Path:
    """
    Checks if a folder with a local dataset exists. If it does exist, return the argument converted
    to a Path instance. If it does not exist, raise a FileNotFoundError.

    :param local_dataset: The dataset folder to check.
    :return: The local_dataset argument, converted to a Path.
    """
    expected_dir = Path(local_dataset)
    if not expected_dir.is_dir():
        raise FileNotFoundError(f"The model uses a dataset in {expected_dir}, but that does not exist.")
    logging.info(f"Model training will use the local dataset provided in {expected_dir}")
    return expected_dir


class MLRunner:

    def __init__(self,
                 experiment_config: ExperimentConfig,
                 container: LightningContainer,
                 project_root: Optional[Path] = None) -> None:
        """
        Driver class to run a ML experiment. Note that the project root argument MUST be supplied when using hi-ml
        as a package!

        :param experiment_config: The ExperimentConfig object to use for training.
        :param container: The LightningContainer object to use for training.
        :param project_root: Project root. This should only be omitted if calling run_ml from the test suite. Supplying
        it is crucial when using hi-ml as a package or submodule!
        """
        self.container = container
        self.experiment_config = experiment_config
        self.container.num_nodes = self.experiment_config.num_nodes
        self.project_root: Path = project_root or fixed_paths.repository_root_directory()
        self.storing_logger: Optional[StoringLogger] = None
        self._has_setup_run = False
        self.checkpoint_handler = CheckpointHandler(container=self.container,
                                                    project_root=self.project_root,
                                                    run_context=RUN_CONTEXT)
        self.trainer: Optional[Trainer] = None
        self.azureml_run_for_logging: Optional[Run] = None
        self.mlflow_run_for_logging: Optional[str] = None
        self.inference_checkpoint: Optional[str] = None  # Passed to trainer.validate and trainer.test in inference mode

    def set_run_tags_from_parent(self) -> None:
        """
        Set metadata for the run
        """
        assert PARENT_RUN_CONTEXT, "This function should only be called in a Hyperdrive run."
        run_tags_parent = PARENT_RUN_CONTEXT.get_tags()
        tags_to_copy = [
            "tag",
            "model_name",
            "execution_mode",
            "recovered_from",
            "friendly_name",
            "build_number",
            "build_user",
            RUN_RECOVERY_FROM_ID_KEY_NAME
        ]
        new_tags = {tag: run_tags_parent.get(tag, "") for tag in tags_to_copy}
        new_tags[RUN_RECOVERY_ID_KEY] = create_run_recovery_id(run=RUN_CONTEXT)
        new_tags[EFFECTIVE_RANDOM_SEED_KEY_NAME] = str(self.container.get_effective_random_seed())
        RUN_CONTEXT.set_tags(new_tags)

    def setup(self, azure_run_info: Optional[AzureRunInfo] = None) -> None:
        """
        Sets the random seeds, calls the setup method on the LightningContainer and then creates the actual
        Lightning modules.

        :param azure_run_info: When running in AzureML or on a local VM, this contains the paths to the datasets.
        This can be missing when running in unit tests, where the local dataset paths are already populated.
        """
        if self._has_setup_run:
            return
        if azure_run_info:
            # Set up the paths to the datasets. azure_run_info already has all necessary information, using either
            # the provided local datasets for VM runs, or the AzureML mount points when running in AML.
            # This must happen before container setup because that could already read datasets.
            input_datasets = azure_run_info.input_datasets
            logging.info(f"Setting the following datasets as local datasets: {input_datasets}")
            if len(input_datasets) > 0:
                local_datasets: List[Path] = []
                for i, dataset in enumerate(input_datasets):
                    if dataset is None:
                        raise ValueError(f"Invalid setup: The dataset at index {i} is None")
                    local_datasets.append(check_dataset_folder_exists(dataset))
                self.container.local_datasets = local_datasets  # type: ignore
        # Ensure that we use fixed seeds before initializing the PyTorch models.
        # MONAI needs a separate method to make all transforms deterministic by default
        seed = self.container.get_effective_random_seed()
        seed_monai_if_available(seed)
        seed_everything(seed)

        # Creating the folder structure must happen before the LightningModule is created, because the output
        # parameters of the container will be copied into the module.
        self.container.create_filesystem(self.project_root)

        # configure recovery container if provided
        self.checkpoint_handler.download_recovery_checkpoints_or_weights()  # type: ignore

        self.container.setup()
        self.container.create_lightning_module_and_store()
        self._has_setup_run = True

        is_offline_run = not is_running_in_azure_ml(RUN_CONTEXT)
        # Get the AzureML context in which the script is running
        if not is_offline_run and PARENT_RUN_CONTEXT is not None:
            logging.info("Setting tags from parent run.")
            self.set_run_tags_from_parent()

    def get_multiple_trainloader_mode(self) -> str:
        # Workaround for a bug in PL 1.5.5: We need to pass the cycle mode for the training data as a trainer argument
        # because training data that uses a CombinedLoader is not split correctly in DDP. This flag cannot be passed
        # through the get_trainer_arguments method of the container because cycle mode is not yet available.
        multiple_trainloader_mode = "max_size_cycle"
        try:
            from SSL.data.datamodules import CombinedDataModule  # type: ignore
            if isinstance(self.data_module, CombinedDataModule):
                self.data_module.prepare_data()
                multiple_trainloader_mode = self.data_module.train_loader_cycle_mode  # type: ignore
                assert multiple_trainloader_mode, "train_loader_cycle_mode should be available now"
        except ModuleNotFoundError:
            pass
        return multiple_trainloader_mode

    def init_training(self) -> None:
        """
        Execute some bookkeeping tasks only once if running distributed and initialize the runner's trainer object.
        """
        if is_global_rank_zero():
            logging.info(f"Model checkpoints are saved at {self.container.checkpoint_folder}")
            write_experiment_summary_file(self.container, outputs_folder=self.container.outputs_folder)
            self.container.before_training_on_global_rank_zero()

        if is_local_rank_zero():
            self.container.before_training_on_local_rank_zero()
        self.container.before_training_on_all_ranks()

        # Set random seeds just before training. Ensure that dataloader workers are also seeded correctly.
        seed_everything(self.container.get_effective_random_seed(), workers=True)

        # Get the container's datamodule
        self.data_module = self.container.get_data_module()

        # Create an AzureML run for logging if running outside AzureML. This run will be used for metrics logging
        # during both training and inference. We can't rely on the automatically generated run inside the AzureMLLogger
        # class because two of those logger objects will be created, so training and inference metrics would be logged
        # in different runs.
        if self.container.log_from_vm:
            run = create_aml_run_object(experiment_name=self.container.effective_experiment_name)
            # Display name should already be set when creating the Run object, but in some scenarios this
            # does not happen. Hence, set it again.
            run.display_name = self.container.tag if self.container.tag else None
            self.azureml_run_for_logging = run

        if not self.container.run_inference_only:

            checkpoint_path_for_recovery = self.checkpoint_handler.get_recovery_or_checkpoint_path_train()
            if not checkpoint_path_for_recovery and self.container.resume_training:
                # If there is no recovery checkpoint (e.g job hasn't been resubmitted) and a source checkpoint is given,
                # use it to resume training.
                checkpoint_path_for_recovery = self.checkpoint_handler.trained_weights_path

            self.trainer, self.storing_logger = create_lightning_trainer(
                container=self.container,
                resume_from_checkpoint=checkpoint_path_for_recovery,
                num_nodes=self.container.num_nodes,
                multiple_trainloader_mode=self.get_multiple_trainloader_mode(),
                azureml_run_for_logging=self.azureml_run_for_logging)

            rank_info = ", ".join(
                f"{env}: {os.getenv(env)}" for env in [ENV_GLOBAL_RANK, ENV_LOCAL_RANK, ENV_NODE_RANK]
            )
            logging.info(f"Environment variables: {rank_info}. trainer.global_rank: {self.trainer.global_rank}")

    def after_ddp_cleanup(self, old_environ: Dict) -> None:
        """
        Run processes cleanup after ddp context to prepare for single device inference.
        Kill all processes in DDP besides rank 0.
        """

        # DDP will start multiple instances of the runner, one for each GPU. Those should terminate here after training.
        # We can now use the global_rank of the Lightning model, rather than environment variables, because DDP has set
        # all necessary properties.
        if self.container.model.global_rank != 0:
            logging.info(f"Terminating training thread with rank {self.container.model.global_rank}.")
            sys.exit()

        logging.info("Removing redundant checkpoint files.")
        cleanup_checkpoints(self.container.checkpoint_folder)
        # Lightning modifies a ton of environment variables. If we first run training and then the test suite,
        # those environment variables will mislead the training runs in the test suite, and make them crash.
        # Hence, restore the original environment after training.
        os.environ.clear()
        os.environ.update(old_environ)

        if ENV_OMPI_COMM_WORLD_RANK in os.environ:
            del os.environ[ENV_OMPI_COMM_WORLD_RANK]

        # From the training setup, torch still thinks that it should run in a distributed manner,
        # and would block on some GPU operations. Hence, clean up distributed training.
        if torch.distributed.is_initialized():  # type: ignore
            torch.distributed.destroy_process_group()  # type: ignore

    def is_crossval_disabled_or_child_0(self) -> bool:
        """
        Returns True if the present run is a non-cross-validation run, or child run 0 of a cross-validation run.
        """
        if self.container.is_crossvalidation_enabled:
            return self.container.crossval_index == 0
        return True

    def set_trainer_for_inference(self) -> None:
        """ Set the runner's PL Trainer object that should be used when running inference on the validation or test set.
        We run inference on a single device because distributed strategies such as DDP use DistributedSampler
        internally, which replicates some samples to make sure all devices have the same batch size in case of
        uneven inputs which biases the results."""
        mlflow_run_id = get_mlflow_run_id_from_previous_loggers(self.trainer)
        self.container.max_num_gpus = 1
        self.trainer, _ = create_lightning_trainer(
            container=self.container,
            num_nodes=1,
            azureml_run_for_logging=self.azureml_run_for_logging,
            mlflow_run_for_logging=mlflow_run_id
        )

    def init_inference(self) -> None:
        """ Prepare the runner for inference: validation or test. The following steps are performed:
        1. Get the checkpoint to use for inference. This is either the checkpoint from the last training epoch or the
        one specified in src_checkpoint argument.
        2. If the container has a run_extra_val_epoch method, call it to run an extra validation epoch.
        3. Create a new trainer instance for inference. This is necessary because the trainer is created with a single
        device in contrast to training that uses DDP if multiple GPUs are available.
        4. Create a new data module instance for inference to account for any requested changes in the dataloading
        parameters (e.g. batch_size, max_num_workers, etc) as part of on_run_extra_validation_epoch.
        """
        self.inference_checkpoint = str(self.checkpoint_handler.get_checkpoint_to_test())
        if self.container.run_extra_val_epoch:
            self.container.on_run_extra_validation_epoch()
        self.set_trainer_for_inference()
        self.data_module = self.container.get_data_module()

    def run_training(self) -> None:
        """
        The main training loop. It creates the Pytorch model based on the configuration options passed in,
        creates a Pytorch Lightning trainer, and trains the model.
        If a checkpoint was specified, then it loads the checkpoint before resuming training.
        The cwd is changed to the outputs folder so that the model can write to current working directory, and still
        everything is put into the right place in AzureML (only the contents of the "outputs" folder is treated as a
        result file).
        """
        with change_working_directory(self.container.outputs_folder):
            assert self.trainer, "Trainer should be initialized before training. Call self.init_training() first."
            self.trainer.fit(self.container.model, datamodule=self.data_module)
        for logger in self.trainer.loggers:
            assert logger is not None
            logger.finalize('success')

    def run_validation(self) -> None:
        """Run validation on the validation set for all models to save time/memory consuming outputs. This is done in
        inference only mode or when the user has requested an extra validation epoch. The cwd is changed to the outputs
        folder """
        if self.container.run_extra_val_epoch or self.container.run_inference_only:
            with change_working_directory(self.container.outputs_folder):
                assert self.trainer, "Trainer should be initialized before validation. Call self.init_inference()."
                self.trainer.validate(
                    self.container.model, datamodule=self.data_module, ckpt_path=self.inference_checkpoint
                )
        else:
            logging.info("Skipping extra validation because the user has not requested it.")

    def run_inference(self) -> None:
        """Run inference on the test set for all models. This is done by calling the LightningModule.test_step method.
        If the LightningModule.test_step method is not overridden, then this method does nothing. The cwd is changed to
        the outputs folder so that the model can write to current working directory, and still everything is put into
        the right place in AzureML (there, only the contents of the "outputs" folder is treated as a result file).
        """
        if self.container.has_custom_test_step():
            logging.info("Running inference via the LightningModule.test_step method")
            with change_working_directory(self.container.outputs_folder):
                assert self.trainer, "Trainer should be initialized before inference. Call self.init_inference()."
                _ = self.trainer.test(
                    self.container.model, datamodule=self.data_module, ckpt_path=self.inference_checkpoint
                )
        else:
            logging.warning("None of the suitable test methods is overridden. Skipping inference completely.")

    def run_regression_test(self) -> None:
        if self.container.regression_test_folder:
            with logging_section("Regression Test"):
                # Comparison with stored results for cross-validation runs only operates on child run 0. This run
                # has usually already downloaded the results for the other runs, and uploaded files to the parent
                # run context.
                regression_metrics_str = self.container.regression_metrics
                regression_metrics = regression_metrics_str.split(',') if regression_metrics_str else None
                # TODO: user should be able to override this value
                crossval_arg_name = self.container.CROSSVAL_INDEX_ARG_NAME

                logging.info("Comparing the current results against stored results.")
                if self.is_crossval_disabled_or_child_0():
                    if is_running_in_azure_ml:
                        if PARENT_RUN_CONTEXT is not None:
                            df = aggregate_hyperdrive_metrics(
                                child_run_arg_name=crossval_arg_name,
                                run=PARENT_RUN_CONTEXT,
                                keep_metrics=regression_metrics)
                        else:
                            df = get_metrics_for_childless_run(
                                run=RUN_CONTEXT,
                                keep_metrics=regression_metrics)

                        if not df.empty:
                            metrics_filename = self.container.outputs_folder / REGRESSION_TEST_METRICS_FILENAME
                            logging.info(f"Saving metrics to {metrics_filename}")
                            df_to_json(df, metrics_filename)

                    compare_folders_and_run_outputs(expected=self.container.regression_test_folder,
                                                    actual=self.container.outputs_folder,
                                                    csv_relative_tolerance=self.container.regression_test_csv_tolerance)
                else:
                    logging.info("Skipping as this is not cross-validation child run 0.")
        else:
            logging.info("Skipping regression test, no available results to compare with.")

    def run(self) -> None:
        """
        Driver function to run a ML experiment
        """
        self.setup()
        try:
            self.init_training()

            if not self.container.run_inference_only:
                # Backup the environment variables in case we need to run a second training in the unit tests.
                old_environ = dict(os.environ)
                # Do training
                with logging_section("Model training"):
                    self.run_training()
                # Update the checkpoint handler state
                self.checkpoint_handler.additional_training_done()
                # Kill all processes besides rank 0 after training is done to start inference on a single device
                self.after_ddp_cleanup(old_environ)

            self.init_inference()

            with logging_section("Model validation"):
                self.run_validation()

            with logging_section("Model inference"):
                self.run_inference()

            self.run_regression_test()

        finally:
            if self.azureml_run_for_logging is not None:
                try:
                    self.azureml_run_for_logging.complete()
                except Exception as ex:
                    logging.error("Failed to complete AzureML run: %s", ex)
