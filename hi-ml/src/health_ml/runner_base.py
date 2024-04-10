#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import List, Optional

from azureml.core import Run
from pytorch_lightning import LightningDataModule, Trainer, seed_everything

from health_azure import AzureRunInfo
from health_azure.utils import (
    PARENT_RUN_CONTEXT,
    RUN_CONTEXT,
    create_aml_run_object,
    create_run_recovery_id,
    is_running_in_azure_ml,
)
from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
from health_ml.model_trainer import create_lightning_trainer
from health_ml.utils import fixed_paths
from health_ml.utils.checkpoint_handler import CheckpointHandler
from health_ml.utils.common_utils import (
    EFFECTIVE_RANDOM_SEED_KEY_NAME,
    RUN_RECOVERY_FROM_ID_KEY_NAME,
    RUN_RECOVERY_ID_KEY,
    change_working_directory,
    seed_monai_if_available,
)
from health_ml.utils.lightning_loggers import StoringLogger, get_mlflow_run_id_from_trainer
from health_ml.utils.type_annotations import PathOrString


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
    logging.info(f"Model will use the local dataset provided in {expected_dir}")
    return expected_dir


class RunnerBase:
    """
    A base class with operations that are shared between the training/test runner and the evaluation-only runner.
    """

    def __init__(
        self, experiment_config: ExperimentConfig, container: LightningContainer, project_root: Optional[Path] = None
    ) -> None:
        """
        Driver class to run an ML experiment. Note that the project root argument MUST be supplied when using hi-ml
        as a package!

        :param experiment_config: The ExperimentConfig object to use for training.
        :param container: The LightningContainer object to use for training.
        :param project_root: Project root. This should only be omitted if calling run_ml from the test suite. Supplying
            it is crucial when using hi-ml as a package or submodule!
        """
        self.container = container
        self.experiment_config = experiment_config
        self.container.num_nodes = self.experiment_config.num_nodes
        self.container.runner_mode = self.experiment_config.mode
        self.project_root: Path = project_root or fixed_paths.repository_root_directory()
        self.storing_logger: Optional[StoringLogger] = None
        self._has_setup_run = False
        self.checkpoint_handler = CheckpointHandler(
            container=self.container, project_root=self.project_root, run_context=RUN_CONTEXT
        )
        self.trainer: Optional[Trainer] = None
        self.azureml_run_for_logging: Optional[Run] = None
        self.mlflow_run_for_logging: Optional[str] = None
        # This is passed to trainer.validate and trainer.test in inference mode
        self.inference_checkpoint: Optional[str] = None

    def validate(self) -> None:
        """
        Checks if all arguments and settings of the object are correct.
        """
        pass

    def setup_azureml(self) -> None:
        """
        Execute setup steps that are specific to AzureML.
        """
        if PARENT_RUN_CONTEXT is not None:
            # Set metadata for the run in AzureML if running in a Hyperdrive job.
            run_tags_parent = PARENT_RUN_CONTEXT.get_tags()
            tags_to_copy = [
                "tag",
                "model_name",
                "execution_mode",
                "recovered_from",
                "friendly_name",
                "build_number",
                "build_user",
                RUN_RECOVERY_FROM_ID_KEY_NAME,
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
            logging.info("Setting tags from parent run.")
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
        self.checkpoint_handler.download_recovery_checkpoints_or_weights()

        # Create an AzureML run for logging if running outside AzureML.
        self.create_logger()

        self.container.setup()
        self.container.create_lightning_module_and_store()
        self._has_setup_run = True

        if is_running_in_azure_ml():
            self.setup_azureml()

    def create_logger(self) -> None:
        """
        Create an AzureML run for logging if running outside AzureML. This run will be used for metrics logging
        during both training and inference. We can't rely on the automatically generated run inside the AzureMLLogger
        class because two of those logger objects will be created, so training and inference metrics would be logged
        in different runs.
        """
        if self.container.log_from_vm:
            run = create_aml_run_object(experiment_name=self.container.effective_experiment_name)
            # Display name should already be set when creating the Run object, but in some scenarios this
            # does not happen. Hence, set it again.
            run.display_name = self.container.tag if self.container.tag else None
            self.azureml_run_for_logging = run

    def get_data_module(self) -> LightningDataModule:
        """
        Reads the datamodule that should be used for training or valuation from the container. This must be
        overridden in subclasses.
        """
        raise NotImplementedError()

    def set_trainer_for_inference(self) -> None:
        """Set the runner's PL Trainer object that should be used when running inference on the validation or test set.
        We run inference on a single device because distributed strategies such as DDP use DistributedSampler
        internally, which replicates some samples to make sure all devices have the same batch size in case of
        uneven inputs which biases the results."""
        mlflow_run_id = get_mlflow_run_id_from_trainer(self.trainer)
        self.container.max_num_gpus = self.container.max_num_gpus_inference
        self.trainer, _ = create_lightning_trainer(
            container=self.container,
            num_nodes=1,
            azureml_run_for_logging=self.azureml_run_for_logging,
            mlflow_run_for_logging=mlflow_run_id,
        )

    def init_inference(self) -> None:
        """Prepare the runner for inference on validation set, test set, or a full dataset.
        The following steps are performed:

        1. Get the checkpoint to use for inference. This is either the checkpoint from the last training epoch or the
        one specified in src_checkpoint argument.

        2. Create a new trainer instance for inference. This is necessary because the trainer is created with a single
        device in contrast to training that uses DDP if multiple GPUs are available.

        3. Create a new data module instance for inference to account for any requested changes in the dataloading
        parameters (e.g. batch_size, max_num_workers, etc) as part of on_run_extra_validation_epoch.
        """
        logging.info("Preparing runner for inference.")
        self.inference_checkpoint = str(self.checkpoint_handler.get_checkpoint_to_test())
        self.set_trainer_for_inference()
        self.data_module = self.get_data_module()

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

    def run(self) -> None:
        """
        Run the training or evaluation. This method must be overridden in subclasses.
        """
        pass

    def run_and_cleanup(self, azure_run_info: Optional[AzureRunInfo] = None) -> None:
        """
        Run the training or evaluation via `self.run` and cleanup afterwards.

        :param azure_run_info: When running in AzureML or on a local VM, this contains the paths to the datasets.
            This can be missing when running in unit tests, where the local dataset paths are already populated.
        """
        self.setup(azure_run_info)
        try:
            self.run()
        finally:
            if self.azureml_run_for_logging is not None:
                try:
                    self.azureml_run_for_logging.complete()
                except Exception as ex:
                    logging.error("Failed to complete AzureML run: %s", ex)
