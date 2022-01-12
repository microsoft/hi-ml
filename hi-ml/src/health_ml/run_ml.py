#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch.multiprocessing
from pytorch_lightning import seed_everything
from pytorch_lightning.core.datamodule import LightningDataModule

from health_azure import AzureRunInfo
from health_azure.utils import (ENV_OMPI_COMM_WORLD_RANK, RUN_CONTEXT, create_run_recovery_id,
                                PARENT_RUN_CONTEXT, is_running_in_azure_ml)

# from health_ml.deep_learning_config import DeepLearningConfig
from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
# from health_ml.model_config_base import ModelConfigBase
from health_ml.model_trainer import create_lightning_trainer, model_train
from health_ml.utils import fixed_paths
from health_ml.utils.common_utils import (
    CROSSVAL_SPLIT_KEY, ModelExecutionMode, change_working_directory,
    logging_section, RUN_RECOVERY_ID_KEY, EFFECTIVE_RANDOM_SEED_KEY_NAME, RUN_RECOVERY_FROM_ID_KEY_NAME)
from health_ml.utils.lightning_loggers import StoringLogger
from health_ml.utils.type_annotations import PathOrString


def check_dataset_folder_exists(local_dataset: PathOrString) -> Path:
    """
    Checks if a folder with a local dataset exists. If it does exist, return the argument converted to a Path instance.
    If it does not exist, raise a FileNotFoundError.
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

        :param container: The LightningContainer object to use for training.
        :param project_root: Project root. This should only be omitted if calling run_ml from the test suite. Supplying
        it is crucial when using hi-ml as a package or submodule!
        """
        # self.model_config = model_config
        # if container is None:
        #     assert isinstance(model_config, ModelConfigBase), \
        #         "When using a built-in model, the configuration should be an instance of ModelConfigBase"
        # container = LightningContainer(model_config)  # type: ignore
        self.container = container

        self.experiment_config = experiment_config
        # self.azure_config: AzureConfig = azure_config or AzureConfig()
        self.container.num_nodes = self.experiment_config.num_nodes
        self.project_root: Path = project_root or fixed_paths.repository_root_directory()
        # self.post_cross_validation_hook = post_cross_validation_hook
        # self.model_deployment_hook = model_deployment_hook
        self.storing_logger: Optional[StoringLogger] = None
        self._has_setup_run = False

    def setup(self, azure_run_info: Optional[AzureRunInfo] = None) -> None:
        """
        If the present object is using one of the built-in models, create a (fake) container for it
        and call the setup method. It sets the random seeds, and then creates the actual Lightning modules.

        :param azure_run_info: When running in AzureML or on a local VM, this contains the paths to the datasets.
        This can be missing when running in unit tests, where the local dataset paths are already populated.
        """
        if self._has_setup_run:
            return
        if azure_run_info:
            # Set up the paths to the datasets. azure_run_info already has all necessary information, using either
            # the provided local datasets for VM runs, or the AzureML mount points when running in AML.
            # This must happen before container setup because that could already read datasets.
            if len(azure_run_info.input_datasets) > 0:
                input_datasets = azure_run_info.input_datasets
                assert len(input_datasets) > 0
                local_datasets = [check_dataset_folder_exists(input_dataset for input_dataset in input_datasets)]
                self.container.local_datasets = local_datasets
        # Ensure that we use fixed seeds before initializing the PyTorch models
        seed_everything(self.container.get_effective_random_seed())

        self.container.setup()
        self.container.create_lightning_module_and_store()
        self._has_setup_run = True

    @property
    def is_offline_run(self) -> bool:
        """
        Returns True if the present run is outside of AzureML, and False if it is inside of AzureML.

        :return:
        """
        return not is_running_in_azure_ml(RUN_CONTEXT)

    @property
    def config_namespace(self) -> str:
        """
        Returns the namespace of the model configuration object, i.e. return the name of the module in which the
        model configuration object or the lightning container object is defined.
        For models defined as lightning containers, this is the namespace of the container class defining the model.

        :return: the namespace of the model configuraton object
        """
        return self.container.__class__.__module__

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
        new_tags[CROSSVAL_SPLIT_KEY] = str(self.container.cross_validation_split_index)
        new_tags[EFFECTIVE_RANDOM_SEED_KEY_NAME] = str(self.container.get_effective_random_seed())
        RUN_CONTEXT.set_tags(new_tags)

    def run(self) -> None:
        """
        Driver function to run a ML experiment. If an offline cross validation run is requested, then
        this function is recursively called for each cross validation split.
        """
        self.setup()
        # Get the AzureML context in which the script is running
        if not self.is_offline_run and PARENT_RUN_CONTEXT is not None:
            logging.info("Setting tags from parent run.")
            self.set_run_tags_from_parent()
        # do training
        with logging_section("Model training"):
            _, storing_logger = model_train(container=self.container,
                                            num_nodes=self.experiment_config.num_nodes)
            self.storing_logger = storing_logger

        RUN_CONTEXT.log(name="Train epochs", value=self.container.num_epochs)

    def is_normal_run_or_crossval_child_0(self) -> bool:
        """
        Returns True if the present run is a non-crossvalidation run, or child run 0 of a crossvalidation run.
        """
        if self.container.perform_cross_validation:
            return self.container.cross_validation_split_index == 0
        return True

    @staticmethod
    def lightning_data_module_dataloaders(data: LightningDataModule) -> Dict[ModelExecutionMode, Callable]:
        """
        Given a lightning data module, return a dictionary of dataloader for each model execution mode.

        :param data: Lightning data module.
        :return: Data loader for each model execution mode.
        """
        return {
            ModelExecutionMode.TEST: data.test_dataloader,
            ModelExecutionMode.VAL: data.val_dataloader,
            ModelExecutionMode.TRAIN: data.train_dataloader
        }

    def run_inference_for_lightning_models(self, checkpoint_paths: List[Path]) -> None:
        """
        Run inference on the test set for all models that are specified via a LightningContainer.
        :param checkpoint_paths: The path to the checkpoint that should be used for inference.
        """
        if len(checkpoint_paths) != 1:
            raise ValueError(f"This method expects exactly 1 checkpoint for inference, but got {len(checkpoint_paths)}")
        # lightning_model = self.container.model

        # Run Lightning's built-in test procedure if the `test_step` method has been overridden
        logging.info("Running inference via the LightningModule.test_step method")
        # Lightning does not cope with having two calls to .fit or .test in the same script. As a workaround for
        # now, restrict number of GPUs to 1, meaning that it will not start DDP.
        self.container.max_num_gpus = 1
        # Without this, the trainer will think it should still operate in multi-node mode, and wrongly start
        # searching for Horovod
        if ENV_OMPI_COMM_WORLD_RANK in os.environ:
            del os.environ[ENV_OMPI_COMM_WORLD_RANK]
        # From the training setup, torch still thinks that it should run in a distributed manner,
        # and would block on some GPU operations. Hence, clean up distributed training.
        if torch.distributed.is_initialized():  # type: ignore
            torch.distributed.destroy_process_group()  # type: ignore
        trainer, _ = create_lightning_trainer(self.container, num_nodes=1)
        self.container.load_model_checkpoint(checkpoint_path=checkpoint_paths[0])
        # When training models that are not built-in models, we have no guarantee that they write
        # files to the right folder. Best guess is to change the current working directory to where files should go.
        with change_working_directory(self.container.outputs_folder):
            trainer.test(self.container.model,
                         datamodule=self.container.get_data_module())
