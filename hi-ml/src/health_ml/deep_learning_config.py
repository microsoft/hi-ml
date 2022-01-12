#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from enum import Enum, unique
from pathlib import Path
from typing import List, Optional

import param
from param import Parameterized

from health_azure.utils import RUN_CONTEXT, PathOrString, is_running_in_azure_ml

from health_ml.utils import fixed_paths
from health_ml.utils.common_utils import (create_unique_timestamp_id,
                                          DEFAULT_CROSS_VALIDATION_SPLIT_INDEX,
                                          DEFAULT_AML_UPLOAD_DIR, DEFAULT_LOGS_DIR_NAME)
from health_ml.utils.type_annotations import TupleFloat2


@unique
class LRWarmUpType(Enum):
    """
    Supported LR warm up types for model training
    """
    NoWarmUp = "NoWarmUp"
    Linear = "Linear"


@unique
class LRSchedulerType(Enum):
    """
    Supported lr scheduler types for model training
    """
    Exponential = "Exponential"
    Step = "Step"
    Polynomial = "Polynomial"
    Cosine = "Cosine"
    MultiStep = "MultiStep"


@unique
class OptimizerType(Enum):
    """
    Supported optimizers for model training
    """
    Adam = "Adam"
    AMSGrad = "AMSGrad"
    SGD = "SGD"
    RMSprop = "RMSprop"


class ExperimentFolderHandler(Parameterized):
    """High level config to abstract the file system related settings for experiments"""
    outputs_folder: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                               doc="The folder where all training and test outputs should go.")
    logs_folder: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                            doc="The folder for all log files and Tensorboard event files")
    project_root: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                             doc="The root folder for the codebase that triggers the training run.")
    run_folder: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                           doc="The folder that contains outputs and the logs subfolder.")

    @staticmethod
    def create(project_root: Path,
               is_offline_run: bool,
               model_name: str,
               output_to: Optional[str] = None) -> ExperimentFolderHandler:
        """
        Creates a new object that holds output folder configurations. When running inside of AzureML, the output
        folders will be directly under the project root. If not running inside AzureML, a folder with a timestamp
        will be created for all outputs and logs.

        :param project_root: The root folder that contains the code that submitted the present training run.
        When running inside the hi-ml repository, it is the git repo root. When consuming hi-ml as a package,
        this should be the root of the source code that calls the package.
        :param is_offline_run: If true, this is a run outside AzureML. If False, it is inside AzureML.
        :param model_name: The name of the model that is trained. This is used to generate a run-specific output
        folder.
        :param output_to: If provided, the output folders will be created as a subfolder of this argument. If not
        given, the output folders will be created inside of the project root.
        """
        if not project_root.is_absolute():
            raise ValueError(f"The project root is required to be an absolute path, but got {project_root}")

        if is_offline_run or output_to:
            if output_to:
                logging.info(f"All results will be written to the specified output folder {output_to}")
                root = Path(output_to).absolute()
            else:
                logging.info("All results will be written to a subfolder of the project root folder.")
                root = project_root.absolute() / DEFAULT_AML_UPLOAD_DIR
            timestamp = create_unique_timestamp_id()
            run_folder = root / f"{timestamp}_{model_name}"
            outputs_folder = run_folder
            logs_folder = run_folder / DEFAULT_LOGS_DIR_NAME
        else:
            logging.info("Running inside AzureML.")
            logging.info("All results will be written to a subfolder of the project root folder.")
            run_folder = project_root
            outputs_folder = project_root / DEFAULT_AML_UPLOAD_DIR
            logs_folder = project_root / DEFAULT_LOGS_DIR_NAME
        logging.info(f"Run outputs folder: {outputs_folder}")
        logging.info(f"Logs folder: {logs_folder}")
        return ExperimentFolderHandler(
            outputs_folder=outputs_folder,
            logs_folder=logs_folder,
            project_root=project_root,
            run_folder=run_folder
        )


class WorkflowParams(param.Parameterized):
    """
    This class contains all parameters that affect how the whole training and testing workflow is executed.
    """
    random_seed: int = param.Integer(42, doc="The seed to use for all random number generators.")
    num_crossval_splits: int = param.Integer(0, bounds=(0, None),
                                             doc="Number of cross validation splits for k-fold cross "
                                                 "validation")
    crossval_split_index: int = param.Integer(DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, bounds=(-1, None),
                                              doc="The index of the cross validation fold this model is "
                                                  "associated with when performing k-fold cross validation")
    weights_url: List[str] = param.List(default=[], class_=str,
                                        doc="If provided, a set of urls from which checkpoints will be downloaded"
                                            "and used for inference.")
    local_weights_path: List[Path] = param.List(default=[], class_=Path,
                                                doc="A list of checkpoints paths to use for inference, "
                                                    "when the job is running outside Azure.")
    model_id: str = param.String(default="",
                                 doc="A model id string in the form 'model name:version' "
                                     "to use a registered model for inference.")
    regression_test_folder: Optional[Path] = \
        param.ClassSelector(class_=Path, default=None, allow_None=True,
                            doc="A path to a folder that contains a set of files. At the end of training and "
                                "model evaluation, all files given in that folder must be present in the job's output "
                                "folder, and their contents must match exactly. When running in AzureML, you need to "
                                "ensure that this folder is part of the snapshot that gets uploaded. The path should "
                                "be relative to the repository root directory.")

    def validate(self) -> None:
        if sum([bool(param) for param in [self.weights_url, self.local_weights_path, self.model_id]]) > 1:
            raise ValueError("Cannot specify more than one of local_weights_path, weights_url or model_id.")

        if self.model_id:
            if len(self.model_id.split(":")) != 2:
                raise ValueError(
                    f"model_id should be in the form 'model_name:version', got {self.model_id}")

        if self.num_crossval_splits == 1:
            raise ValueError("At least two splits required to perform cross validation, but got "
                             f"{self.num_crossval_splits}. To train without cross validation, set "
                             "num_crossval_splits=0.")
        if 0 < self.num_crossval_splits <= self.crossval_split_index:
            raise ValueError(f"Cross validation split index is out of bounds: {self.crossval_split_index}, "
                             f"which is invalid for CV with {self.num_crossval_splits} splits.")
        elif self.num_crossval_splits == 0 and self.crossval_split_index != -1:
            raise ValueError(f"Cross validation split index must be -1 for a non cross validation run, "
                             f"found num_crossval_splits = {self.num_crossval_splits} "
                             f"and crossval_split_index={self.crossval_split_index}")

    @property
    def is_running_in_aml(self) -> bool:
        """
        Whether the current run is executing inside Azure ML

        :return: True if the run is executing inside Azure ML, or False if outside AzureML.
        """
        return is_running_in_azure_ml(RUN_CONTEXT)

    @property
    def perform_cross_validation(self) -> bool:
        """
        True if cross validation will be be performed as part of the training procedure.

        :return:
        """
        return self.num_crossval_splits > 1

    def get_effective_random_seed(self) -> int:
        """
        Returns the random seed set as part of this configuration. If the configuration corresponds
        to a cross validation split, then the cross validation fold index will be added to the
        set random seed in order to return the effective random seed.

        :return:
        """
        seed = self.random_seed
        if self.perform_cross_validation:
            # offset the random seed based on the cross validation split index so each
            # fold has a different initial random state.
            seed += self.crossval_split_index
        return seed


class DatasetParams(param.Parameterized):
    azure_datasets: List[str] = param.List(default=[], allow_None=False,
                                           doc="If provided, the ID of one or more datasets to use when running in"
                                               " AzureML.This dataset must exist as a folder of the same name in the"
                                               " 'datasets' container in the datasets storage account. This dataset"
                                               " will be mounted and made available at the 'local_dataset' path"
                                               " when running in AzureML.")
    local_datasets: List[str] = param.List(default=[], allow_None=False,
                                           doc="A list of one or more paths to the dataset to use, when training"
                                               " outside of Azure ML.")
    dataset_mountpoints: List[str] = param.List(default=[], allow_None=False,
                                                doc="The path at which the AzureML dataset should be made available "
                                                    "via mounting or downloading. This only affects jobs running in "
                                                    "AzureML. If empty, use a random mount/download point.")

    def validate(self) -> None:
        if not self.azure_datasets and self.local_datasets is None:
            raise ValueError("Either of local_dataset or azure_datasets must be set.")

        if self.dataset_mountpoints and len(self.azure_datasets) != len(self.dataset_mountpoints):
            raise ValueError(f"Expected the number of azure datasets to equal the number of mountpoints, "
                             f"got datasets [{','.join(self.azure_datasets)}] "
                             f"and mountpoints [{','.join(self.dataset_mountpoints)}]")


class OutputParams(param.Parameterized):
    output_to: str = param.String(default="",
                                  doc="If provided, the run outputs will be written to the given folder. If not "
                                      "provided, outputs will go into a subfolder of the project root folder.")
    file_system_config: ExperimentFolderHandler = param.ClassSelector(default=ExperimentFolderHandler(),
                                                                      class_=ExperimentFolderHandler,
                                                                      instantiate=False,
                                                                      doc="File system related configs")
    _model_name: str = param.String("", doc="The human readable name of the model (for example, Liver). This is "
                                            "usually set from the class name.")

    @property
    def model_name(self) -> str:
        """
        Gets the human readable name of the model (e.g., Liver). This is usually set from the class name.
        :return: A model name as a string.
        """
        return self._model_name

    def set_output_to(self, output_to: PathOrString) -> None:
        """
        Adjusts the file system settings in the present object such that all outputs are written to the given folder.
        :param output_to: The absolute path to a folder that should contain the outputs.
        """
        if isinstance(output_to, Path):
            output_to = str(output_to)
        self.output_to = output_to
        self.create_filesystem()

    def create_filesystem(self, project_root: Path = fixed_paths.repository_root_directory()) -> None:
        """
        Creates new file system settings (outputs folder, logs folder) based on the information stored in the
        present object. If any of the folders do not yet exist, they are created.
        :param project_root: The root folder for the codebase that triggers the training run.
        """
        self.file_system_config = ExperimentFolderHandler.create(
            project_root=project_root,
            model_name=self.model_name,
            is_offline_run=not is_running_in_azure_ml(RUN_CONTEXT),
            output_to=self.output_to
        )

    @property
    def outputs_folder(self) -> Path:
        """Gets the full path in which the model outputs should be stored."""
        return self.file_system_config.outputs_folder

    @property
    def logs_folder(self) -> Path:
        """Gets the full path in which the model logs should be stored."""
        return self.file_system_config.logs_folder


class OptimizerParams(param.Parameterized):
    l_rate: float = param.Number(1e-4, doc="The initial learning rate", bounds=(0, None))
    _min_l_rate: float = param.Number(0.0, doc="The minimum learning rate for the Polynomial and Cosine schedulers.",
                                      bounds=(0.0, None))
    l_rate_scheduler: LRSchedulerType = param.ClassSelector(default=LRSchedulerType.Polynomial,
                                                            class_=LRSchedulerType,
                                                            instantiate=False,
                                                            doc="Learning rate decay method (Cosine, Polynomial, "
                                                                "Step, MultiStep or Exponential)")
    l_rate_exponential_gamma: float = param.Number(0.9, doc="Controls the rate of decay for the Exponential "
                                                            "LR scheduler.")
    l_rate_step_gamma: float = param.Number(0.1, doc="Controls the rate of decay for the "
                                                     "Step LR scheduler.")
    l_rate_step_step_size: int = param.Integer(50, bounds=(0, None),
                                               doc="The step size for Step LR scheduler")
    l_rate_multi_step_gamma: float = param.Number(0.1, doc="Controls the rate of decay for the "
                                                           "MultiStep LR scheduler.")
    l_rate_multi_step_milestones: Optional[List[int]] = param.List(None, bounds=(1, None),
                                                                   allow_None=True, class_=int,
                                                                   doc="The milestones for MultiStep decay.")
    l_rate_polynomial_gamma: float = param.Number(1e-4, doc="Controls the rate of decay for the "
                                                            "Polynomial LR scheduler.")
    l_rate_warmup: LRWarmUpType = param.ClassSelector(default=LRWarmUpType.NoWarmUp, class_=LRWarmUpType,
                                                      instantiate=False,
                                                      doc="The type of learning rate warm up to use. "
                                                          "Can be NoWarmUp (default) or Linear.")
    l_rate_warmup_epochs: int = param.Integer(0, bounds=(0, None),
                                              doc="Number of warmup epochs (linear warmup) before the "
                                                  "scheduler starts decaying the learning rate. "
                                                  "For example, if you are using MultiStepLR with "
                                                  "milestones [50, 100, 200] and warmup epochs = 100, warmup "
                                                  "will last for 100 epochs and the first decay of LR "
                                                  "will happen on epoch 150")
    optimizer_type: OptimizerType = param.ClassSelector(default=OptimizerType.Adam, class_=OptimizerType,
                                                        instantiate=False, doc="The optimizer_type to use")
    opt_eps: float = param.Number(1e-4, doc="The epsilon parameter of RMSprop or Adam")
    rms_alpha: float = param.Number(0.9, doc="The alpha parameter of RMSprop")
    adam_betas: TupleFloat2 = param.NumericTuple((0.9, 0.999), length=2,
                                                 doc="The betas parameter of Adam, default is (0.9, 0.999)")
    momentum: float = param.Number(0.6, doc="The momentum parameter of the optimizers")
    weight_decay: float = param.Number(1e-4, doc="The weight decay used to control L2 regularization")

    def validate(self) -> None:
        if len(self.adam_betas) < 2:
            raise ValueError(
                "The adam_betas parameter should be the coefficients used for computing running averages of "
                "gradient and its square")

        if self.l_rate_scheduler == LRSchedulerType.MultiStep:
            if not self.l_rate_multi_step_milestones:
                raise ValueError("Must specify l_rate_multi_step_milestones to use LR scheduler MultiStep")
            if sorted(set(self.l_rate_multi_step_milestones)) != self.l_rate_multi_step_milestones:
                raise ValueError("l_rate_multi_step_milestones must be a strictly increasing list")
            if self.l_rate_multi_step_milestones[0] <= 0:
                raise ValueError("l_rate_multi_step_milestones cannot be negative or 0.")

    @property
    def min_l_rate(self) -> float:
        return self._min_l_rate

    @min_l_rate.setter
    def min_l_rate(self, value: float) -> None:
        if value > self.l_rate:
            raise ValueError("l_rate must be >= min_l_rate, found: {}, {}".format(self.l_rate, value))
        self._min_l_rate = value


class TrainerParams(param.Parameterized):
    max_epochs: int = param.Integer(100, bounds=(1, None), doc="Number of epochs to train.")
    recovery_checkpoint_save_interval: int = param.Integer(10, bounds=(0, None),
                                                           doc="Save epoch checkpoints when epoch number is a multiple "
                                                               "of recovery_checkpoint_save_interval. The intended use "
                                                               "is to allow restore training from failed runs.")
    recovery_checkpoints_save_last_k: int = param.Integer(default=1, bounds=(-1, None),
                                                          doc="Number of recovery checkpoints to keep. Recovery "
                                                              "checkpoints will be stored as recovery_epoch:{"
                                                              "epoch}.ckpt. If set to -1 keep all recovery "
                                                              "checkpoints.")
    detect_anomaly: bool = param.Boolean(False, doc="If true, test gradients for anomalies (NaN or Inf) during "
                                                    "training.")
    use_mixed_precision: bool = param.Boolean(False, doc="If true, mixed precision training is activated during "
                                                         "training.")
    max_num_gpus: int = param.Integer(default=-1, doc="The maximum number of GPUS to use. If set to a value < 0, use"
                                                      "all available GPUs. In distributed training, this is the "
                                                      "maximum number of GPUs per node.")
    pl_progress_bar_refresh_rate: Optional[int] = \
        param.Integer(default=None,
                      doc="PyTorch Lightning trainer flag 'progress_bar_refresh_rate': How often to refresh progress "
                          "bar (in steps). Value 0 disables progress bar. Value None chooses automatically.")
    pl_num_sanity_val_steps: int = \
        param.Integer(default=0,
                      doc="PyTorch Lightning trainer flag 'num_sanity_val_steps': Number of validation "
                          "steps to run before training, to identify possible problems")
    pl_deterministic: bool = \
        param.Boolean(default=False,
                      doc="Controls the PyTorch Lightning trainer flags 'deterministic' and 'benchmark'. If "
                          "'pl_deterministic' is True, results are perfectly reproducible. If False, they are not, but "
                          "you may see training speed increases.")
    pl_find_unused_parameters: bool = \
        param.Boolean(default=False,
                      doc="Controls the PyTorch Lightning flag 'find_unused_parameters' for the DDP plugin. "
                          "Setting it to True comes with a performance hit.")
    pl_limit_train_batches: Optional[int] = \
        param.Integer(default=None,
                      doc="PyTorch Lightning trainer flag 'limit_train_batches': Limit the training dataset to the "
                          "given number of batches.")
    pl_limit_val_batches: Optional[int] = \
        param.Integer(default=None,
                      doc="PyTorch Lightning trainer flag 'limit_val_batches': Limit the validation dataset to the "
                          "given number of batches.")
    pl_profiler: Optional[str] = \
        param.String(default=None,
                     doc="The value to use for the 'profiler' argument for the Lightning trainer. "
                         "Set to either 'simple', 'advanced', or 'pytorch'")
    monitor_gpu: bool = param.Boolean(default=False,
                                      doc="If True, add the GPUStatsMonitor callback to the Lightning trainer object. "
                                          "This will write GPU utilization metrics every 50 batches by default.")
    monitor_loading: bool = param.Boolean(default=True,
                                          doc="If True, add the BatchTimeCallback callback to the Lightning trainer "
                                              "object. This will monitor how long individual batches take to load.")

    @property
    def use_gpu(self) -> bool:
        """
        Returns True if a GPU is available, and the self.max_num_gpus flag allows it to be used. Returns False
        otherwise (i.e., if there is no GPU available, or self.max_num_gpus==0)
        """
        if self.max_num_gpus == 0:
            return False
        from health_ml.utils.common_utils import is_gpu_available
        return is_gpu_available()

    def num_gpus_per_node(self) -> int:
        """
        Computes the number of gpus to use for each node: either the number of gpus available on the device
        or restrict it to max_num_gpu, whichever is smaller. Returns 0 if running on a CPU device.
        """
        import torch
        available_gpus = torch.cuda.device_count()  # type: ignore
        num_gpus = available_gpus if self.use_gpu else 0
        message_suffix = "" if self.use_gpu else ", but not using them because use_gpu == False"
        logging.info(f"Number of available GPUs: {available_gpus}{message_suffix}")
        if 0 <= self.max_num_gpus < num_gpus:
            num_gpus = self.max_num_gpus
            logging.info(f"Restricting the number of GPUs to {num_gpus}")
        elif self.max_num_gpus > num_gpus:
            logging.warning(f"You requested max_num_gpus {self.max_num_gpus} but there are only {num_gpus} available.")
        return num_gpus
