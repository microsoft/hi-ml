#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
import os
import param
from enum import Enum, unique
from param import Parameterized
from pathlib import Path
from typing import Any, Dict, List, Optional

from azureml.train.hyperdrive import HyperDriveConfig

from health_azure import create_crossval_hyperdrive_config
from health_azure.himl import (create_grid_hyperdrive_config, create_crossval_hyperparam_args_v2,
                               create_grid_hyperparam_args_v2)
from health_azure.amulet import (ENV_AMLT_PROJECT_NAME, ENV_AMLT_INPUT_OUTPUT,
                                 ENV_AMLT_SNAPSHOT_DIR, ENV_AMLT_AZ_BATCHAI_DIR,
                                 is_amulet_job, get_amulet_aml_working_dir)
from health_azure.utils import (RUN_CONTEXT, PathOrString, is_global_rank_zero, is_running_in_azure_ml)
from health_ml.utils import fixed_paths
from health_ml.utils.checkpoint_utils import CheckpointParser
from health_ml.utils.common_utils import (CHECKPOINT_FOLDER,
                                          create_unique_timestamp_id,
                                          DEFAULT_AML_UPLOAD_DIR,
                                          DEFAULT_LOGS_DIR_NAME)
from health_ml.utils.type_annotations import IntOrFloat, TupleFloat2


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
               output_to: Optional[Path] = None) -> ExperimentFolderHandler:
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
                root = Path(output_to).resolve()
            else:
                logging.info("All results will be written to a subfolder of the project root folder.")
                root = project_root.resolve() / DEFAULT_AML_UPLOAD_DIR
            if is_global_rank_zero():
                timestamp = create_unique_timestamp_id()
                run_folder = root / f"{timestamp}_{model_name}"
            else:
                # Handle the case where there are multiple DDP threads on the same machine outside AML.
                # Each child process will be started with the current working directory set to be the output
                # folder of the rank 0 process. We want all other process to write to that same folder.
                run_folder = Path.cwd()
            outputs_folder = run_folder
            logs_folder = run_folder / DEFAULT_LOGS_DIR_NAME
        else:
            logging.info("Running inside AzureML.")
            logging.info("All results will be written to a subfolder of the project root folder.")
            if not is_amulet_job():
                run_folder = project_root
                outputs_folder = project_root / DEFAULT_AML_UPLOAD_DIR
                logs_folder = project_root / DEFAULT_LOGS_DIR_NAME
            else:
                # Job submitted via Amulet
                amlt_root_folder = Path(os.environ[ENV_AMLT_INPUT_OUTPUT])
                project_name = os.environ[ENV_AMLT_PROJECT_NAME]
                snapshot_dir = get_amulet_aml_working_dir()
                assert snapshot_dir, \
                    f"Either {ENV_AMLT_SNAPSHOT_DIR} or {ENV_AMLT_AZ_BATCHAI_DIR} must exist in env vars"
                print(f"Found the following environment variables set by Amulet: "
                      f"AZURE_ML_INPUT_OUTPUT: {amlt_root_folder}, AZUREML_ARM_PROJECT_NAME: {project_name}")
                run_id = RUN_CONTEXT.id
                run_folder = amlt_root_folder / "projects" / project_name / "amlt-code" / run_id
                outputs_folder = snapshot_dir / DEFAULT_AML_UPLOAD_DIR
                logs_folder = snapshot_dir / DEFAULT_LOGS_DIR_NAME

        logging.info(f"Run outputs folder: {outputs_folder}")
        logging.info(f"Logs folder: {logs_folder}")
        logging.info(f"Run root directory: {run_folder}")
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
    src_checkpoint: CheckpointParser = param.ClassSelector(class_=CheckpointParser, default=None,
                                                           instantiate=False, doc=CheckpointParser.DOC)
    crossval_count: int = param.Integer(default=1, bounds=(0, None),
                                        doc="The number of splits to use when doing cross-validation. "
                                            "Use 1 to disable cross-validation")
    crossval_index: int = param.Integer(default=0, bounds=(0, None),
                                        doc="When doing cross validation, this is the index of the current "
                                            "split. Valid values: 0 .. (crossval_count -1)")
    hyperdrive: bool = param.Boolean(False,
                                     doc="If True, use the Hyperdrive configuration specified in the "
                                         "LightningContainer to run hyperparameter tuning. If False, just "
                                         "run a plain single training job. This cannot be combined with "
                                         "the flags --different_seeds or --crossval_count.")
    regression_test_folder: Optional[Path] = \
        param.ClassSelector(class_=Path, default=None, allow_None=True,
                            doc="A path to a folder that contains a set of files. At the end of training and "
                                "model evaluation, all files given in that folder must be present in the job's output "
                                "folder, and their contents must match exactly. When running in AzureML, you need to "
                                "ensure that this folder is part of the snapshot that gets uploaded. The path should "
                                "be relative to the repository root directory.")
    regression_test_csv_tolerance: float = \
        param.Number(default=0.0, allow_None=False,
                     doc="When comparing CSV files during regression tests, use this value as the maximum allowed "
                         "relative difference of actual and expected results. Default: 0.0 (must match exactly)")
    regression_metrics: str = param.String(default=None, doc="A list of names of metrics to compare")
    run_inference_only: bool = param.Boolean(False, doc="If True, run only inference and skip training after loading"
                                                        "model weights from the specified checkpoint in "
                                                        "`src_checkpoint` flag. Inference is run on both validation "
                                                        "and test sets. If False, run training and inference.")
    resume_training: bool = param.Boolean(False, doc="If True, resume training from the src_checkpoint.")
    tag: str = param.String(doc="A string that will be used as the display name of the run in AzureML.")
    experiment: str = param.String(default="", doc="The name of the AzureML experiment to use for this run. If not "
                                   "provided, the name of the model class will be used.")
    log_from_vm: bool = param.Boolean(False, doc="If True, a training run outside AzureML will still log its "
                                      "metrics to AzureML. Both intermediate validation metrics and final test results"
                                      "will be recorded. You need to have an AzureML workspace config.json file "
                                      "and will be asked for interactive authentication.")
    different_seeds: int = param.Integer(default=0, bounds=(0, None),
                                         doc="If > 0, run the same training job multiple times with different seeds. "
                                         "This uses AzureML hyperdrive to run multiple jobs in parallel, and hence "
                                         "cannot be used when running outside AzureML. "
                                         "This cannot be combined with the --hyperdrive or the --crossval_count flags.")

    CROSSVAL_INDEX_ARG_NAME = "crossval_index"
    CROSSVAL_COUNT_ARG_NAME = "crossval_count"
    RANDOM_SEED_ARG_NAME = "random_seed"

    def validate(self) -> None:
        if self.crossval_count > 1:
            if not (0 <= self.crossval_index < self.crossval_count):
                raise ValueError(f"Attribute crossval_index out of bounds (crossval_count = {self.crossval_count})")

        if self.run_inference_only and not self.src_checkpoint:
            raise ValueError(f"Cannot run inference without a src_checkpoint. {CheckpointParser.INFO_MESSAGE}")
        if self.resume_training and not self.src_checkpoint:
            raise ValueError(f"Cannot resume training without a src_checkpoint. {CheckpointParser.INFO_MESSAGE}")

    @property
    def is_running_in_aml(self) -> bool:
        """
        Whether the current run is executing inside Azure ML

        :return: True if the run is executing inside Azure ML, or False if outside AzureML.
        """
        return is_running_in_azure_ml(RUN_CONTEXT)

    def get_effective_random_seed(self) -> int:
        """
        Returns the random seed set as part of this configuration. If the configuration corresponds
        to a cross validation split, then the cross validation fold index will be added to the
        set random seed in order to return the effective random seed.
        :return:
        """
        seed = self.random_seed
        if self.is_crossvalidation_enabled:
            # Offset the random seed based on the cross validation split index so each
            # fold has a different initial random state. Cross validation index 0 will have
            # a different seed from a non cross validation run.
            seed += self.crossval_index + 1
        return seed

    @property
    def is_crossvalidation_enabled(self) -> bool:
        """
        Returns True if the present parameters indicate that cross-validation should be used.
        """
        return self.crossval_count > 1

    def get_crossval_hyperdrive_config(self) -> HyperDriveConfig:
        # For crossvalidation, the name of the metric to monitor does not matter because no early termination or such
        # is specified.
        return create_crossval_hyperdrive_config(num_splits=self.crossval_count,
                                                 cross_val_index_arg_name=self.CROSSVAL_INDEX_ARG_NAME,
                                                 metric_name="val/loss"
                                                 )

    def get_different_seeds_hyperdrive_config(self) -> HyperDriveConfig:
        """Returns a configuration object for AzureML Hyperdrive that varies the random seed for each run."""
        return create_grid_hyperdrive_config(values=list(map(str, range(self.different_seeds))),
                                             argument_name=self.RANDOM_SEED_ARG_NAME,
                                             metric_name="val/loss"
                                             )

    def get_crossval_hyperparam_args_v2(self) -> Dict[str, Any]:
        """
        Wrapper function to create hyperparameter search arguments specifically for running cross validation
        with AML SDK v2

        :return: A dictionary of hyperparameter search arguments and values.
        """
        return create_crossval_hyperparam_args_v2(num_splits=self.crossval_count,
                                                  cross_val_index_arg_name=self.CROSSVAL_INDEX_ARG_NAME,
                                                  metric_name="val/loss")

    def get_grid_hyperparam_args_v2(self) -> Dict[str, Any]:
        """
        Wrapper function to create hyperparameter search arguments specifically for running grid search
        with AML SDK v2

        :return: A dictionary of hyperparameter search arguments and values.
        """
        return create_grid_hyperparam_args_v2(values=list(map(str, range(self.different_seeds))),
                                              argument_name=self.RANDOM_SEED_ARG_NAME,
                                              metric_name="val/loss")


class DatasetParams(param.Parameterized):
    datastore: str = param.String(default="", doc="Datastore to look for data in")
    azure_datasets: List[str] = param.List(default=[], class_=str,
                                           doc="If provided, the ID of one or more datasets to use when running in"
                                               " AzureML. This dataset must exist as a folder of the same name "
                                               "in the 'datasets' container in the datasets storage account. This "
                                               "dataset will be mounted and made available at the 'local_dataset' "
                                               "path when running in AzureML.")
    local_datasets: List[Path] = param.List(default=[], class_=Path,
                                            doc="A list of one or more paths to the dataset to use, when training"
                                                " outside of Azure ML.")
    dataset_mountpoints: List[Path] = param.List(default=[], class_=Path,
                                                 doc="The path at which the AzureML dataset should be made "
                                                     "available via mounting or downloading. This only affects "
                                                     "jobs running in AzureML. If empty, use a random "
                                                     "mount/download point.")

    def validate(self) -> None:
        if (not self.azure_datasets) and (not self.local_datasets):
            raise ValueError("Either local_datasets or azure_datasets must be set.")

        if self.dataset_mountpoints and len(self.azure_datasets) != len(self.dataset_mountpoints):
            raise ValueError(f"Expected the number of azure datasets to equal the number of mountpoints, "
                             f"got datasets [{','.join(self.azure_datasets)}] "
                             f"and mountpoints [{','.join([str(m) for m in self.dataset_mountpoints])}]")


class OutputParams(param.Parameterized):
    output_to: Optional[Path] = param.ClassSelector(class_=Path, default=None,
                                                    doc="If provided, the run outputs will be written to the given "
                                                        "folder. If not provided, outputs will go into a subfolder "
                                                        "of the project root folder.")
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
        Adjusts the file system settings in the present object such that all outputs are written to the given
        folder.

        :param output_to: The absolute path to a folder that should contain the outputs.
        """
        self.output_to = Path(output_to)
        self.create_filesystem(project_root=fixed_paths.repository_root_directory())

    def create_filesystem(self, project_root: Path) -> None:
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

    @property
    def checkpoint_folder(self) -> Path:
        """Gets the full path in which the model checkpoints should be stored during training."""
        return self.outputs_folder / CHECKPOINT_FOLDER


class OptimizerParams(param.Parameterized):
    l_rate: float = param.Number(1e-4, doc="The initial learning rate", bounds=(0, None))
    _min_l_rate: float = param.Number(0.0,
                                      doc="The minimum learning rate for the Polynomial and Cosine schedulers.",
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
    autosave_every_n_val_epochs: int = param.Integer(1, bounds=(0, None),
                                                     doc="Save epoch checkpoints every N validation epochs. "
                                                         "If pl_check_val_every_n_epoch > 1, this means that "
                                                         "checkpoints are saved every "
                                                         "N * pl_check_val_every_n_epoch training epochs.")
    detect_anomaly: bool = param.Boolean(False,
                                         doc="If true, test gradients for anomalies (NaN or Inf) during training.")
    use_mixed_precision: bool = \
        param.Boolean(True,
                      doc="If True, use float16 precision (Native Adaptive Mixed Precision) during training. "
                          "If False, use float32.")
    max_num_gpus: int = param.Integer(default=-1,
                                      doc="The maximum number of GPUS to use. If set to a value < 0, use"
                                          "all available GPUs. In distributed training, this is the "
                                          "maximum number of GPUs per node.")
    pl_progress_bar_refresh_rate: Optional[int] = \
        param.Integer(default=None,
                      doc="PyTorch Lightning trainer flag 'progress_bar_refresh_rate': How often to refresh "
                          "progress bar (in steps). Value 0 disables progress bar. If None choose, automatically.")
    pl_num_sanity_val_steps: int = \
        param.Integer(default=0,
                      doc="PyTorch Lightning trainer flag 'num_sanity_val_steps': Number of validation "
                          "steps to run before training, to identify possible problems")
    pl_deterministic: bool = \
        param.Boolean(default=False,
                      doc="Controls the PyTorch Lightning trainer flags 'deterministic' and 'benchmark'. If "
                          "'pl_deterministic' is True, results are perfectly reproducible. If False, they are not, "
                          "but you may see training speed increases.")
    pl_find_unused_parameters: bool = \
        param.Boolean(default=False,
                      doc="Controls the PyTorch Lightning flag 'find_unused_parameters' for the DDP plugin. "
                          "Setting it to True comes with a performance hit.")
    pl_limit_train_batches: Optional[IntOrFloat] = \
        param.Number(default=None,
                     doc="PyTorch Lightning trainer flag 'limit_train_batches': Limit the training dataset to the "
                         "given number of batches if integer, or proportion of training dataset if float.")
    pl_limit_val_batches: Optional[IntOrFloat] = \
        param.Number(default=None,
                     doc="PyTorch Lightning trainer flag 'limit_val_batches': Limit the validation dataset to the "
                         "given number of batches if integer, or proportion of validation dataset if float.")
    pl_limit_test_batches: Optional[IntOrFloat] = \
        param.Number(default=None,
                     doc="PyTorch Lightning trainer flag 'limit_test_batches': Limit the test dataset to the "
                         "given number of batches if integer, or proportion of test dataset if float.")
    pl_fast_dev_run: Optional[int] = \
        param.Integer(default=0,
                      doc="PyTorch Lightning trainer flag 'fast_dev_run': Runs n if set to 'n' batch(es) of train, val"
                          "and test. Default to 0 to use all train, val and test batches available. Setting "
                          "pl_fast_dev_run to n > 0 overrides pl_limit_{train, val, test}_batches to the same value n."
                          "Additionally, by setting this argument, ALL (tuner, checkpoint callbacks, early stopping "
                          "callbacks, loggers and loggger callbacks) will be disabled and run for only a single epoch."
                          "This must be used only for debbuging purposes.")
    pl_profiler: Optional[str] = \
        param.String(default=None,
                     doc="The value to use for the 'profiler' argument for the Lightning trainer. "
                         "Set to either 'simple', 'advanced', or 'pytorch'")
    pl_sync_batchnorm: bool = param.Boolean(default=True,
                                            doc="PyTorch Lightning trainer flag 'sync_batchnorm': If True, "
                                            "synchronize batchnorm across all GPUs when running in ddp mode."
                                            "If False, batchnorm is not synchronized.")
    monitor_gpu: bool = param.Boolean(default=False,
                                      doc="If True, add the GPUStatsMonitor callback to the Lightning trainer object. "
                                          "This will write GPU utilization metrics every 50 batches by default.")
    monitor_loading: bool = param.Boolean(default=False,
                                          doc="If True, add the BatchTimeCallback callback to the Lightning trainer "
                                              "object. This will monitor how long individual batches take to load.")
    monitor_training: bool = param.Boolean(default=False,
                                           doc="If True, add the TrainingDiagnosisCallback to the Lightning trainer "
                                               "object. This will monitor when training, validation and test starts "
                                               "and ends and also intermediate steps.")
    run_extra_val_epoch: bool = param.Boolean(default=False,
                                              doc="If True, run an additional validation epoch at the end of training "
                                              "to produce plots outputs on the validation set. This is to reduce "
                                              "any validation overheads during training time and produce "
                                              "additional time or memory consuming outputs only once after "
                                              "training is finished on the validation set.")
    pl_accumulate_grad_batches: int = param.Integer(default=1,
                                                    doc="The number of batches over which gradients are accumulated, "
                                                    "before a parameter update is done.")
    pl_log_every_n_steps: int = param.Integer(default=50,
                                              doc="PyTorch Lightning trainer flag 'log_every_n_steps': How often to"
                                              "log within steps. Default to 50.")
    pl_replace_sampler_ddp: bool = param.Boolean(default=True,
                                                 doc="PyTorch Lightning trainer flag 'replace_sampler_ddp' that "
                                                 "sets the sampler for distributed training with shuffle=True during "
                                                 "training and shuffle=False during validation. Default to True. Set to"
                                                 "False to set your own sampler.")

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
            logging.warning(
                f"You requested max_num_gpus {self.max_num_gpus} but there are only {num_gpus} available.")
        return num_gpus
