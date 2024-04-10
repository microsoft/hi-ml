from enum import Enum
from pathlib import Path
from typing import Optional
import param


class DebugDDPOptions(Enum):
    OFF = "OFF"
    INFO = "INFO"
    DETAIL = "DETAIL"


class RunnerMode(Enum):
    TRAIN = "train"
    EVAL_FULL = "eval_full"


class LogLevel(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    WARN = "WARN"
    INFO = "INFO"
    DEBUG = "DEBUG"


DEBUG_DDP_ENV_VAR = "TORCH_DISTRIBUTED_DEBUG"


class ExperimentConfig(param.Parameterized):
    cluster: str = param.String(
        default="",
        allow_None=False,
        doc="The name of the GPU or CPU cluster inside the AzureML workspace"
        "that should execute the job. To run on your local machine, omit this argument.",
    )
    num_nodes: int = param.Integer(
        default=1, doc="The number of virtual machines that will be allocated for this job in AzureML."
    )
    model: str = param.String(
        doc="The fully or partially qualified name of the model to train/test -e.g. mymodule.configs.MyConfig."
    )
    model_variant: str = param.String(
        default="",
        doc="The name of the model variant to choose, by calling the 'set_model_variant' method. Model variants "
        "are selected before applying commandline overrides of parameters",
    )
    mount_in_azureml: bool = param.Boolean(
        False,
        doc="If False (default), consume datasets in AzureML by downloading at "
        "job start. If True, datasets in AzureML are mounted (read on demand "
        "over the network). When running outside AzureML, datasets will "
        "always be mounted.",
    )
    docker_shm_size: str = param.String("400g", doc="The shared memory in the Docker image for the AzureML VMs.")
    wait_for_completion: bool = param.Boolean(
        default=False,
        doc="If True, wait for AML Run to complete before proceeding. If False, submit the run to AML and exit",
    )
    conda_env: Optional[Path] = param.ClassSelector(
        class_=Path,
        default=None,
        allow_None=True,
        doc="The Conda environment file that should be used when submitting the present run to "
        "AzureML. If not specified, the environment file in the current folder or one of its "
        "parents will be used.",
    )
    debug_ddp: DebugDDPOptions = param.ClassSelector(
        default=DebugDDPOptions.OFF,
        class_=DebugDDPOptions,
        doc=f"Flag to override the environment var {DEBUG_DDP_ENV_VAR}"
        "that can be used to trigger logging and collective "
        "synchronization checks to ensure all ranks are synchronized "
        "appropriately. Default is `OFF`. It can be set to either "
        "`INFO` or `DETAIL` for different levels of logging. "
        "`DETAIL` may impact the application performance and thus "
        "should only be used when debugging issues",
    )
    strictly_aml_v1: bool = param.Boolean(
        default=False, doc="If True, use AzureML v1 SDK. If False (default), use the v2 of the SDK"
    )
    identity_based_auth: bool = param.Boolean(
        default=False,
        doc="If True, use identity based authentication to access blob storage data via datastores. If False (default),"
        " then use SAS tokens / account keys if available. Only supported when using v2 sdk.",
    )
    workspace_config_path: Optional[Path] = param.ClassSelector(
        class_=Path,
        default=None,
        allow_None=True,
        doc="The path to the AzureML workspace configuration file. If not specified, the "
        "configuration file in the current folder or one of its parents will be used.",
    )
    max_run_duration: str = param.String(
        default="",
        doc="The maximum runtime that is allowed for this job in AzureML. This is given as a floating"
        "point number with a string suffix s, m, h, d for seconds, minutes, hours, day. Examples: '3.5h', '2d'",
    )
    mode: RunnerMode = param.ClassSelector(
        class_=RunnerMode,
        default=RunnerMode.TRAIN,
        doc=f"The mode to run the experiment in. Can be one of '{RunnerMode.TRAIN}' (training and evaluation on the "
        f"test set), or '{RunnerMode.EVAL_FULL}' for evaluation on the full dataset specified by the "
        "'get_eval_data_module' method of the container.",
    )
    log_level: Optional[RunnerMode] = param.ClassSelector(
        class_=LogLevel,
        default=None,
        doc=f"The log level to use. Can be one of {list(map(str, LogLevel))}",
    )

    @property
    def submit_to_azure_ml(self) -> bool:
        """Returns True if the experiment should be submitted to AzureML, False if it should be run locally.

        :return: True if the experiment should be submitted to AzureML, False if it should be run locally."""
        return self.cluster != ""
