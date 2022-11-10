from enum import Enum
from pathlib import Path
from typing import Optional
import param


class DebugDDPOptions(Enum):
    OFF = "OFF"
    INFO = "INFO"
    DETAIL = "DETAIL"


DEBUG_DDP_ENV_VAR = "TORCH_DISTRIBUTED_DEBUG"


class ExperimentConfig(param.Parameterized):
    cluster: str = param.String(default="", allow_None=False,
                                doc="The name of the GPU or CPU cluster inside the AzureML workspace"
                                    "that should execute the job. To run on your local machine, omit this argument.")
    num_nodes: int = param.Integer(default=1, doc="The number of virtual machines that will be allocated for this"
                                                  "job in AzureML.")
    model: str = param.String(doc="The fully qualified name of the model to train/test -e.g."
                                  "mymodule.configs.MyConfig.")
    mount_in_azureml: bool = param.Boolean(False,
                                           doc="If False (default), consume datasets in AzureML by downloading at "
                                               "job start. If True, datasets in AzureML are mounted (read on demand "
                                               "over the network). When running outside AzureML, datasets will "
                                               "always be mounted.")
    docker_shm_size: str = param.String("400g",
                                        doc="The shared memory in the Docker image for the AzureML VMs.")
    wait_for_completion: bool = param.Boolean(default=False,
                                              doc="If True, wait for AML Run to complete before proceeding. "
                                                  "If False, submit the run to AML and exit")
    conda_env: Optional[Path] = \
        param.ClassSelector(class_=Path, default=None, allow_None=True,
                            doc="The Conda environment file that should be used when submitting the present run to "
                                "AzureML. If not specified, the environment file in the current folder or one of its "
                                "parents will be used.")
    debug_ddp: DebugDDPOptions = param.ClassSelector(default=DebugDDPOptions.OFF, class_=DebugDDPOptions,
                                                     doc=f"Flag to override the environment var {DEBUG_DDP_ENV_VAR}"
                                                         "that can be used to trigger logging and collective "
                                                         "synchronization checks to ensure all ranks are synchronized "
                                                         "appropriately. Default is `OFF`. It can be set to either "
                                                         "`INFO` or `DETAIL` for different levels of logging. "
                                                         "`DETAIL` may impact the application performance and thus "
                                                         "should only be used when debugging issues")
    strictly_aml_v1: bool = param.Boolean(default=False, doc="If True, use AzureML v1 SDK. If False (default), use "
                                                             "the v2 of the SDK")
