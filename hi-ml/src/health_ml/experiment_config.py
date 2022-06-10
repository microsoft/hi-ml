from pathlib import Path
from typing import Optional
import param


class ExperimentConfig(param.Parameterized):
    cluster: str = param.String(default="", allow_None=False,
                                doc="The name of the GPU or CPU cluster inside the AzureML workspace"
                                    "that should execute the job. To run on your local machine, omit this argument.")
    num_nodes: int = param.Integer(default=1, doc="The number of virtual machines that will be allocated for this"
                                                  "job in AzureML.")
    model: str = param.String(doc="The fully qualified name of the model to train/test -e.g."
                                  "mymodule.configs.MyConfig.")
    tag: str = param.String(doc="A string that will be used as the display name of the run in AzureML.")
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
