import param


class ExperimentConfig(param.Parameterized):
    cluster: str = param.String(default="", allow_None=False,
                                doc="The name of the GPU or CPU cluster inside the AzureML workspace"
                                    "that should execute the job.")
    num_nodes: int = param.Integer(default=1, doc="The number of virtual machines that will be allocated for this"
                                                  "job in AzureML.")
    model: str = param.String(doc="The fully qualified name of the model to train/test -e.g."
                                  "mymodule.configs.MyConfig.")
    azureml: bool = param.Boolean(False, doc="If True, submit the executing script to run on AzureML.")
    docker_shm_size: str = param.String("400g",
                                        doc="The shared memory in the Docker image for the AzureML VMs.")
