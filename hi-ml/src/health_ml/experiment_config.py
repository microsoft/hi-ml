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
    mount_in_azureml: bool = param.Boolean(False,
                                           doc="If False (default), consume datasets in AzureML by downloading at "
                                               "job start. If True, datasets in AzureML are mounted (read on demand "
                                               "over the network). When running outside AzureML, datasets will "
                                               "always be mounted.")
