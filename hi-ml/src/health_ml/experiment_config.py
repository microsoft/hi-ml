import param

from health_azure.utils import GenericConfig


class ExperimentConfig(GenericConfig):
    cluster: str = param.String(doc="The name of the GPU or CPU cluster inside the AzureML workspace, that should "
                                    "execute the job.")
    num_nodes: int = param.Integer(default=1, doc="The number of virtual machines that will be allocated for this"
                                                  "job in AzureML.")
    model: str = param.String(doc="The name of the model to train/test.")
    azureml: bool = param.Boolean(False, doc="If True, submit the executing script to run on AzureML.")
