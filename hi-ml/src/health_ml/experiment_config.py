import param
from typing import Optional


class ExperimentConfig(param.Parameterized):
    cluster: str = param.String(doc="The name of the GPU or CPU cluster inside the AzureML workspace, that should "
                                    "execute the job.")
    num_nodes: int = param.Integer(default=1, doc="The number of virtual machines that will be allocated for this"
                                                  "job in AzureML.")
    model: str = param.String(doc="The name of the model to train/test.")
    azureml: bool = param.Boolean(False, doc="If True, submit the executing script to run on AzureML.")
    model_configs_namespace: Optional[str] = param.String(default=None, allow_None=True,
                                                          doc="Optional string representing the  path to an"
                                                              " alternative directory containing the config for"
                                                              " the model specified")
