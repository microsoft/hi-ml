#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Wrapper functions for running local Python scripts on Azure ML.
"""
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

from attr import dataclass
from azureml.core import Run
from azureml.core import Workspace
from azureml.data import OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

from health.azure.datasets import _input_dataset_key
from health.azure.datasets import _output_dataset_key
from health.azure.datasets import get_datastore
from health.azure.datasets import get_or_create_dataset

logger = logging.getLogger('health.azure')
logger.setLevel(logging.INFO)

# Re-use the Run object across the package, to avoid repeated and possibly costly calls to create it.
RUN_CONTEXT = Run.get_context()

WORKSPACE_CONFIG_JSON = "config.json"

@dataclass
class WorkspaceConfig:
    workspace_name: str = ""
    subscription_id: str = ""
    resource_group: str = ""


@dataclass
class AzureRunInformation:
    input_datasets: List[Path]
    output_datasets: List[Path]
    run: Run
    is_running_in_azure: bool
    # In Azure, this would be the "outputs" folder. In local runs: "." or create a timestamped folder.
    # The folder that we create here must be added to .amlignore
    output_folder: Path
    log_folder: Path


class DatasetConfig:
    """
    Contains information to use AzureML datasets as inputs or outputs.
    """

    def __init__(self,
                 name: str,
                 datastore: str = "",
                 version: Optional[int] = None,
                 use_mounting: Optional[bool] = None,
                 target_folder: str = "",
                 local_folder: str = ""):
        """
        Creates a new configuration for using an AzureML dataset.
        :param name: The name of the dataset, as it was registered in the AzureML workspace. For output datasets,
        this will be the name given to the newly created dataset.
        :param datastore: The name of the AzureML datastore that holds the dataset. This can be empty if the AzureML
        workspace has only a single datastore, or if the default datastore should be used.
        :param version: The version of the dataset that should be used. This is only used for input datasets.
        If the version is not specified, the latest version will be used.
        :param use_mounting: If True, the dataset will be "mounted", that is, individual files will be read
        or written on-demand over the network. If False, the dataset will be fully downloaded before the job starts,
        respectively fully uploaded at job end for output datasets.
        Defaults: False (downloading) for datasets that are script inputs, True (mounting) for datasets that are script
        outputs.
        :param target_folder: The folder into which the dataset should be downloaded or mounted. If left empty, a
        random folder on /tmp will be chosen.
        :param local_folder: The folder on the local machine at which the dataset is available. This
        is used only for runs outside of AzureML.
        """
        # This class would be a good candidate for a dataclass, but having an explicit constructor makes
        # documentation tools in the editor work nicer.
        name = name.strip()
        if not name:
            raise ValueError("The name of the dataset must be a non-empty string.")
        self.name = name
        self.datastore = datastore
        self.version = version
        self.use_mounting = use_mounting
        self.target_folder = target_folder
        self.local_folder = local_folder

    def to_input_dataset(self,
                         workspace: Workspace,
                         dataset_index: int) -> DatasetConsumptionConfig:
        """
        Creates a configuration for using an AzureML dataset inside of an AzureML run. This will make the AzureML
        dataset with given name available as a named input, using INPUT_0 as the key for dataset index 0.
        :param workspace: The AzureML workspace to read from.
        :param dataset_index: Suffix for using datasets as named inputs, the dataset will be marked INPUT_{index}
        """
        status = f"Dataset {self.name} (index {dataset_index}) will be "
        azureml_dataset = get_or_create_dataset(workspace=workspace,
                                                dataset_name=self.name,
                                                datastore_name=self.datastore)
        named_input = azureml_dataset.as_named_input(_input_dataset_key(index=dataset_index))
        path_on_compute = self.target_folder or None
        use_mounting = False if self.use_mounting is None else self.use_mounting
        if use_mounting:
            status += "mounted at "
            result = named_input.as_mount(path_on_compute)
        else:
            status += "downloaded to "
            result = named_input.as_download(path_on_compute)
        if path_on_compute:
            status += f"{path_on_compute}."
        else:
            status += "a randomly chosen folder."
        logging.info(status)
        return result

    def to_output_dataset(self,
                          workspace: Workspace,
                          dataset_index: int) -> OutputFileDatasetConfig:
        """
        Creates a configuration to write a script output to an AzureML dataset. The name and datastore of this new
        dataset will be taken from the present object.
        :param workspace: The AzureML workspace to read from.
        :param dataset_index: Suffix for using datasets as named inputs, the dataset will be marked OUTPUT_{index}
        :return:
        """
        status = f"Output dDataset {self.name} (index {dataset_index}) will be "
        datastore = get_datastore(workspace, self.datastore)
        dataset = OutputFileDatasetConfig(name=_output_dataset_key(index=dataset_index),
                                          destination=(datastore, self.name + "/"))
        if self.target_folder:
            raise ValueError("Output datasets can't have a target_folder set.")
        use_mounting = True if self.use_mounting is None else self.use_mounting
        if use_mounting:
            status += "mounted and uploaded while the job runs."
            result = dataset.as_mount()
        else:
            status += "uploaded when the job completes."
            result = dataset.as_upload()
        logging.info(status)
        return result


StrOrDatasetConfig = Union[str, DatasetConfig]


def _replace_string_datasets(datasets: List[StrOrDatasetConfig],
                             default_datastore_name: str) -> List[DatasetConfig]:
    """
    Processes a list of input or output datasets. All entries in the list that are strings are turned into
    DatasetConfig objects, using the string as the dataset name, and pointing to the default datastore.
    :param datasets: A list of datasets, each given either as a string or a DatasetConfig object.
    :param default_datastore_name: The datastore to use for all datasets that are only specified via their name.
    :return: A list of DatasetConfig objects, in the same order as the input list.
    """
    return [DatasetConfig(name=d, datastore=default_datastore_name) if isinstance(d, str) else d
            for d in datasets]


def is_running_in_azure(run: Run = RUN_CONTEXT) -> bool:
    """
    Returns True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    :param run: The run to check. If omitted, use the default run in RUN_CONTEXT
    :return: True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    """
    return hasattr(run, 'experiment')


def submit_to_azure_if_needed(
        workspace_config: Optional[WorkspaceConfig],
        workspace_config_path: Optional[Path],
        default_datastore: str = "",
        input_datasets: Optional[List[StrOrDatasetConfig]] = None,
        output_datasets: Optional[List[StrOrDatasetConfig]] = None,
        num_nodes: int = 1,
        # TODO antonsc: Does the root folder option make sense? Clearly it can't be a folder below the folder where
        # the script lives. But would it ever be a folder further up?
        root_folder: Optional[Path] = None) -> AzureRunInformation:
    """
    Submit a folder to Azure, if needed and run it.

    :param workspace_config: Optional workspace config.
    :param workspace_config_file: Optional path to workspace config file.
    :return: None.
    """
    workspace: Workspace = None
    if workspace_config is not None:
        workspace = Workspace.get(
            name=workspace_config.workspace_name,
            subscription_id=workspace_config.subscription_id,
            resource_group=workspace_config.resource_group)
    else:
        workspace = Workspace.from_config(path=workspace_config_path)

    if workspace is None:
        raise ValueError("Unable to get workspace.")

    print(f"Loaded: {workspace.name}")

    input_datasets = _replace_string_datasets(input_datasets, default_datastore_name=default_datastore)
    output_datasets = _replace_string_datasets(output_datasets, default_datastore_name=default_datastore)
    in_azure = is_running_in_azure()
    if in_azure:
        returned_input_datasets = [RUN_CONTEXT.input_datasets[_input_dataset_key(index)] for index in
                                   range(len(input_datasets))]
        returned_output_datasets = [RUN_CONTEXT.output_datasets[_output_dataset_key(index)] for index in
                                    range(len(output_datasets))]
    else:
        returned_input_datasets = [d.local_folder or None for d in input_datasets]
        returned_output_datasets = [d.local_folder or None for d in output_datasets]
    return AzureRunInformation(
        input_datasets=returned_input_datasets,
        output_datasets=returned_output_datasets,
        run=RUN_CONTEXT,
        is_running_in_azure=in_azure,
        output_folder=root_folder / "outputs",
        log_folder=root_folder / "logs"
    )


def main() -> None:
    """
    Handle submit_to_azure if called from the command line.
    """
    parser = ArgumentParser()
    parser.add_argument("-w", "--workspace_name", type=str, required=False,
                        help="Azure ML workspace name")
    parser.add_argument("-s", "--subscription_id", type=str, required=False,
                        help="AzureML subscription id")
    parser.add_argument("-r", "--resource_group", type=str, required=False,
                        help="AzureML resource group")
    parser.add_argument("-p", "--workspace_config_path", type=str, required=False,
                        help="AzureML workspace config file")

    args = parser.parse_args()
    config = WorkspaceConfig(
        workspace_name=args.workspace_name,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group)

    submit_to_azure_if_needed(
        config,
        args.workspace_config_path)


if __name__ == "__main__":
    main()
