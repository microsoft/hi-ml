#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from azureml.core import Dataset, Datastore, Workspace
from azureml.data import FileDataset, OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.dataprep.fuse.daemon import MountContext

from health_azure.utils import PathOrString, get_workspace


def get_datastore(workspace: Workspace, datastore_name: str) -> Datastore:
    """
    Retrieves a datastore of a given name from an AzureML workspace. The datastore_name argument can be omitted if
    the workspace only contains a single datastore. Raises a ValueError if there is no datastore of the given name.

    :param workspace: The AzureML workspace to read from.
    :param datastore_name: The name of the datastore to retrieve.
    :return: An AzureML datastore.
    """
    datastores = workspace.datastores
    existing_stores = list(datastores.keys())
    if not datastore_name:
        # First check if there is only one datastore, which is then obviously unique.
        # Only then try to use the default datastore, because there may not be a default set.
        if len(existing_stores) == 1:
            return datastores[existing_stores[0]]
        datastore = workspace.get_default_datastore()
        logging.info(f"Using the workspace default datastore {datastore.name} to access datasets.")
        return datastore
    if datastore_name in datastores:
        return datastores[datastore_name]
    raise ValueError(f"Datastore \"{datastore_name}\" was not found in the \"{workspace.name}\" workspace. "
                     f"Existing datastores: {existing_stores}")


def get_or_create_dataset(workspace: Workspace, datastore_name: str, dataset_name: str) -> FileDataset:
    """
    Looks in the AzureML datastore for a dataset of the given name. If there is no such dataset, a dataset is
    created and registered, assuming that the files are in a folder that has the same name as the dataset.
    For example, if dataset_name is 'foo', then the 'foo' dataset should be pointing to the folder
    <container_root>/datasets/dataset_name/
    """
    if not dataset_name:
        raise ValueError("No dataset name provided.")
    try:
        logging.info(f"Trying to retrieve AzureML Dataset '{dataset_name}'")
        azureml_dataset = Dataset.get_by_name(workspace, name=dataset_name)
        logging.info("Dataset found.")
    except Exception:
        logging.info(f"Retrieving datastore '{datastore_name}' from AzureML workspace")
        datastore = get_datastore(workspace, datastore_name)
        logging.info(f"Creating a new dataset from data in folder '{dataset_name}' in the datastore")
        # Ensure that there is a / at the end of the file path, otherwise folder that share a prefix could create
        # trouble (for example, folders foo and foo_bar exist, and I'm trying to create a dataset from "foo")
        azureml_dataset = Dataset.File.from_files(path=(datastore, dataset_name + "/"))
        logging.info("Registering the dataset for future use.")
        azureml_dataset.register(workspace, name=dataset_name)
    return azureml_dataset


def _input_dataset_key(index: int) -> str:
    return f"INPUT_{index}"


def _output_dataset_key(index: int) -> str:
    return f"OUTPUT_{index}"


class DatasetConfig:
    """
    Contains information to use AzureML datasets as inputs or outputs.
    """

    def __init__(self,
                 name: str,
                 datastore: str = "",
                 version: Optional[int] = None,
                 use_mounting: Optional[bool] = None,
                 target_folder: Optional[PathOrString] = None,
                 local_folder: Optional[PathOrString] = None):
        """
        :param name: The name of the dataset, as it was registered in the AzureML workspace. For output datasets,
            this will be the name given to the newly created dataset.
        :param datastore: The name of the AzureML datastore that holds the dataset. This can be empty if the AzureML
            workspace has only a single datastore, or if the default datastore should be used.
        :param version: The version of the dataset that should be used. This is only used for input datasets.
            If the version is not specified, the latest version will be used.
        :param use_mounting: If True, the dataset will be "mounted", that is, individual files will be read
            or written on-demand over the network. If False, the dataset will be fully downloaded before the job starts,
            respectively fully uploaded at job end for output datasets.
            Defaults: False (downloading) for datasets that are script inputs, True (mounting) for datasets that are
            script outputs.
        :param target_folder: The folder into which the dataset should be downloaded or mounted. If left empty, a
            random folder on /tmp will be chosen. Do NOT use "." as the target_folder.
        :param local_folder: The folder on the local machine at which the dataset is available. This
            is used only for runs outside of AzureML. If this is empty then the target_folder will be used to
            mount or download the dataset.
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
        # If target_folder is "" then convert to None
        self.target_folder = Path(target_folder) if target_folder else None
        if str(self.target_folder) == ".":
            raise ValueError("Can't mount or download a dataset to the current working directory.")
        self.local_folder = Path(local_folder) if local_folder else None

    def to_input_dataset_local(self, workspace: Optional[Workspace]) -> Tuple[Path, Optional[MountContext]]:
        """
        Return a local path to the dataset when outside of an AzureML run.
        If local_folder is supplied, then this is assumed to be a local dataset, and this is returned.
        Otherwise the dataset is mounted or downloaded to either the target folder or a temporary folder and that is
        returned.

        :param workspace: The AzureML workspace to read from.
        :return: Tuple of (path to dataset, optional mountcontext)
        """
        status = f"Dataset '{self.name}' will be "

        if self.local_folder is not None:
            status += f"obtained from local folder {str(self.local_folder)}"
            print(status)
            return self.local_folder, None

        if workspace is None:
            raise ValueError(f"Unable to make dataset '{self.name} available for a local run because no AzureML "
                             "workspace has been provided. Provide a workspace, or set a folder for local execution.")

        azureml_dataset = get_or_create_dataset(workspace=workspace,
                                                dataset_name=self.name,
                                                datastore_name=self.datastore)
        target_path = self.target_folder or Path(tempfile.mkdtemp())
        use_mounting = self.use_mounting if self.use_mounting is not None else False
        if use_mounting:
            status += f"mounted at {target_path}"
            print(status)
            mount_context = azureml_dataset.mount(mount_point=str(target_path))
            result = target_path, mount_context
        else:
            status += f"downloaded to {target_path}"
            print(status)
            azureml_dataset.download(target_path=str(target_path), overwrite=False)
            result = target_path, None
        return result

    def to_input_dataset(self,
                         workspace: Workspace,
                         dataset_index: int) -> DatasetConsumptionConfig:
        """
        Creates a configuration for using an AzureML dataset inside of an AzureML run. This will make the AzureML
        dataset with given name available as a named input, using INPUT_0 as the key for dataset index 0.

        :param workspace: The AzureML workspace to read from.
        :param dataset_index: Suffix for using datasets as named inputs, the dataset will be marked INPUT_{index}
        """
        status = f"In AzureML, dataset {self.name} (index {dataset_index}) will be "
        azureml_dataset = get_or_create_dataset(workspace=workspace,
                                                dataset_name=self.name,
                                                datastore_name=self.datastore)
        named_input = azureml_dataset.as_named_input(_input_dataset_key(index=dataset_index))
        # If running on windows then self.target_folder may be a WindowsPath, make sure it is
        # in posix format for Azure.
        path_on_compute = self.target_folder.as_posix() if self.target_folder is not None else None
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
        print(status)
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
        status = f"Output dataset {self.name} (index {dataset_index}) will be "
        datastore = get_datastore(workspace, self.datastore)
        dataset = OutputFileDatasetConfig(name=_output_dataset_key(index=dataset_index),
                                          destination=(datastore, self.name + "/"))
        # TODO: Can we get tags into here too?
        dataset = dataset.register_on_complete(name=self.name)
        if self.target_folder:
            raise ValueError("Output datasets can't have a target_folder set.")
        use_mounting = True if self.use_mounting is None else self.use_mounting
        if use_mounting:
            status += "uploaded while the job runs."
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


def create_dataset_configs(all_azure_dataset_ids: List[str],
                           all_dataset_mountpoints: Sequence[PathOrString],
                           all_local_datasets: List[Optional[Path]],
                           datastore: Optional[str] = None,
                           use_mounting: bool = False) -> List[DatasetConfig]:
    """
    Sets up all the dataset consumption objects for the datasets provided. The returned list will have the same length
    as there are non-empty azure dataset IDs.

    Valid arguments combinations:
    N azure datasets, 0 or N mount points, 0 or N local datasets

    :param all_azure_dataset_ids: The name of all datasets on blob storage that will be used for this run.
    :param all_dataset_mountpoints: When using the datasets in AzureML, these are the per-dataset mount points.
    :param all_local_datasets: The paths for all local versions of the datasets.
    :param datastore: The name of the AzureML datastore that holds the dataset. This can be empty if the AzureML
        workspace has only a single datastore, or if the default datastore should be used.
    :param use_mounting: If True, the dataset will be "mounted", that is, individual files will be read
        or written on-demand over the network. If False, the dataset will be fully downloaded before the job starts,
        respectively fully uploaded at job end for output datasets.
    :return: A list of DatasetConfig objects, in the same order as datasets were provided in all_azure_dataset_ids,
    omitting datasets with an empty name.
    """
    datasets: List[DatasetConfig] = []
    num_local = len(all_local_datasets)
    num_azure = len(all_azure_dataset_ids)
    num_mount = len(all_dataset_mountpoints)
    if num_azure > 0 and (num_local == 0 or num_local == num_azure) and (num_mount == 0 or num_mount == num_azure):
        # Test for valid settings: If we have N azure datasets, the local datasets and mount points need to either
        # have exactly the same length, or 0. In the latter case, empty mount points and no local dataset will be
        # assumed below.
        count = num_azure
    elif num_azure == 0 and num_mount == 0:
        # No datasets in Azure at all: This is possible for runs that for example download their own data from the web.
        # There can be any number of local datasets, but we are not checking that. In MLRunner.setup, there is a check
        # that leaves local datasets intact if there are no Azure datasets.
        return []
    else:
        raise ValueError("Invalid dataset setup. You need to specify N entries in azure_datasets and a matching "
                         "number of local_datasets and dataset_mountpoints")
    for i in range(count):
        azure_dataset = all_azure_dataset_ids[i] if i < num_azure else ""
        if not azure_dataset:
            continue
        mount_point = all_dataset_mountpoints[i] if i < num_mount else ""
        local_dataset = all_local_datasets[i] if i < num_local else None
        config = DatasetConfig(name=azure_dataset,
                               target_folder=mount_point,
                               local_folder=local_dataset,
                               use_mounting=use_mounting,
                               datastore=datastore or "")
        datasets.append(config)
    return datasets


def find_workspace_for_local_datasets(aml_workspace: Optional[Workspace],
                                      workspace_config_path: Optional[Path],
                                      dataset_configs: List[DatasetConfig]) -> Optional[Workspace]:
    """
    If any of the dataset_configs require an AzureML workspace then try to get one, otherwise return None.

    :param aml_workspace: There are two optional parameters used to glean an existing AzureML Workspace. The simplest is
        to pass it in as a parameter.
    :param workspace_config_path: The 2nd option is to specify the path to the config.json file downloaded from the
        Azure portal from which we can retrieve the existing Workspace.
    :param dataset_configs: List of DatasetConfig describing the input datasets.
    :return: Workspace if required, None otherwise.
    """
    workspace: Workspace = None
    # Check whether an attempt will be made to mount or download a dataset when running locally.
    # If so, try to get the AzureML workspace.
    if any(dc.local_folder is None for dc in dataset_configs):
        try:
            workspace = get_workspace(aml_workspace, workspace_config_path)
            logging.info(f"Found workspace for datasets: {workspace.name}")
        except Exception as ex:
            logging.info(f"Could not find workspace for datasets. Exception: {ex}")
    return workspace


def setup_local_datasets(aml_workspace: Optional[Workspace],
                         workspace_config_path: Optional[Path],
                         dataset_configs: List[DatasetConfig]) -> Tuple[List[Optional[Path]], List[MountContext]]:
    """
    When running outside of AzureML, setup datasets to be used locally.

    For each DatasetConfig, if local_folder is supplied, then this is assumed to be a local dataset, and this is
    used. Otherwise the dataset is mounted or downloaded to either the target folder or a temporary folder and that is
    used.

    :param aml_workspace: There are two optional parameters used to glean an existing AzureML Workspace. The simplest is
        to pass it in as a parameter.
    :param workspace_config_path: The 2nd option is to specify the path to the config.json file downloaded from the
        Azure portal from which we can retrieve the existing Workspace.
    :param dataset_configs: List of DatasetConfig describing the input datasets.
    :return: Pair of: list of optional paths to the input datasets, list of mountcontexts, one for each mounted dataset.
    """
    workspace = find_workspace_for_local_datasets(aml_workspace, workspace_config_path, dataset_configs)

    mounted_input_datasets: List[Optional[Path]] = []
    mount_contexts: List[MountContext] = []

    for d in dataset_configs:
        target_path, mount_context = d.to_input_dataset_local(workspace)

        mounted_input_datasets.append(target_path)

        if mount_context is not None:
            mount_context.start()
            mount_contexts.append(mount_context)

    return mounted_input_datasets, mount_contexts
