#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import List, Optional, Union

from azureml.core import Dataset, Datastore, Workspace
from azureml.data import FileDataset, OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig


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
        if len(existing_stores) == 1:
            return datastores[existing_stores[0]]
        raise ValueError("No datastore name provided. This is only possible if the workspace has a single datastore. "
                         f"However, the workspace has {len(existing_stores)} datastores: {existing_stores}")
    if datastore_name in datastores:
        return datastores[datastore_name]
    raise ValueError(f"Datastore {datastore_name} was not found in the workspace. Existing datastores: "
                     f"{existing_stores}")


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
                 target_folder: str = "",
                 local_folder: Optional[Path] = None):
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
