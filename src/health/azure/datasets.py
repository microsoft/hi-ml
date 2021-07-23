import logging

from azureml.core import Dataset
from azureml.core import Datastore
from azureml.core import Workspace
from azureml.data import FileDataset


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
    except:
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
