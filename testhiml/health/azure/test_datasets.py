from unittest import mock

import pytest
from azureml.core import Dataset
from azureml.data import FileDataset
from azureml.data import OutputFileDatasetConfig
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

from health.azure import datasets
from health.azure.datasets import DatasetConfig
from testhiml.health.azure.utils import DEFAULT_DATASTORE
from testhiml.health.azure.utils import DEFAULT_WORKSPACE


def test_datasetconfig_init() -> None:
    with pytest.raises(ValueError) as ex:
        DatasetConfig(name=" ")
    assert "name of the dataset must be a non-empty string" in str(ex)


def test_get_datastore_fails() -> None:
    """
    Retrieving a datastore that does not exist should fail
    """
    does_not_exist = "does_not_exist"
    with pytest.raises(ValueError) as ex:
        datasets.get_datastore(workspace=DEFAULT_WORKSPACE, datastore_name=does_not_exist)
    assert f"Datastore {does_not_exist} was not found" in str(ex)


def test_get_datastore_without_name() -> None:
    """
    Trying to get a datastore without name should only work if there is a single datastore
    """
    assert len(DEFAULT_WORKSPACE.datastores) > 1
    with pytest.raises(ValueError) as ex:
        datasets.get_datastore(workspace=DEFAULT_WORKSPACE, datastore_name="")
    assert "No datastore name provided" in str(ex)


def test_get_datastore() -> None:
    """
    Tests getting a datastore by name.
    """
    name = DEFAULT_DATASTORE
    datastore = datasets.get_datastore(workspace=DEFAULT_WORKSPACE, datastore_name=name)
    assert isinstance(datastore, AzureBlobDatastore)
    assert datastore.name == name
    assert len(DEFAULT_WORKSPACE.datastores) > 1
    # Now mock the datastores property of the workspace, to pretend there is only a single datastore.
    # With that in place, we can get the datastore without the name
    faked_stores = {name: datastore}
    with mock.patch("azureml.core.Workspace.datastores", faked_stores):
        single_store = datasets.get_datastore(workspace=DEFAULT_WORKSPACE, datastore_name="")
    assert isinstance(single_store, AzureBlobDatastore)
    assert single_store.name == name


def test_dataset_input() -> None:
    """
    Test turning a dataset setup object to an actual AML input dataset.
    """
    # This dataset must exist in the workspace already, or at least in blob storage.
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE)
    aml_dataset = dataset_config.to_input_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.path_on_compute is None
    assert aml_dataset.mode == "download"
    # Downloading or mounting to a given path
    target_folder = "/tmp/foo"
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, target_folder=target_folder)
    aml_dataset = dataset_config.to_input_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.path_on_compute == target_folder
    # Use mounting instead of downloading
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, use_mounting=True)
    aml_dataset = dataset_config.to_input_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert isinstance(aml_dataset, DatasetConsumptionConfig)
    assert aml_dataset.mode == "mount"


def test_dataset_output() -> None:
    """
    Test turning a dataset setup object to an actual AML output dataset.
    """
    name = "new_dataset"
    dataset_config = DatasetConfig(name=name, datastore=DEFAULT_DATASTORE)
    aml_dataset = dataset_config.to_output_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert isinstance(aml_dataset, OutputFileDatasetConfig)
    assert isinstance(aml_dataset.destination, tuple)
    assert aml_dataset.destination[0].name == DEFAULT_DATASTORE
    assert aml_dataset.destination[1] == name + "/"
    assert aml_dataset.mode == "mount"
    # Use downloading instead of mounting
    dataset_config = DatasetConfig(name="hello_world", datastore=DEFAULT_DATASTORE, use_mounting=False)
    aml_dataset = dataset_config.to_output_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert isinstance(aml_dataset, OutputFileDatasetConfig)
    assert aml_dataset.mode == "upload"
    # Mounting at a fixed folder is not possible
    with pytest.raises(ValueError) as ex:
        dataset_config = DatasetConfig(name=name, datastore=DEFAULT_DATASTORE, target_folder="something")
        dataset_config.to_output_dataset(workspace=DEFAULT_WORKSPACE, dataset_index=1)
    assert "Output datasets can't have a target_folder set" in str(ex)


def test_datasets_from_string() -> None:
    """
    Test the conversion of datasets that are only specified as strings.
    """
    dataset1 = "foo"
    dataset2 = "bar"
    store = "store"
    default_store = "default"
    datasets = [dataset1, DatasetConfig(name=dataset2, datastore=store)]
    replaced = datasets._replace_string_datasets(datasets, default_datastore_name=default_store)
    assert len(replaced) == len(datasets)
    for d in replaced:
        assert isinstance(d, DatasetConfig)
    assert replaced[0].name == dataset1
    assert replaced[0].datastore == default_store
    assert replaced[1] == datasets[1]


def test_get_dataset() -> None:
    """
    Test if a dataset that does not yet exist can be created from a folder in blob storage
    """
    # A folder with a single tiny file
    tiny_dataset = "himl-tiny_dataset"
    workspace = DEFAULT_WORKSPACE
    # Check first that there is no dataset yet of that name. If there is, delete that dataset (it would come
    # from previous runs of this test)
    try:
        existing_dataset = Dataset.get_by_name(workspace, name=tiny_dataset)
        existing_dataset.unregister_all_versions()
    except Exception as ex:
        assert "Cannot find dataset registered" in str(ex)
    dataset = datasets.get_or_create_dataset(workspace=workspace,
                                             datastore_name=DEFAULT_DATASTORE,
                                             dataset_name=tiny_dataset)
    assert isinstance(dataset, FileDataset)
    # We should now be able to get that dataset without special means
    dataset2 = Dataset.get_by_name(workspace, name=tiny_dataset)
    # Delete the dataset again
    dataset2.unregister_all_versions()


def test_dataset_keys() -> None:
    """
    Check that dataset keys are non-empty strings, and that inputs and outputs have different keys.
    """
    in1 = datasets._input_dataset_key(1)
    out1 = datasets._output_dataset_key(1)
    assert in1
    assert out1
    assert in1 != out1
