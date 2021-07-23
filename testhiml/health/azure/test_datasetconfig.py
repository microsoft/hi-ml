from unittest import mock

import pytest
from azureml.core import Datastore
from azureml.data.azure_storage_datastore import AzureBlobDatastore

from health.azure.datasets import DatasetConfig
from health.azure.datasets import get_datastore
from testhiml.health.azure.utils import aml_workspace


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
        get_datastore(workspace=aml_workspace(), datastore_name=does_not_exist)
    assert f"Datastore {does_not_exist} was not found" in str(ex)


def test_get_datastore_1() -> None:
    """
    Tests getting a datastore by name.
    """
    name = "innereyedatasets"
    workspace = aml_workspace()
    datastore = get_datastore(workspace=workspace, datastore_name=name)
    assert isinstance(datastore, AzureBlobDatastore)
    assert datastore.name == name
    # Trying to get a datastore without name should only work if there is a single datastore
    assert len(workspace.datastores) > 1
    with pytest.raises(ValueError) as ex:
        get_datastore(workspace=workspace, datastore_name="")
    assert "No datastore name provided" in str(ex)
    # Now mock the datastores property of the workspace, to pretend there is only a single datastore.
    # With that in place, we can get the datastore without the name
    faked_stores = {name: datastore}
    with mock.patch("azureml.core.Workspace.datastores", faked_stores):
        single_store = get_datastore(workspace=workspace, datastore_name="")
    assert isinstance(single_store, AzureBlobDatastore)
    assert single_store.name == name
