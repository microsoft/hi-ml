import shutil
from pathlib import Path

import pytest

from health_ml.configs import hello_world as hello_config  # type: ignore
from health_ml.lightning_container import LightningContainer
from health_ml.utils.config_loader import ModelConfigLoader, path_to_namespace


def test_find_module_search_specs() -> None:
    config_loader = ModelConfigLoader()
    module_spec = config_loader.find_module_search_specs(model_name="health_ml.utils.config_loader.Foo")
    assert module_spec.name == "health_ml.utils.config_loader"
    module_spec = config_loader.find_module_search_specs(model_name="DoesNotExist")
    assert module_spec.name == "health_ml.configs"


def test_get_default_search_module() -> None:
    config_loader = ModelConfigLoader()
    search_module = config_loader.default_module_spec()
    assert search_module.name == "health_ml.configs"


def test_create_model_config_from_name_errors() -> None:
    config_loader = ModelConfigLoader()
    # if no model name is given, an exception should be raised
    with pytest.raises(Exception) as e:
        config_loader.create_model_config_from_name("")
    assert "the model name is missing" in str(e)

    # if no config is found matching the model name, an exception should be raised
    with pytest.raises(Exception) as e:
        config_loader.create_model_config_from_name("idontexist")
    assert "was not found in search namespace" in str(e)

    with pytest.raises(Exception) as e:
        config_loader.create_model_config_from_name("testhiml.idontexist.idontexist")
    assert "Module testhiml.idontexist was not found" in str(e)


def test_create_model_config_from_name_duplicates() -> None:
    config_loader = ModelConfigLoader()
    config_name = "HelloWorld"
    # if exactly one config is found, expect a LightningContainer to be returned
    container = config_loader.create_model_config_from_name(config_name)
    assert isinstance(container, LightningContainer)
    assert container.model_name == config_name

    # if > 1 config is found matching the model name, an exception should be raised
    hello_config_path = Path(hello_config.__file__)
    # This file must be excluded from coverage reports, check .coveragerc
    duplicate_config_file = hello_config_path.parent / "temp_config_for_unittests.py"
    shutil.copyfile(hello_config_path, duplicate_config_file)
    with pytest.raises(Exception) as e:
        config_loader.create_model_config_from_name(config_name)
    assert "Multiple instances of model " in str(e)
    duplicate_config_file.unlink()


def test_path_to_namespace() -> None:
    """
    A test to check conversion between paths and python namespaces.
    """
    assert path_to_namespace(Path("/foo/bar/baz"), root=Path("/foo")) == "bar.baz"


def test_config_fully_qualified() -> None:
    """
    Test if we can load model configs when giving a full Python namespace.
    """
    # This name was deliberately chosen to be outside the default searchar namespace
    model_name = "health_ml.utils.config_loader.ModelConfigLoader"
    config_loader = ModelConfigLoader()
    model = config_loader.create_model_config_from_name(model_name=model_name)
    assert type(model).__name__ == "ModelConfigLoader"


def test_config_fully_qualified_invalid() -> None:
    """
    Test error handling if giving a too long namespace
    """
    namespace = "health_ml.utils.config_loader.foo"
    model_name = namespace + ".Foo"
    config_loader = ModelConfigLoader()
    with pytest.raises(ValueError) as ex:
        config_loader.create_model_config_from_name(model_name=model_name)
    assert f"Module {namespace} was not found" in str(ex)
