import shutil
from pathlib import Path
from typing import Any

import pytest

from health_ml.lightning_container import LightningContainer
from health_ml.utils.config_loader import ModelConfigLoader


@pytest.fixture(scope="module")
def config_loader() -> ModelConfigLoader:
    return ModelConfigLoader()


@pytest.fixture(scope="module")
def hello_config() -> Any:
    from health_ml.configs import hello_container
    assert Path(hello_container.__file__).exists(), "Can't find hello_container config"
    return hello_container


def test_get_default_search_module(config_loader: ModelConfigLoader) -> None:
    search_module = config_loader.get_default_search_module()
    assert search_module == "health_ml.configs"


def test_create_model_config_from_name(config_loader: ModelConfigLoader, hello_config: Any
                                       ) -> None:
    # if no model name is given, an exception should be raised
    with pytest.raises(Exception) as e:
        config_loader.create_model_config_from_name("")
        assert "the model name is missing" in str(e)

    # if no config is found matching the model name, an exception should be raised
    with pytest.raises(Exception) as e:
        config_loader.create_model_config_from_name("idontexist")
        assert "was not found in search namespaces" in str(e)

    # if > 1 config is found matching the model name, an exception should be raised
    config_name = "HelloContainer"
    hello_config_path = Path(hello_config.__file__)
    duplicate_config_file = hello_config_path.parent / "hello_container_2.py"
    duplicate_config_file.touch()
    shutil.copyfile(str(hello_config_path), str(duplicate_config_file))
    with pytest.raises(Exception) as e:
        config_loader.create_model_config_from_name(config_name)
        assert "Multiple instances of model name " in str(e)
    duplicate_config_file.unlink()

    # if exactly one config is found, expect a LightningContainer to be returned
    container = config_loader.create_model_config_from_name(config_name)
    assert isinstance(container, LightningContainer)
    assert container.model_name == config_name


def test_config_in_dif_location(tmp_path: Path, hello_config: Any) -> None:
    himl_root = Path(hello_config.__file__).parent.parent
    new_config_path = himl_root / "hello_container_to_delete.py"
    new_config_path.touch()
    hello_config_path = Path(hello_config.__file__)
    shutil.copyfile(str(hello_config_path), str(new_config_path))
    config_loader = ModelConfigLoader(model_configs_namespace="health_ml")

    config_name = "HelloContainer"
    # Trying to find this config should now cause an exception as it should find it in both "health_ml" and
    # in "health_ml.configs"
    with pytest.raises(Exception) as e:
        config_loader.create_model_config_from_name(config_name)
        assert "Multiple instances of model name HelloContainer were found in namespaces: " \
               "dict_keys(['health_ml.configs.hello_container', 'health_ml.hello_container_to_delete']) " in str(e)
    new_config_path.unlink()
