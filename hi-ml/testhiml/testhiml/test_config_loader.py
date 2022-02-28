import shutil
from pathlib import Path
from typing import Any

import pytest

from health_azure.utils import is_running_on_azure_agent
from health_ml.lightning_container import LightningContainer
from health_ml.utils.config_loader import ModelConfigLoader, path_to_namespace
from testhiml.utils.fixed_paths_for_tests import full_test_data_path, tests_root_directory


@pytest.fixture(scope="module")
def config_loader() -> ModelConfigLoader:
    return ModelConfigLoader(**{"model": "HelloContainer"})


@pytest.fixture(scope="module")
def hello_config() -> Any:
    from health_ml.configs import hello_world  # type: ignore
    assert Path(hello_world.__file__).exists(), "Can't find hello_container config"
    return hello_world


def test_find_module_search_specs(config_loader: ModelConfigLoader) -> None:
    # By default, property module_search_specs includes the default config path - health_ml.configs
    len_search_specs_before = len(config_loader.module_search_specs)
    assert any([m.name == "health_ml.configs" for m in config_loader.module_search_specs])
    config_loader._find_module_search_specs()
    # nothing should have been added to module_search_specs
    assert len(config_loader.module_search_specs) == len_search_specs_before


def test_find_module_search_specs_outside_default_dir() -> None:
    if is_running_on_azure_agent():
        return
    model_name = "NewConfig"

    dummy_config_dir = Path.cwd() / "test_configs"
    dummy_config_dir.mkdir()
    dummy_config_path = dummy_config_dir / "new_config.py"
    dummy_config = f"""class {model_name}:
    def __init__(self):
        pass
"""
    dummy_config_path.touch()
    dummy_config_path.write_text(dummy_config)

    dummy_config_namespace = f"test_configs.new_config.{model_name}"
    config_loader2 = ModelConfigLoader(**{"model": f"{dummy_config_namespace}"})
    # The root "testhiml" should now be in the system path and the module "outputs" should be in module_search_specs
    # this wont be in the previous results, since the default path was used. The default search_spec (health_ml.configs)
    # should also be in the results for hte new
    assert any([m.name == "new_config" for m in config_loader2.module_search_specs])
    assert any([m.name == "health_ml.configs" for m in config_loader2.module_search_specs])

    # If the file doesnt exist but the parent module does, the module will still be appended to module_search_specs
    # at this stage
    config_loader3 = ModelConfigLoader(**{"model": "test_configs.new_config.idontexist"})
    assert any([m.name == "new_config" for m in config_loader3.module_search_specs])

    # If the parent module doesn't exist, an Exception should be raised
    with pytest.raises(Exception) as e:
        ModelConfigLoader(**{"model": "testhiml.idontexist.idontexist"})
    assert "was not found" in str(e)

    shutil.rmtree(dummy_config_dir)


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
    config_name = "HelloWorld"
    hello_config_path = Path(hello_config.__file__)
    duplicate_config_file = hello_config_path.parent / "hello_world.py"
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
    model_name = "HelloWorld"
    new_config_path = himl_root / "hello_world_to_delete.py"
    new_config_path.touch()
    hello_config_path = Path(hello_config.__file__)
    shutil.copyfile(str(hello_config_path), str(new_config_path))
    config_loader = ModelConfigLoader(model=model_name)

    # Trying to find this config should now cause an exception as it should find it in both "health_ml" and
    # in "health_ml.configs"
    with pytest.raises(Exception) as e:
        config_loader.create_model_config_from_name(model_name)
        assert "Multiple instances of model name HelloContainer were found in namespaces: " \
               "dict_keys(['health_ml.configs.hello_world', 'health_ml.hello_world_to_delete']) " in str(e)
    new_config_path.unlink()


@pytest.mark.parametrize("is_external", [True, False])
def test_path_to_namespace(is_external: bool) -> None:
    """
    A test to check conversion between namespace to path for InnerEye and external namespaces
    """
    tests_root_dir = tests_root_directory()
    if is_external:
        folder_name = "logs"
        full_folder = tests_root_dir / folder_name
        assert path_to_namespace(
            path=full_folder,
            root=tests_root_dir
        ) == folder_name
    else:
        assert path_to_namespace(
            path=full_test_data_path(),
            root=tests_root_dir
        ) == "test_data"
