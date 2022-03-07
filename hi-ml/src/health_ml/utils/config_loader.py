#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import importlib
import inspect
import logging
from importlib._bootstrap import ModuleSpec
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Optional

from health_ml.lightning_container import LightningContainer


class ModelConfigLoader:
    """
    Helper class to manage model config loading.
    """

    def __init__(self) -> None:
        pass

    def default_module_spec(self) -> ModuleSpec:
        from health_ml import configs  # type: ignore

        default_module = configs.__name__
        return find_spec(default_module)

    def find_module_search_specs(self, model_name: str) -> ModuleSpec:
        """
        Given model name (either only the class name or fully qualified), return the ModuleSpec that should be used for
        loading. If the model name is only the class name, the function will return the result of calling
        default_module_spec. Otherwise, this will return the module of the (fully qualified) model name.
        """
        model_namespace_parts = model_name.split(".")
        if len(model_namespace_parts) == 1:
            # config must be in the default path, nothing to be done
            return self.default_module_spec()

        module_name = ".".join(model_namespace_parts[:-1])
        logging.debug(f"Getting specification for module {module_name}")
        try:
            custom_spec: Optional[ModuleSpec] = find_spec(module_name)
        except Exception:
            custom_spec = None
        if custom_spec is None:
            raise ValueError(f"Module {module_name} was not found.")
        return custom_spec

    def _get_model_config(self, module_spec: ModuleSpec, model_name: str) -> Optional[LightningContainer]:
        """
        Given a module specification check to see if it has a class property with
        the <model_name> provided, and instantiate that config class with the
        provided <config_overrides>. Otherwise, return None.

        :param module_spec:
        :return: Instantiated model config if it was found.
        """
        # noinspection PyBroadException
        try:
            logging.debug(f"Importing {module_spec.name}")
            target_module = importlib.import_module(module_spec.name)
            # The "if" clause checks that obj is a class, of the desired name, that is
            # defined in this module rather than being imported into it (and hence potentially
            # being found twice).
            _class = next(
                obj
                for name, obj in inspect.getmembers(target_module)
                if inspect.isclass(obj) and name == model_name and inspect.getmodule(obj) == target_module
            )
            logging.info(f"Found class {_class} in file {module_spec.origin}")
        # ignore the exception which will occur if the provided module cannot be loaded
        # or the loaded module does not have the required class as a member
        except Exception as e:
            exception_text = str(e)
            if exception_text != "":
                logging.warning(f"Error when trying to import module {module_spec.name}: {exception_text}")
            return None
        model_config = _class()
        return model_config

    def _search_recursively_and_store(self, module_spec: ModuleSpec, model_name: str) -> Dict[str, LightningContainer]:
        """
        Given a root namespace eg: A.B.C searches recursively in all child namespaces
        for class property with the <model_name> provided. If found, this is
        instantiated with the provided overrides, and added to the configs dictionary.

        :param module_search_spec:
        """
        configs: Dict[str, LightningContainer] = {}
        root_namespace = module_spec.name
        namespaces_to_search: List[str] = []
        if module_spec.submodule_search_locations:
            logging.debug(
                f"Searching through {len(module_spec.submodule_search_locations)} folders that match namespace "
                f"{module_spec.name}: {module_spec.submodule_search_locations}"
            )
            for root in module_spec.submodule_search_locations:
                # List all python files in all the dirs under root, except for private dirs (prefixed with .)
                all_py_files = [x for x in Path(root).rglob("*.py") if ".." not in str(x)]
                for f in all_py_files:
                    if f.is_file() and "__pycache__" not in str(f) and f.name != "setup.py":
                        sub_namespace = path_to_namespace(f, root=root)
                        namespaces_to_search.append(root_namespace + "." + sub_namespace)
        elif module_spec.origin:
            # The module search spec already points to a python file: Search only that.
            namespaces_to_search.append(module_spec.name)
        else:
            raise ValueError(f"Unable to process module spec: {module_spec}")

        for n in namespaces_to_search:  # type: ignore
            _module_spec = None
            # noinspection PyBroadException
            try:
                _module_spec = find_spec(n)  # type: ignore
            except Exception:
                continue

            if _module_spec:
                config = self._get_model_config(_module_spec, model_name=model_name)
                if config:
                    configs[n] = config  # type: ignore
        return configs

    def create_model_config_from_name(self, model_name: str) -> LightningContainer:
        """
        Returns a model configuration for a model of the given name.

        :param model_name: Class name (for example, "HelloWorld") if the model config is in the default search
        namespace, or fully qualified name of the model, like mymodule.configs.MyConfig)
        """
        if not model_name:
            raise ValueError("Unable to load a model configuration because the model name is missing.")

        logging.info(f"Trying to locate model {model_name}")

        name_parts = model_name.split(".")
        class_name = name_parts[-1]
        module_spec = self.find_module_search_specs(model_name)
        configs = self._search_recursively_and_store(module_spec=module_spec, model_name=class_name)
        if len(configs) == 0:
            raise ValueError(f"Model '{model_name}' was not found in search namespace {module_spec.name}")
        elif len(configs) > 1:
            raise ValueError(
                f"Multiple instances of model '{model_name}' were found in namespaces: {[*configs.keys()]}"
            )
        else:
            return list(configs.values())[0]


def path_to_namespace(path: Path, root: Path) -> str:
    """
    Given a path (in form R/A/B/C) and a root directory R, create a namespace string A.B.C.
    The path must be located under the root directory.

    :param path: Path to convert to namespace
    :param root: Path prefix to remove from namespace.
    :return: A Python namespace string
    """
    return ".".join([Path(x).stem for x in path.relative_to(root).parts])
