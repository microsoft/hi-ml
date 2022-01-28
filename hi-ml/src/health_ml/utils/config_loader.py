#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import importlib
import inspect
import logging
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional

import param
from importlib._bootstrap import ModuleSpec

from health_azure.utils import PathOrString
from health_ml.lightning_container import LightningContainer
from health_ml.utils import fixed_paths


class ModelConfigLoader(param.Parameterized):
    """
    Helper class to manage model config loading.
    """

    def __init__(self, **params: Any):
        super().__init__(**params)
        default_module = self.get_default_search_module()
        self.module_search_specs: List[ModuleSpec] = [importlib.util.find_spec(default_module)]  # type: ignore
        self._find_module_search_specs()

    def _find_module_search_specs(self) -> None:
        """
        Given the fully qualified model name, append the root folder to the system path (so that the config
        file can be discovered) and try to find a spec for the specifed module. If found, appends the spec
        to self.module_search_specs
        """
        model_namespace_parts = self.model.split(".")
        if len(model_namespace_parts) == 1:
            # config must be in the default path. This is already in module_search_specs so we dont need to do anything
            return
        else:
            # Get the root folder of the fully qualified model name and ensure it is in the path to enable
            # discovery of the config file
            model_namespace_path = Path(self.model.replace(".", "/"))
            root_namespace = str(Path(model_namespace_path.parts[0]).absolute())
            if root_namespace not in sys.path:
                print(f"Adding {str(root_namespace)} to path")
                sys.path.insert(0, str(root_namespace))

            # Strip the root folder (now in the path) and the class name from the model namespace, leaving the
            # module name - e.g. "mymodule.configs"
            model_namespace = ".".join([str(p) for p in model_namespace_path.parts[1:-1]])  # type: ignore

        custom_spec = importlib.util.find_spec(model_namespace)  # type: ignore
        if custom_spec is None:
            raise ValueError(f"Search namespace {model_namespace} was not found.")
        self.module_search_specs.append(custom_spec)

    @staticmethod
    def get_default_search_module() -> str:
        from health_ml import configs  # type: ignore
        return configs.__name__

    def create_model_config_from_name(self, model_name: str) -> LightningContainer:
        """
        Returns a model configuration for a model of the given name.
        To avoid having to import torch here, there are no references to LightningContainer.
        Searching for a class member called <model_name> in the search modules provided recursively.

        :param model_name: Fully qualified name of the model for which to get the configs for - i.e.
            mymodule.configs.MyConfig
        """
        if not model_name:
            raise ValueError("Unable to load a model configuration because the model name is missing.")

        # get the class name from the fully qualified name
        model_name = model_name.split(".")[-1]

        configs: Dict[str, LightningContainer] = {}

        def _get_model_config(module_spec: ModuleSpec) -> Optional[LightningContainer]:
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
                _class = next(obj for name, obj in inspect.getmembers(target_module)
                              if inspect.isclass(obj)
                              and name == model_name  # noqa: W503
                              and inspect.getmodule(obj) == target_module)  # noqa: W503
                logging.info(f"Found class {_class} in file {module_spec.origin}")
            # ignore the exception which will occur if the provided module cannot be loaded
            # or the loaded module does not have the required class as a member
            except Exception as e:
                exception_text = str(e)
                if exception_text != "":
                    logging.warning(f"(from attempt to import module {module_spec.name}): {exception_text}")
                return None
            model_config = _class()
            return model_config

        def _search_recursively_and_store(module_search_spec: ModuleSpec) -> None:
            """
            Given a root namespace eg: A.B.C searches recursively in all child namespaces
            for class property with the <model_name> provided. If found, this is
            instantiated with the provided overrides, and added to the configs dictionary.

            :param module_search_spec:
            """
            root_namespace = module_search_spec.name
            namespaces_to_search: List[str] = []
            if module_search_spec.submodule_search_locations:
                logging.debug(f"Searching through {len(module_search_spec.submodule_search_locations)} folders that "
                              f"match namespace {module_search_spec.name}: "
                              f"{module_search_spec.submodule_search_locations}")
                for root in module_search_spec.submodule_search_locations:
                    for n in Path(root).rglob("*"):
                        if n.is_file() and "__pycache__" not in str(n):
                            sub_namespace = path_to_namespace(n, root=root)
                            namespaces_to_search.append(root_namespace + "." + sub_namespace)
            elif module_search_spec.origin:
                # The module search spec already points to a python file: Search only that.
                namespaces_to_search.append(module_search_spec.name)
            else:
                raise ValueError(f"Unable to process module spec: {module_search_spec}")

            for n in namespaces_to_search:  # type: ignore
                _module_spec = None
                # noinspection PyBroadException
                try:
                    _module_spec = find_spec(n)  # type: ignore
                except Exception:
                    pass

                if _module_spec:
                    config = _get_model_config(_module_spec)
                    if config:
                        configs[n] = config  # type: ignore

        for search_spec in self.module_search_specs:
            _search_recursively_and_store(search_spec)

        if len(configs) == 0:
            raise ValueError(
                f"Model name {model_name} was not found in search namespaces: "
                f"{[s.name for s in self.module_search_specs]}.")
        elif len(configs) > 1:
            raise ValueError(
                f"Multiple instances of model name {model_name} were found in namespaces: {configs.keys()}.")
        else:
            return list(configs.values())[0]


def path_to_namespace(path: Path, root: PathOrString = fixed_paths.repository_root_directory()) -> str:
    """
    Given a path (in form R/A/B/C) and an optional root directory R, create a namespace A.B.C.
    If root is provided, then path must be a relative child to it.

    :param path: Path to convert to namespace
    :param root: Path prefix to remove from namespace (default is project root)
    :return:
    """
    return ".".join([Path(x).stem for x in path.relative_to(root).parts])
