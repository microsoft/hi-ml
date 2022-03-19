#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Utility functions for interacting with AzureML runs
"""
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentError,
    ArgumentParser,
    Namespace,
    OPTIONAL,
    SUPPRESS,
    _UNRECOGNIZED_ARGS_ATTR,
)
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import conda_merge
import pandas as pd
import param
import ruamel.yaml
from azureml._restclient.constants import RunStatus
from azureml.core import Environment, Experiment, Run, Workspace, get_run
from azureml.core.authentication import InteractiveLoginAuthentication, ServicePrincipalAuthentication
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.train.hyperdrive import HyperDriveRun

T = TypeVar("T")

EXPERIMENT_RUN_SEPARATOR = ":"
DEFAULT_UPLOAD_TIMEOUT_SECONDS: int = 36_000  # 10 Hours

# The version to use when creating an AzureML Python environment. We create all environments with a unique hashed
# name, hence version will always be fixed
ENVIRONMENT_VERSION = "1"

# Environment variables used for authentication
ENV_SERVICE_PRINCIPAL_ID = "HIML_SERVICE_PRINCIPAL_ID"
ENV_SERVICE_PRINCIPAL_PASSWORD = "HIML_SERVICE_PRINCIPAL_PASSWORD"
ENV_TENANT_ID = "HIML_TENANT_ID"
ENV_RESOURCE_GROUP = "HIML_RESOURCE_GROUP"
ENV_SUBSCRIPTION_ID = "HIML_SUBSCRIPTION_ID"
ENV_WORKSPACE_NAME = "HIML_WORKSPACE_NAME"

# Environment variables used for multi-node training
ENV_AZ_BATCHAI_MPI_MASTER_NODE = "AZ_BATCHAI_MPI_MASTER_NODE"
ENV_MASTER_ADDR = "MASTER_ADDR"
ENV_MASTER_IP = "MASTER_IP"
ENV_MASTER_PORT = "MASTER_PORT"
ENV_OMPI_COMM_WORLD_RANK = "OMPI_COMM_WORLD_RANK"
ENV_NODE_RANK = "NODE_RANK"
ENV_GLOBAL_RANK = "GLOBAL_RANK"
ENV_LOCAL_RANK = "LOCAL_RANK"

ENVIRONMENT_VERSION = "1"
FINAL_MODEL_FOLDER = "final_model"
MODEL_ID_KEY_NAME = "model_id"
PYTHON_ENVIRONMENT_NAME = "python_environment_name"
RUN_CONTEXT = Run.get_context()
PARENT_RUN_CONTEXT = getattr(RUN_CONTEXT, "parent", None)
WORKSPACE_CONFIG_JSON = "config.json"

# Names for sections in a Conda environment definition
CONDA_NAME = "name"
CONDA_CHANNELS = "channels"
CONDA_DEPENDENCIES = "dependencies"
CONDA_PIP = "pip"


# By default, define several environment variables that work around known issues in the software stack
DEFAULT_ENVIRONMENT_VARIABLES = {
    "AZUREML_OUTPUT_UPLOAD_TIMEOUT_SEC": "3600",
    # Occasionally uploading data during the run takes too long, and makes the job fail. Default is 300.
    "AZUREML_RUN_KILL_SIGNAL_TIMEOUT_SEC": "900",
    "MKL_SERVICE_FORCE_INTEL": "1",
    # Switching to a new software stack in AML for mounting datasets
    "RSLEX_DIRECT_VOLUME_MOUNT": "true",
    "RSLEX_DIRECT_VOLUME_MOUNT_MAX_CACHE_SIZE": "1",
    "DATASET_MOUNT_CACHE_SIZE": "1",
}

PathOrString = Union[Path, str]


class IntTuple(param.NumericTuple):
    """
    Parameter class that must always have integer values
    """

    def _validate(self, val: Any) -> None:
        """
        Check that input "val" is indeed a tuple of integers. If it is a tuple of some other type, raises
        a ValueError

        :param val: The value to be checked
        """
        super()._validate(val)
        if val is not None:
            for i, n in enumerate(val):
                if not isinstance(n, int):
                    raise ValueError(
                        f"{self.name}: tuple element at index {i} with value {n} in {val} is not an integer"
                    )


class GenericConfig(param.Parameterized):
    def __init__(self, should_validate: bool = True, throw_if_unknown_param: bool = False, **params: Any):
        """
        Instantiates the config class, ignoring parameters that are not overridable.

        :param should_validate: If True, the validate() method is called directly after init.
        :param throw_if_unknown_param: If True, raise an error if the provided "params" contains any key that does not
                                correspond to an attribute of the class.
        :param params: Parameters to set.
        """
        # check if illegal arguments are passed in
        legal_params = self.get_overridable_parameters()
        current_param_names = self.param.values().keys()
        illegal = [k for k, v in params.items() if (k in current_param_names) and (k not in legal_params)]

        if illegal:
            raise ValueError(
                "The following parameters cannot be overridden as they are either "
                f"readonly, constant, or private members : {illegal}"
            )
        if throw_if_unknown_param:
            # check if parameters not defined by the config class are passed in
            unknown = [k for k, v in params.items() if (k not in current_param_names)]
            if unknown:
                raise ValueError(f"The following parameters do not exist: {unknown}")
        # set known arguments
        super().__init__(**{k: v for k, v in params.items() if k in legal_params.keys()})
        if should_validate:
            self.validate()

    def validate(self) -> None:
        """
        Validation method called directly after init to be overridden by children if required
        """
        pass


def set_fields_and_validate(config: param.Parameterized, fields_to_set: Dict[str, Any], validate: bool = True) -> None:
    """
    Add further parameters and, if validate is True, validate. We first try set_param, but that
    fails when the parameter has a setter.

    :param config: The model configuration
    :param fields_to_set: A dictionary of key, value pairs where each key represents a parameter to be added
        and val represents its value
    :param validate: Whether to validate the value of the parameter after adding.
    """
    assert isinstance(config, param.Parameterized)
    for key, value in fields_to_set.items():
        try:
            config.set_param(key, value)
        except ValueError:
            setattr(config, key, value)
    if validate:
        config.validate()


def create_argparser(config: param.Parameterized) -> ArgumentParser:
    """
    Creates an ArgumentParser with all fields of the given config that are overridable.

    :param config: The config whose parameters should be used to populate the argument parser
    :return: ArgumentParser
    """
    assert isinstance(config, param.Parameterized)
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    _add_overrideable_config_args_to_parser(config, parser)
    return parser


def _add_overrideable_config_args_to_parser(config: param.Parameterized, parser: ArgumentParser) -> ArgumentParser:
    """
    Adds all overridable fields of the config class to the given argparser.
    Fields that are marked as readonly, constant or private are ignored.

    :param parser: Parser to add properties to.
    """

    def parse_bool(x: str) -> bool:
        """
        Parse a string as a bool. Supported values are case insensitive and one of:
        'on', 't', 'true', 'y', 'yes', '1' for True
        'off', 'f', 'false', 'n', 'no', '0' for False.

        :param x: string to test.
        :return: Bool value if string valid, otherwise a ValueError is raised.
        """
        sx = str(x).lower()
        if sx in ("on", "t", "true", "y", "yes", "1"):
            return True
        if sx in ("off", "f", "false", "n", "no", "0"):
            return False
        raise ValueError(f"Invalid value {x}, please supply one of True, true, false or False.")

    def _get_basic_type(_p: param.Parameter) -> Union[type, Callable]:
        """
        Given a parameter, get its basic Python type, e.g.: param.Boolean -> bool.
        Throw exception if it is not supported.

        :param _p: parameter to get type and nargs for.
        :return: Type
        """
        if isinstance(_p, param.Boolean):
            p_type: Callable = parse_bool
        elif isinstance(_p, param.Integer):
            p_type = lambda x: _p.default if x == "" else int(x)
        elif isinstance(_p, param.Number):
            p_type = lambda x: _p.default if x == "" else float(x)
        elif isinstance(_p, param.String):
            p_type = str
        elif isinstance(_p, param.List):
            p_type = lambda x: [_p.class_(item) for item in x.split(",")]
        elif isinstance(_p, param.NumericTuple):
            float_or_int = lambda y: int(y) if isinstance(_p, IntTuple) else float(y)
            p_type = lambda x: tuple([float_or_int(item) for item in x.split(",")])
        elif isinstance(_p, param.ClassSelector):
            p_type = _p.class_
        elif isinstance(_p, CustomTypeParam):
            p_type = _p.from_string

        else:
            raise TypeError(f"Parameter of type {_p} is not supported")

        return p_type

    def add_boolean_argument(parser: ArgumentParser, k: str, p: param.Parameter) -> None:
        """
        Add a boolean argument.
        If the parameter default is False then allow --flag (to set it True) and --flag=Bool as usual.
        If the parameter default is True then allow --no-flag (to set it to False) and --flag=Bool as usual.

        :param parser: parser to add a boolean argument to.
        :param k: argument name.
        :param p: boolean parameter.
        """
        if not p.default:
            # If the parameter default is False then use nargs="?" (OPTIONAL).
            # This means that the argument is optional.
            # If it is not supplied, i.e. in the --flag mode, use the "const" value, i.e. True.
            # Otherwise, i.e. in the --flag=value mode, try to parse the argument as a bool.
            parser.add_argument("--" + k, help=p.doc, type=parse_bool, default=False, nargs=OPTIONAL, const=True)
        else:
            # If the parameter default is True then create an exclusive group of arguments.
            # Either --flag=value as usual
            # Or --no-flag to store False in the parameter k.
            group = parser.add_mutually_exclusive_group(required=False)
            group.add_argument("--" + k, help=p.doc, type=parse_bool)
            group.add_argument("--no-" + k, dest=k, action="store_false")
            parser.set_defaults(**{k: p.default})

    for k, p in get_overridable_parameters(config).items():
        # param.Booleans need to be handled separately, they are more complicated because they have
        # an optional argument.
        if isinstance(p, param.Boolean):
            add_boolean_argument(parser, k, p)
        else:
            parser.add_argument("--" + k, help=p.doc, type=_get_basic_type(p), default=p.default)

    return parser


@dataclass
class ParserResult:
    """
    Stores the results of running an argument parser, broken down into a argument-to-value dictionary,
    arguments that the parser does not recognize.
    """

    args: Dict[str, Any]
    unknown: List[str]
    overrides: Dict[str, Any]


def _create_default_namespace(parser: ArgumentParser) -> Namespace:
    """
    Creates an argparse Namespace with all parser-specific default values set.

    :param parser: The parser to work with.
    :return: the Namespace object
    """
    # This is copy/pasted from parser.parse_known_args
    namespace = Namespace()
    for action in parser._actions:
        if action.dest is not SUPPRESS:
            if not hasattr(namespace, action.dest):
                if action.default is not SUPPRESS:
                    setattr(namespace, action.dest, action.default)
    for dest in parser._defaults:
        if not hasattr(namespace, dest):
            setattr(namespace, dest, parser._defaults[dest])
    return namespace


def parse_arguments(
    parser: ArgumentParser, fail_on_unknown_args: bool = False, args: Optional[List[str]] = None
) -> ParserResult:
    """
    Parses a list of commandline arguments with a given parser. Returns results broken down into a full
    arguments dictionary, a dictionary of arguments that were set to non-default values, and unknown
    arguments.

    :param parser: The parser to use
    :param fail_on_unknown_args: If True, raise an exception if the parser encounters an argument that it does
        not recognize. If False, unrecognized arguments will be ignored, and added to the "unknown" field of
        the parser result.
    :param args: Arguments to parse. If not given, use those in sys.argv
    :return: The parsed arguments, and overrides
    """
    if args is None:
        args = sys.argv[1:]
    # The following code is a slightly modified version of what happens in parser.parse_known_args. This had to be
    # copied here because otherwise we would not be able to achieve the priority order that we desire.
    namespace = _create_default_namespace(parser)

    try:
        namespace, unknown = parser._parse_known_args(args, namespace)
        if hasattr(namespace, _UNRECOGNIZED_ARGS_ATTR):
            unknown.extend(getattr(namespace, _UNRECOGNIZED_ARGS_ATTR))
            delattr(namespace, _UNRECOGNIZED_ARGS_ATTR)
    except ArgumentError:
        parser.print_usage(sys.stderr)
        err = sys.exc_info()[1]
        parser._print_message(str(err), sys.stderr)
        raise
    # Parse the arguments a second time, without supplying defaults, to see which arguments actually differ
    # from defaults.
    namespace_without_defaults, _ = parser._parse_known_args(args, Namespace())
    parsed_args = vars(namespace).copy()
    overrides = vars(namespace_without_defaults).copy()
    if len(unknown) > 0 and fail_on_unknown_args:
        raise ValueError(f"Unknown arguments: {unknown}")
    return ParserResult(
        args=parsed_args,
        unknown=unknown,
        overrides=overrides,
    )


def parse_args_and_update_config(config: Any, args: List[str]) -> Any:
    """
    Given a model config and a list of command line arguments, creates an argparser, adds arguments from the config
    parses the list of provided args and updates the config accordingly. Returns the updated config

    :param config: The model configuration
    :param args: A list of command line args to parse
    :return: The config, updated with the values of the provided args
    """
    parser = create_argparser(config)
    parser_results = parse_arguments(parser, args=args)
    _ = apply_overrides(config, parser_results.args)
    return config


def get_overridable_parameters(config: Any) -> Dict[str, param.Parameter]:
    """
    Get properties that are not constant, readonly or private (eg: prefixed with an underscore).

    :param config: The model configuration
    :return: A dictionary of parameter names and their definitions.
    """
    assert isinstance(config, param.Parameterized)
    return dict((k, v) for k, v in config.param.params().items() if reason_not_overridable(v) is None)


def reason_not_overridable(value: param.Parameter) -> Optional[str]:
    """
    Given a parameter, check for attributes that denote it is not overrideable (e.g. readonly, constant,
    private etc). If such an attribute exists, return a string containing a single-word description of the
    reason. Otherwise returns None.

    :param value: a parameter value
    :return: None if the parameter is overridable; otherwise a one-word string explaining why not.
    """
    if value.readonly:
        return "readonly"
    elif value.constant:
        return "constant"
    elif is_private_field_name(value.name):
        return "private"
    elif isinstance(value, param.Callable):
        return "callable"
    return None


def apply_overrides(
    config: Any,
    overrides_to_apply: Optional[Dict[str, Any]],
    should_validate: bool = False,
    keys_to_ignore: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Applies the provided `values` overrides to the config.
    Only properties that are marked as overridable are actually overwritten.

    :param config: The model configuration
    :param overrides_to_apply: A dictionary mapping from field name to value.
    :param should_validate: If true, run the .validate() method after applying overrides.
    :param keys_to_ignore: keys to ignore in reporting failed overrides. If None, do not report.
    :return: A dictionary with all the fields that were modified.
    """

    def _apply(_overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        applied: Dict[str, Any] = {}
        if _overrides is not None:
            overridable_parameters = get_overridable_parameters(config).keys()
            for k, v in _overrides.items():
                if k in overridable_parameters:
                    applied[k] = v
                    setattr(config, k, v)

        return applied

    actual_overrides = _apply(overrides_to_apply)
    if keys_to_ignore is not None:
        report_on_overrides(config, overrides_to_apply, keys_to_ignore)  # type: ignore
    if should_validate:
        config.validate()
    return actual_overrides


def report_on_overrides(config: Any, overrides_to_apply: Dict[str, Any], keys_to_ignore: Set[str]) -> None:
    """
    Logs a warning for every parameter whose value is not as given in "overrides_to_apply", other than those
    in keys_to_ignore.

    :param config: The model configuration
    :param overrides_to_apply: override dictionary, parameter names to values
    :param keys_to_ignore: set of dictionary keys not to report on
    """
    assert isinstance(config, param.Parameterized)
    current_params = config.param.params()
    for key, desired in overrides_to_apply.items():
        if key in keys_to_ignore:
            continue
        actual = getattr(config, key, None)
        if actual == desired:
            continue
        if key not in current_params:
            reason = "parameter is undefined"
        else:
            val = current_params[key]
            reason = reason_not_overridable(val)  # type: ignore
            if reason is None:
                reason = "for UNKNOWN REASONS"
            else:
                reason = f"parameter is {reason}"
        # We could raise an error here instead - to be discussed.
        logging.warning(f"Override {key}={desired} failed: {reason} in class {config.__class__.name}")


def create_from_matching_params(from_object: param.Parameterized, cls_: Type[T]) -> T:
    """
    Creates an object of the given target class, and then copies all attributes from the `from_object` to
    the newly created object, if there is a matching attribute. The target class must be a subclass of
    param.Parameterized.

    :param from_object: The object to read attributes from.
    :param cls_: The name of the class for the newly created object.
    :return: An instance of cls_
    """
    c = cls_()
    if not isinstance(c, param.Parameterized):
        raise ValueError(f"The created object must be a subclass of param.Parameterized, but got {type(c)}")
    for param_name, p in c.param.params().items():  # type: ignore
        if not p.constant and not p.readonly:
            setattr(c, param_name, getattr(from_object, param_name))
    return c


class CustomTypeParam(param.Parameter):
    def _validate(self, val: Any) -> None:
        """
        Validate that the input "val" has the expected format. For example, if this custom type should represent a
        list, verify here that it is so.

        :param val: the value to be verified
        """
        super()._validate(val)

    def from_string(self, x: str) -> Any:
        """
        Base method for taking an input string and returning it evaluated as its expected type (e.g. from_string("3")
        would most likely return int("3")"

        :param x: The string to be evaluated
        :return: The evaluated format of the string
        """
        raise NotImplementedError()


class ListOrDictParam(CustomTypeParam):
    """
    Wrapper class to allow either a List or Dict inside of a Parameterized object.
    """

    def _validate(self, val: Any) -> None:
        """
        Checks that input "val" is indeed a List or Dict object

        :param val: the value to be checked
        """

        if val is None:
            if not self.allow_None:
                raise ValueError("Value must not be None")
            else:
                return
        if not (isinstance(val, List) or isinstance(val, Dict)):
            raise ValueError(f"{val} must be an instance of List or Dict, found {type(val)}")
        super()._validate(val)

    def from_string(self, x: str) -> Union[Dict, List]:
        """
        Parse a string as either a dictionary or list or, if not possible, raise a ValueError.

        For example:
            - from_string('{"x":3, "y":2}') will return a dictionary object
            - from_string('["a", "b", "c"]') will return a list object
            - from_string("['foo']") will return a list object
            - from_string('["foo","bar"') will raise an Exception (missing close bracket)
            - from_string({'learning':3"') will raise an Exception (missing close bracket)

        :param x: the string to parse
        :return: a List or Dict object, as evaluated from the input string
        """
        if x.startswith("{") or x.startswith("["):
            res = json.loads(x.replace("'", '"'))
        else:
            res = [str(item) for item in x.split(",")]
        if isinstance(res, Dict):
            return res
        elif isinstance(res, List):
            return res
        else:
            raise ValueError("Parameter should resolve to List or Dict")


class RunIdOrListParam(CustomTypeParam):
    """
    Wrapper class to allow either a List or string inside of a Parameterized object.
    """

    def _validate(self, val: Any) -> None:
        """
        Checks that the input "val" is indeed a non-empty list or string

        :param val: The value to check
        """
        if val is None:
            if not self.allow_None:
                raise ValueError("Value must not be None")
            else:
                return
        if len(val) == 0 or not (isinstance(val, str) or isinstance(val, list)):
            raise ValueError(f"{val} must be an instance of List or string, found {type(val)}")
        super()._validate(val)

    def from_string(self, x: str) -> List[str]:
        """
        Given a string representing one or more run_ids, first attempts to split into a list, and then
        evaluates each item in the list as a genuine run id

        :param x: The string to evaluate
        :return: a list of one or more strings representing run ids
        """
        res = [str(item) for item in x.split(",")]
        return [determine_run_id_type(x) for x in res]


class CheckpointDownloader:
    def __init__(
        self,
        run_id: str,
        checkpoint_filename: str,
        azure_config_json_path: Optional[Path] = None,
        aml_workspace: Workspace = None,
        download_dir: PathOrString = "checkpoints",
        remote_checkpoint_dir: PathOrString = "checkpoints",
    ) -> None:
        """
        Utility class for downloading checkpoint files from an Azure ML run

        :param run_id: Recovery ID of the run from which to load the checkpoint.
        :param checkpoint_filename: Name of the checkpoint file, expected to be inside the
        `outputs/checkpoints/` directory (e.g. `"best_checkpoint.ckpt"`).
        :param azure_config_json_path: An optional Azure ML settings (JSON file) to use to access the specified
            experiment run. If not running inside an AML Run, and no aml_workspace object is provided, this
            is required.
        :param aml_workspace: An optional Azure ML Workspace object. If not running inside an AML Run, and no
            azure_config_json_path is provided, this is required.
        :param download_dir: The local directory in which to save the downloaded checkpoint files.
        :param remote_checkpoint_dir: The remote folder from which to download the checkpoint file
        """
        self.azure_config_json_path = azure_config_json_path
        self.aml_workspace = aml_workspace
        self.run_id = run_id
        self.checkpoint_filename = checkpoint_filename
        self.download_dir = Path(download_dir)
        self.remote_checkpoint_dir = Path(remote_checkpoint_dir)

    @property
    def local_checkpoint_dir(self) -> Path:
        # in case we run_id is a run recovery id, extract the run id
        run_id_parts = self.run_id.split(":")
        run_id = run_id_parts[-1]
        return self.download_dir / run_id

    @property
    def remote_checkpoint_path(self) -> Path:
        return self.remote_checkpoint_dir / self.checkpoint_filename

    @property
    def local_checkpoint_path(self) -> Path:
        return self.local_checkpoint_dir / self.remote_checkpoint_path

    def download_checkpoint_if_necessary(self) -> Path:
        """Downloads the specified checkpoint if it does not already exist.

        :return: The local path to the downloaded checkpoint file.
        """
        workspace = get_workspace(aml_workspace=self.aml_workspace, workspace_config_path=self.azure_config_json_path)

        if not self.local_checkpoint_path.exists():
            self.local_checkpoint_dir.mkdir(exist_ok=True, parents=True)
            download_checkpoints_from_run_id(
                self.run_id, str(self.remote_checkpoint_path), self.local_checkpoint_dir, aml_workspace=workspace
            )
            assert self.local_checkpoint_path.exists()

        return self.local_checkpoint_path


def is_private_field_name(name: str) -> bool:
    """
    A private field is any Python class member that starts with an underscore eg: _hello

    :param name: a string representing the name of the class member
    """
    return name.startswith("_")


def determine_run_id_type(run_or_recovery_id: str) -> str:
    """
    Determine whether a run id is of type "run id" or "run recovery id". Run recovery ideas take the form
    "experiment_name:run_id". If the input
    string takes the format of a run recovery id, only the run id part will be returned. If it is a run id already,
    it will be returned without transformation.

    :param run_or_recovery_id: The id to determine as either a run id or a run recovery id
    :return: A string representing the run id
    """
    if run_or_recovery_id is None:
        raise ValueError("Expected run_id or run_recovery_id but got None")
    parts = run_or_recovery_id.split(EXPERIMENT_RUN_SEPARATOR)
    if len(parts) > 1:
        # return only the run_id, which comes after the colon
        return parts[1]
    return run_or_recovery_id


def find_file_in_parent_folders(file_name: str, stop_at_path: List[Path]) -> Optional[Path]:
    """Searches for a file of the given name in the current working directory, or any of its parent folders.
    Searching stops if either the file is found, or no parent folder can be found, or the search has reached any
    of the given folders in stop_at_path.

    :param file_name: The name of the file to find.
    :param stop_at_path: A list of folders. If any of them is reached, search stops.
    :return: The absolute path of the file if found, or None if it was not found.
    """

    def return_file_or_parent(start_at: Path) -> Optional[Path]:
        logging.debug(f"Searching for file {file_name} in {start_at}")
        expected = start_at / file_name
        if expected.is_file() and expected.name == file_name:
            return expected
        if start_at.parent == start_at or start_at in stop_at_path:
            return None
        return return_file_or_parent(start_at.parent)

    return return_file_or_parent(start_at=Path.cwd())


def find_file_in_parent_to_pythonpath(file_name: str) -> Optional[Path]:
    """
    Recurse up the file system, starting at the current working directory, to find a file. Stop when we hit
    any of the folders in PYTHONPATH.

    :param file_name: The file name of the file to find.
    :return: The path to the file, or None if it cannot be found.
    """
    pythonpaths: List[Path] = []
    if "PYTHONPATH" in os.environ:
        pythonpaths = [Path(path_string) for path_string in os.environ["PYTHONPATH"].split(os.pathsep)]
    return find_file_in_parent_folders(file_name=file_name, stop_at_path=pythonpaths)


def get_workspace(aml_workspace: Optional[Workspace] = None, workspace_config_path: Optional[Path] = None) -> Workspace:
    """
    Retrieve an Azure ML Workspace from one of several places:
      1. If the function has been called during an AML run (i.e. on an Azure agent), returns the associated workspace
      2. If a Workspace object has been provided by the user, return that
      3. If a path to a Workspace config file has been provided, load the workspace according to that.

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to instantiate the workspace.

    :param aml_workspace: If provided this is returned as the AzureML Workspace.
    :param workspace_config_path: If not provided with an AzureML Workspace, then load one given the information in this
        config
    :return: An AzureML workspace.
    """
    if is_running_in_azure_ml(RUN_CONTEXT):
        return RUN_CONTEXT.experiment.workspace

    if aml_workspace:
        return aml_workspace

    if workspace_config_path is None:
        workspace_config_path = find_file_in_parent_to_pythonpath(WORKSPACE_CONFIG_JSON)
        if workspace_config_path:
            logging.info(f"Using the workspace config file {str(workspace_config_path.absolute())}")
        else:
            raise ValueError("No workspace config file given, nor can we find one.")

    if not isinstance(workspace_config_path, Path):
        raise ValueError("Workspace config path is not a path, check your input.")
    elif workspace_config_path.is_file():
        auth = get_authentication()
        workspace = Workspace.from_config(path=str(workspace_config_path), auth=auth)
        logging.info(f"Logged into AzureML workspace {workspace.name}")
        return workspace

    raise ValueError("Workspace config file does not exist or cannot be read.")


def create_run_recovery_id(run: Run) -> str:
    """
     Creates a unique ID for a run, from which the experiment name and the run ID can be re-created

    :param run: an instantiated run.
    :return: recovery id for a given run in format: [experiment name]:[run id]
    """
    return str(run.experiment.name + EXPERIMENT_RUN_SEPARATOR + run.id)


def split_recovery_id(id_str: str) -> Tuple[str, str]:
    """
    Splits a run ID into the experiment name and the actual run.
    The argument can be in the format 'experiment_name:run_id',
    or just a run ID like user_branch_abcde12_123. In the latter case, everything before the last
    two alphanumeric parts is assumed to be the experiment name.

    :param id_str: The string run ID.
    :return: experiment name and run name
    """
    components = id_str.strip().split(EXPERIMENT_RUN_SEPARATOR)
    if len(components) > 2:
        raise ValueError(f"recovery_id must be in the format: 'experiment_name:run_id', but got: {id_str}")
    elif len(components) == 2:
        return components[0], components[1]
    else:
        recovery_id_regex = r"^(\w+)_\d+_[0-9a-f]+$|^(\w+)_\d+$"
        match = re.match(recovery_id_regex, id_str)
        if not match:
            raise ValueError(f"The recovery ID was not in the expected format: {id_str}")
        return (match.group(1) or match.group(2)), id_str


def fetch_run(workspace: Workspace, run_recovery_id: str) -> Run:
    """
    Finds an existing run in an experiment, based on a recovery ID that contains the experiment ID and the actual RunId.
    The run can be specified either in the experiment_name:run_id format, or just the run_id.

    :param workspace: the configured AzureML workspace to search for the experiment.
    :param run_recovery_id: The Run to find. Either in the full recovery ID format, experiment_name:run_id or
        just the run_id
    :return: The AzureML run.
    """
    experiment, run = split_recovery_id(run_recovery_id)
    try:
        experiment_to_recover = Experiment(workspace, experiment)
    except Exception as ex:
        raise Exception(f"Unable to retrieve run {run} in experiment {experiment}: {str(ex)}")
    run_to_recover = fetch_run_for_experiment(experiment_to_recover, run)
    logging.info(f"Fetched run #{run_to_recover.number} {run} from experiment {experiment}.")
    return run_to_recover


def fetch_run_for_experiment(experiment_to_recover: Experiment, run_id: str) -> Run:
    """
    Gets an AzureML Run object for a given run ID in an experiment.

    :param experiment_to_recover: an experiment
    :param run_id: a string representing the Run ID of one of the runs of the experiment
    :return: the run matching run_id_or_number; raises an exception if not found
    """
    try:
        return get_run(experiment=experiment_to_recover, run_id=run_id, rehydrate=True)
    except Exception:
        available_runs = experiment_to_recover.get_runs()
        available_ids = ", ".join([run.id for run in available_runs])
        raise Exception(
            f"Run {run_id} not found for experiment: {experiment_to_recover.name}. Available runs are: {available_ids}"
        )


def get_authentication() -> Union[InteractiveLoginAuthentication, ServicePrincipalAuthentication]:
    """
    Creates a service principal authentication object with the application ID stored in the present object. The
    application key is read from the environment.

    :return: A ServicePrincipalAuthentication object that has the application ID and key or None if the key is not
        present
    """
    service_principal_id = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_ID, allow_missing=True)
    tenant_id = get_secret_from_environment(ENV_TENANT_ID, allow_missing=True)
    service_principal_password = get_secret_from_environment(ENV_SERVICE_PRINCIPAL_PASSWORD, allow_missing=True)
    if service_principal_id and tenant_id and service_principal_password:
        return ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=service_principal_id,
            service_principal_password=service_principal_password,
        )
    logging.info(
        "Using interactive login to Azure. To use Service Principal authentication, set the environment "
        f"variables {ENV_SERVICE_PRINCIPAL_ID}, {ENV_SERVICE_PRINCIPAL_PASSWORD}, and {ENV_TENANT_ID}"
    )
    return InteractiveLoginAuthentication()


def get_secret_from_environment(name: str, allow_missing: bool = False) -> Optional[str]:
    """
    Gets a password or key from the secrets file or environment variables.

    :param name: The name of the environment variable to read. It will be converted to uppercase.
    :param allow_missing: If true, the function returns None if there is no entry of the given name in any of the
        places searched. If false, missing entries will raise a ValueError.
    :return: Value of the secret. None, if there is no value and allow_missing is True.
    """
    name = name.upper()
    value = os.environ.get(name, None)
    if not value and not allow_missing:
        raise ValueError(f"There is no value stored for the secret named '{name}'")
    return value


def to_azure_friendly_string(x: Optional[str]) -> Optional[str]:
    """
    Given a string, ensure it can be used in Azure by replacing everything apart from a-z, A-Z, 0-9, or _ with _,
    and replace multiple _ with a single _.

    :param x: Optional string to be converted.
    :return: Converted string, if one supplied. None otherwise.
    """
    if x is None:
        return x
    else:
        return re.sub("_+", "_", re.sub(r"\W+", "_", x))


def _log_conda_dependencies_stats(conda: CondaDependencies, message_prefix: str) -> None:
    """
    Write number of conda and pip packages to logs.

    :param conda: A conda dependencies object
    :param message_prefix: A message to prefix to the log string.
    """
    conda_packages_count = len(list(conda.conda_packages))
    pip_packages_count = len(list(conda.pip_packages))
    logging.info(f"{message_prefix}: {conda_packages_count} conda packages, {pip_packages_count} pip packages")
    logging.debug("  Conda packages:")
    for p in conda.conda_packages:
        logging.debug(f"    {p}")
    logging.debug("  Pip packages:")
    for p in conda.pip_packages:
        logging.debug(f"    {p}")


def _split_dependency(dep_str: str) -> Tuple[str, ...]:
    """Splits a string like those coming from PIP constraints into 3 parts: package name, operator, version.
    The operator field can be empty if no constraint is found at all

    :param dep_str: A pip constraint string, like "package-name>=1.0.1"
    :return: A tuple of [package name, operator, version ]
    """
    parts: List[str] = re.split("(<=|==|=|>=|<|>)", dep_str)
    if len(parts) == 1:
        return (parts[0].strip(), "", "")
    if len(parts) == 3:
        return tuple(p.strip() for p in parts)
    raise ValueError(f"Unable to split this package string: {dep_str}")

class PipDependency:
    def __init__(self, dependency_str: str) -> None:
        self.package_name = ""
        self.operator = ""
        self.version = ""
        self._split_dependency_str(dependency_str)

    def _split_dependency_str(self, dependency_str: str) -> None:
        parts = _split_dependency(dependency_str)
        self.package_name = parts[0]
        if len(parts) > 0:
            self.operator = parts[1]
            self.version = parts[2]


def _resolve_package_clash(duplicate_dependencies: List[PipDependency]) -> PipDependency:
    """Given a list of duplicate package names with conflicting versions, if exactly one of these
    is pinned, return that, otherwise raise a ValueError

    :param duplicate_dependencies: a list of PipDependency objects with the same package name
    :raises ValueError: if none of the depencencies specify a pinned version
    :return: A single PipDependency object specifying a pinned version (e.g. 'pkg==0.1')
    """
    pinned_operator = "=="
    found_pinned_dependecy = None
    for dependency in duplicate_dependencies:
        if dependency.operator == pinned_operator:
            if not found_pinned_dependecy:
                found_pinned_dependecy = dependency
            else:
                raise ValueError(f"Found more than one pinned dependency for package: {dependency.package_name}")
    if found_pinned_dependecy:
        return found_pinned_dependecy
    else:
        num_clashes = len(duplicate_dependencies)
        pkg_name = duplicate_dependencies[0].package_name
        raise ValueError(
            f"Encountered {num_clashes} requirements for package {pkg_name}, none of which specify"
            " a pinned version.")


def _resolve_dependencies(all_dependencies: Dict[str, List[PipDependency]],
                          ) -> List[PipDependency]:
    """Apply conflict resolution for pip package versions. Given a dictionary of package name: PipDependency
    objects, applies the following logic:
        - if the package only appears once in all definitions, keep that package version
        - if the package appears in multiple definitions, and is pinned only once, keep that package version
        - otherwise, raise a ValueError

    :param all_dependencies: a dictionary of package name: list of PipDependency objects including description of
        the specified names and versions for that package
    :return: a list of unique PipDependency objects
    """
    unique_dependencies = []
    for dep_name, dep_list in all_dependencies.items():
        if len(dep_list) == 1:
            keep_dependency = dep_list[0]
            unique_dependencies.append(keep_dependency)
        else:
            keep_dependency = _resolve_package_clash(dep_list)
            unique_dependencies.append(keep_dependency)
    return unique_dependencies


def _retrieve_unique_deps(dependencies: List[str]) -> List[str]:
    """
    Given a list of conda dependencies, which may contain duplicate versions
    of the same package name with the same or different versions, returns a
    list of them where each package name occurs only once. If a
    package name appears more than once, a simple resolution strategy will be applied:
    If any of the versions is listed with an equality constraint, that will be kept, irrespective
    of the other constraints, even if they clash with the equality constraint. Multiple equality
    constraints raise an error.

    :param dependencies: the original list of package names to deduplicate
    :return: a list of package specifications in which each package name occurs only once
    """
    all_deps: DefaultDict[str, List[PipDependency]] = defaultdict()
    for dep in dependencies:
        dependency = PipDependency(dep)

        dependency_name = dependency.package_name
        if dependency_name in all_deps:
            all_deps[dependency_name].append(dependency)
        else:
            all_deps[dependency_name] = [dependency]

    unique_deps: List[PipDependency] = _resolve_dependencies(all_deps)

    unique_deps_list = [f"{dep.package_name}{dep.operator}{dep.version}" for dep in unique_deps]
    return unique_deps_list


def _get_pip_dependencies(parsed_yaml: Any) -> Optional[Tuple[int, List[Any]]]:
    """Gets the first pip dependencies section of a Conda yaml file. Returns the index at which the pip section
    was found, and the pip section itself. If no pip section was found, returns None

    :param parsed_yaml: the conda yaml file to get the pip requirements from
    :return: the index at which the pip section was found, and the pip section itself
    """
    if CONDA_DEPENDENCIES in parsed_yaml:
        for i, dep in enumerate(parsed_yaml.get(CONDA_DEPENDENCIES)):
            if isinstance(dep, dict) and CONDA_PIP in dep:
                return i, dep[CONDA_PIP]
    return None


def is_pip_include_dependency(package: str) -> bool:
    """Returns True if the given package name (as used in a Conda environment file) relies on PIP includes,
    in the format "-r requirements.txt"

    :param package: The name of the PIP dependency to check.
    :return: True if the package name is a PIP include statement.
    """
    return package.strip().startswith("-r ")


def is_conda_file_with_pip_include(conda_file: Path) -> Tuple[bool, Dict]:
    """Checks if the given Conda environment file uses the "include" syntax in the pip section, like
    `-r requirements.txt`. If it uses pip includes, the function returns True and a modified Conda yaml
    without all the pip include statements. If no pip include statements are found, False is returned and the
    unmodified Conda yaml.

    :param conda_file: The path of a Conda environment file.
    :return: True if the file uses pip includes, False if not. Seconda return value is the modified Conda environment
    without the PIP include statements.
    """
    conda_yaml = conda_merge.read_file(str(conda_file))
    pip_dep = _get_pip_dependencies(conda_yaml)
    if pip_dep is not None:
        pip_index, pip = pip_dep
        pip_without_include = [package for package in pip if not is_pip_include_dependency(package)]
        if len(pip) != len(pip_without_include):
            if len(pip_without_include) == 0:
                # Avoid empty PIP dependencies section, this causes a failure in conda_merge
                conda_yaml.get(CONDA_DEPENDENCIES).pop(pip_index)
            else:
                conda_yaml.get(CONDA_DEPENDENCIES)[pip_index] = {CONDA_PIP: pip_without_include}
            return True, conda_yaml
    return False, conda_yaml


def merge_conda_files(
    conda_files: List[Path],
    result_file: Path,
    pip_files: Optional[List[Path]] = None,
) -> None:
    """
    Merges the given Conda environment files using the conda_merge package, optionally adds any
    dependencies from pip requirements files, and writes the merged file to disk.

    :param conda_files: The Conda environment files to read.
    :param result_file: The location where the merge results should be written.
    :param pip_files: An optional list of one or more pip requirements files including extra dependencies.
    """
    env_definitions: List[Any] = []
    for file in conda_files:
        _, pip_without_include = is_conda_file_with_pip_include(file)
        env_definitions.append(pip_without_include)
    unified_definition = {}

    extra_pip_deps = []
    for pip_file in pip_files or []:
        additional_pip_deps = [d for d in pip_file.read_text().split("\n") if d and not is_pip_include_dependency(d)]
        extra_pip_deps.extend(additional_pip_deps)

    name = conda_merge.merge_names(env.get(CONDA_NAME) for env in env_definitions)
    if name:
        unified_definition[CONDA_NAME] = name

    try:
        channels = conda_merge.merge_channels(env.get(CONDA_CHANNELS) for env in env_definitions)
    except conda_merge.MergeError:
        logging.error("Failed to merge channel priorities.")
        raise
    if channels:
        unified_definition[CONDA_CHANNELS] = channels

    try:
        deps_to_merge = [env.get(CONDA_DEPENDENCIES) for env in env_definitions]
        if len(extra_pip_deps) > 0:
            deps_to_merge.append([{CONDA_PIP: extra_pip_deps}])
        deps = conda_merge.merge_dependencies(deps_to_merge)

        # Remove duplicated pip packages from merged dependencies sections. Note that for a package that is
        # duplicated, the first value encountered will be retained.
        pip_deps_entries = [d for d in deps if isinstance(d, dict) and CONDA_PIP in d]  # type: ignore
        if len(pip_deps_entries) == 0:
            raise ValueError("Didn't find a dictionary with the key 'pip' in the list of dependencies")
        pip_deps_entry: Dict[str, List[str]] = pip_deps_entries[0]
        pip_deps = pip_deps_entry[CONDA_PIP]
        # temporarily remove pip dependencies from deps to be added back after deduplicaton
        deps.remove(pip_deps_entry)

        # remove all non-pip duplicates from the list of dependencies
        unique_deps = _retrieve_unique_deps(deps)

        unique_pip_deps = _retrieve_unique_deps(pip_deps)

        # finally add back the deduplicated list of dependencies
        unique_deps.append({CONDA_PIP: unique_pip_deps})  # type: ignore

    except conda_merge.MergeError:
        logging.error("Failed to merge dependencies.")
        raise
    if unique_deps:
        unified_definition[CONDA_DEPENDENCIES] = unique_deps
    else:
        raise ValueError("No dependencies found in any of the conda files.")

    with result_file.open("w") as f:
        ruamel.yaml.dump(unified_definition, f, indent=2, default_flow_style=False)
    _log_conda_dependencies_stats(CondaDependencies(result_file), "Merged Conda environment")


def create_python_environment(
    conda_environment_file: Path,
    pip_extra_index_url: str = "",
    workspace: Optional[Workspace] = None,
    private_pip_wheel_path: Optional[Path] = None,
    docker_base_image: str = "",
    environment_variables: Optional[Dict[str, str]] = None,
) -> Environment:
    """
    Creates a description for the Python execution environment in AzureML, based on the arguments.
    The environment will have a name that uniquely identifies it (it is based on hashing the contents of the
    Conda file, the docker base image, environment variables and private wheels.

    :param environment_variables: The environment variables that should be set when running in AzureML.
    :param docker_base_image: The Docker base image that should be used when creating a new Docker image.
    :param pip_extra_index_url: If provided, use this PIP package index to find additional packages when building
        the Docker image.
    :param workspace: The AzureML workspace to work in, required if private_pip_wheel_path is supplied.
    :param private_pip_wheel_path: If provided, add this wheel as a private package to the AzureML environment.
    :param conda_environment_file: The file that contains the Conda environment definition.
    """
    conda_dependencies = CondaDependencies(conda_dependencies_file_path=conda_environment_file)
    yaml_contents = conda_environment_file.read_text()
    if pip_extra_index_url:
        # When an extra-index-url is supplied, swap the order in which packages are searched for.
        # This is necessary if we need to consume packages from extra-index that clash with names of packages on
        # pypi
        conda_dependencies.set_pip_option(f"--index-url {pip_extra_index_url}")
        conda_dependencies.set_pip_option("--extra-index-url https://pypi.org/simple")
    # By default, define several environment variables that work around known issues in the software stack
    environment_variables = {**DEFAULT_ENVIRONMENT_VARIABLES, **(environment_variables or {})}
    # See if this package as a whl exists, and if so, register it with AzureML environment.
    if private_pip_wheel_path is not None:
        if not private_pip_wheel_path.is_file():
            raise FileNotFoundError(f"Cannot add private wheel: {private_pip_wheel_path} is not a file.")
        if workspace is None:
            raise ValueError("To use a private pip wheel, an AzureML workspace must be provided.")
        whl_url = Environment.add_private_pip_wheel(
            workspace=workspace, file_path=str(private_pip_wheel_path), exist_ok=True
        )
        conda_dependencies.add_pip_package(whl_url)
        logging.info(f"Added add_private_pip_wheel {private_pip_wheel_path} to AzureML environment.")
    # Create a name for the environment that will likely uniquely identify it. AzureML does hashing on top of that,
    # and will re-use existing environments even if they don't have the same name.
    # Hashing should include everything that can reasonably change. Rely on hashlib here, because the built-in
    hash_string = "\n".join(
        [
            yaml_contents,
            docker_base_image,
            # Changing the index URL can lead to differences in package version resolution
            pip_extra_index_url,
            str(environment_variables),
            # Use the path of the private wheel as a proxy. This could lead to problems if
            # a new environment uses the same private wheel file name, but the wheel has different
            # contents. In hi-ml PR builds, the wheel file name is unique to the build, so it
            # should not occur there.
            str(private_pip_wheel_path),
        ]
    )
    # Python's hash function gives different results for the same string in different python instances,
    # hence need to use hashlib
    sha1 = hashlib.sha1(hash_string.encode("utf8"))
    overall_hash = sha1.hexdigest()[:32]
    unique_env_name = f"HealthML-{overall_hash}"
    env = Environment(name=unique_env_name)
    env.python.conda_dependencies = conda_dependencies
    if docker_base_image:
        env.docker.base_image = docker_base_image
    env.environment_variables = environment_variables
    return env


def register_environment(workspace: Workspace, environment: Environment) -> Environment:
    """
    Try to get the AzureML environment by name and version from the AzureML workspace. If it succeeds, return that
    environment object. If that fails, register the environment on the workspace. If the version is not specified
    on the environment object, uses the value of ENVIRONMENT_VERSION.

    :param workspace: The AzureML workspace to use.
    :param environment: An AzureML execution environment.
    :return: An AzureML Environment object. If the environment did already exist on the workspace, returns that,
        otherwise returns the newly registered environment.
    """
    try:
        env = Environment.get(workspace, name=environment.name, version=environment.version)
        logging.info(f"Using existing Python environment '{env.name}' with version '{env.version}'.")
        return env
    # If environment doesn't exist, AML raises a generic Exception
    except Exception:  # type: ignore
        if environment.version is None:
            environment.version = ENVIRONMENT_VERSION
        logging.info(
            f"Python environment '{environment.name}' does not yet exist, creating and registering it"
            f" with version '{environment.version}'"
        )
        return environment.register(workspace)


def run_duration_string_to_seconds(s: str) -> Optional[int]:
    """
    Parse a string that represents a timespan, and returns it converted into seconds. The string is expected to be
    floating point number with a single character suffix s, m, h, d for seconds, minutes, hours, day.
    Examples: '3.5h', '2d'. If the argument is an empty string, None is returned.

    :param s: The string to parse.
    :return: The timespan represented in the string converted to seconds.
    """
    s = s.strip()
    if not s:
        return None
    suffix = s[-1]
    if suffix == "s":
        multiplier = 1
    elif suffix == "m":
        multiplier = 60
    elif suffix == "h":
        multiplier = 60 * 60
    elif suffix == "d":
        multiplier = 24 * 60 * 60
    else:
        raise ValueError("s", f"Invalid suffix: Must be one of 's', 'm', 'h', 'd', but got: {s}")  # type: ignore
    return int(float(s[:-1]) * multiplier)


def set_environment_variables_for_multi_node() -> None:
    """
    Sets the environment variables that PyTorch Lightning needs for multi-node training.
    """
    if ENV_AZ_BATCHAI_MPI_MASTER_NODE in os.environ:
        # For AML BATCHAI
        os.environ[ENV_MASTER_ADDR] = os.environ[ENV_AZ_BATCHAI_MPI_MASTER_NODE]
    elif ENV_MASTER_IP in os.environ:
        # AKS
        os.environ[ENV_MASTER_ADDR] = os.environ[ENV_MASTER_IP]
    else:
        logging.info("No settings for the MPI central node found. Assuming that this is a single node training job.")
        return

    if ENV_MASTER_PORT not in os.environ:
        os.environ[ENV_MASTER_PORT] = "6105"

    if ENV_OMPI_COMM_WORLD_RANK in os.environ:
        os.environ[ENV_NODE_RANK] = os.environ[ENV_OMPI_COMM_WORLD_RANK]  # node rank is the world_rank from mpi run
    env_vars = ", ".join(f"{var} = {os.environ[var]}" for var in [ENV_MASTER_ADDR, ENV_MASTER_PORT, ENV_NODE_RANK])
    print(f"Distributed training: {env_vars}")


def is_run_and_child_runs_completed(run: Run) -> bool:
    """
    Checks if the given run has successfully completed. If the run has child runs, it also checks if the child runs
    completed successfully.

    :param run: The AzureML run to check.
    :return: True if the run and all child runs completed successfully.
    """

    def is_completed(run_: Run) -> bool:
        status = run_.get_status()
        if run_.status == RunStatus.COMPLETED:
            return True
        logging.info(f"Run {run_.id} in experiment {run_.experiment.name} finished with status {status}.")
        return False

    runs = list(run.get_children())
    runs.append(run)
    return all(is_completed(run) for run in runs)


def get_most_recent_run_id(run_recovery_file: Path) -> str:
    """
    Gets the string name of the most recently executed AzureML run. This is picked up from the `most_recent_run.txt`
    file.

    :param run_recovery_file: The path of the run recovery file
    :return: The run id
    """
    assert run_recovery_file.is_file(), f"No such file: {run_recovery_file}"

    run_id = run_recovery_file.read_text().strip()
    logging.info(f"Read this run ID from file: {run_id}.")
    return run_id


def get_most_recent_run(run_recovery_file: Path, workspace: Workspace) -> Run:
    """
    Gets the name of the most recently executed AzureML run, instantiates that Run object and returns it.

    :param run_recovery_file: The path of the run recovery file
    :param workspace: Azure ML Workspace
    :return: The Run
    """
    run_or_recovery_id = get_most_recent_run_id(run_recovery_file)
    return get_aml_run_from_run_id(run_or_recovery_id, aml_workspace=workspace)


def get_aml_run_from_run_id(
    run_id: str, aml_workspace: Optional[Workspace] = None, workspace_config_path: Optional[Path] = None
) -> Run:
    """
    Returns an AML Run object, given the run id (run recovery id will also be accepted but is not recommended
    since AML no longer requires the experiment name in order to find the run from a workspace).

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to create the workspace.

    :param run_id: The run id of the run to download. Can optionally be a run recovery id
    :param aml_workspace: Optional AML Workspace object
    :param workspace_config_path: Optional path to config file containing AML Workspace settings
    :return: An Azure ML Run object
    """
    run_id_ = determine_run_id_type(run_id)
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    return workspace.get_run(run_id_)


def get_latest_aml_runs_from_experiment(
    experiment_name: str,
    num_runs: int = 1,
    tags: Optional[Dict[str, str]] = None,
    aml_workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
) -> List[Run]:
    """
    Retrieves the experiment <experiment_name> from the identified workspace and returns <num_runs> latest
    runs from it, optionally filtering by tags - e.g. {'tag_name':'tag_value'}

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to create the workspace.

    :param experiment_name: The experiment name to download runs from
    :param num_runs: The number of most recent runs to return
    :param tags: Optional tags to filter experiments by
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional config file containing settings for the AML Workspace
    :return: a list of one or more Azure ML Run objects
    """
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    experiment: Experiment = workspace.experiments[experiment_name]
    return list(islice(experiment.get_runs(tags=tags), num_runs))


def get_run_file_names(run: Run, prefix: str = "") -> List[str]:
    """
    Get the remote path to all files for a given Run which optionally start with a given prefix

    :param run: The AML Run to look up associated files for
    :param prefix: The optional prefix to filter Run files by
    :return: A list of paths within the Run's container
    """
    all_files = run.get_file_names()
    print(f"Selecting files with prefix {prefix}")
    return [f for f in all_files if f.startswith(prefix)] if prefix else all_files


def _download_files_from_run(run: Run, output_dir: Path, prefix: str = "", validate_checksum: bool = False) -> None:
    """
    Download all files for a given AML run, where the filenames may optionally start with a given
    prefix.

    :param run: The AML Run to download associated files for
    :param output_dir: Local directory to which the Run files should be downloaded.
    :param prefix: Optional prefix to filter Run files by
    :param validate_checksum: Whether to validate the content from HTTP response
    """
    run_paths = get_run_file_names(run, prefix=prefix)
    if len(run_paths) == 0:
        raise ValueError("No such files were found for this Run.")

    for run_path in run_paths:
        output_path = output_dir / run_path
        _download_file_from_run(run, run_path, output_path, validate_checksum=validate_checksum)


def download_files_from_run_id(
    run_id: str,
    output_folder: Path,
    prefix: str = "",
    workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
    validate_checksum: bool = False,
) -> None:
    """
    For a given Azure ML run id, first retrieve the Run, and then download all files, which optionally start
    with a given prefix. E.g. if the Run creates a folder called "outputs", which you wish to download all
    files from, specify prefix="outputs". To download all files associated with the run, leave prefix empty.

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to instantiate the workspace.

    If function is called in a distributed PyTorch training script, the files will only be downloaded once per node
    (i.e, all process where is_local_rank_zero() == True). All processes will exit this function once all downloads
    are completed.

    :param run_id: The id of the Azure ML Run
    :param output_folder: Local directory to which the Run files should be downloaded.
    :param prefix: Optional prefix to filter Run files by
    :param workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path to settings for Azure ML Workspace
    :param validate_checksum: Whether to validate the content from HTTP response
    """
    workspace = get_workspace(aml_workspace=workspace, workspace_config_path=workspace_config_path)
    run = get_aml_run_from_run_id(run_id, aml_workspace=workspace)
    _download_files_from_run(run, output_folder, prefix=prefix, validate_checksum=validate_checksum)
    torch_barrier()


def _download_file_from_run(
    run: Run, filename: str, output_file: Path, validate_checksum: bool = False
) -> Optional[Path]:
    """
    Download a single file from an Azure ML Run, optionally validating the content to ensure the file is not
    corrupted during download. If running inside a distributed setting, will only attempt to download the file
    onto the node with local_rank==0. This prevents multiple processes on the same node from trying to download
    the same file, which can lead to errors.

    :param run: The AML Run to download associated file for
    :param filename: The name of the file as it exists in Azure storage
    :param output_file: Local path to which the file should be downloaded
    :param validate_checksum: Whether to validate the content from HTTP response
    :return: The path to the downloaded file if local rank is zero, else None
    """
    if not is_local_rank_zero():
        return None

    run.download_file(filename, output_file_path=str(output_file), _validate_checksum=validate_checksum)
    return output_file


def download_file_if_necessary(run: Run, filename: str, output_file: Path, overwrite: bool = False) -> Path:
    """Download any file from an Azure ML run if it doesn't exist locally.

    :param run: AML Run object.
    :param remote_dir: Remote directory from where the file is downloaded.
    :param download_dir: Local directory where to save the downloaded file.
    :param filename: Name of the file to be downloaded (e.g. `"outputs/test_output.csv"`).
    :param overwrite: Whether to force the download even if the file already exists locally.
    :return: Local path to the downloaded file.
    """
    if not overwrite and output_file.exists():
        print("File already exists at", output_file)
    else:
        output_file.parent.mkdir(exist_ok=True, parents=True)
        _download_file_from_run(run, filename, output_file, validate_checksum=True)
        assert output_file.exists()
        print("File is downloaded at", output_file)
    return output_file


def is_global_rank_zero() -> bool:
    """
    Tries to guess if the current process is running as DDP rank zero, before the training has actually started,
    by looking at environment variables.

    :return: True if the current process is global rank 0.
    """
    # When doing multi-node training, this indicates which node the present job is on. This is set in
    # set_environment_variables_for_multi_node
    node_rank = os.getenv(ENV_NODE_RANK, "0")
    return is_local_rank_zero() and node_rank == "0"


def is_local_rank_zero() -> bool:
    """
    Tries to guess if the current process is running as DDP local rank zero (i.e., the process that is responsible for
    GPU 0 on each node).

    :return: True if the current process is local rank 0.
    """
    # The per-node jobs for rank zero do not have any of the rank-related environment variables set. PL will
    # set them only once starting its child processes.
    global_rank = os.getenv(ENV_GLOBAL_RANK)
    local_rank = os.getenv(ENV_LOCAL_RANK)
    return global_rank is None and local_rank is None


def download_from_datastore(
    datastore_name: str,
    file_prefix: str,
    output_folder: Path,
    aml_workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
    overwrite: bool = False,
    show_progress: bool = False,
) -> None:
    """
    Download file(s) from an Azure ML Datastore that are registered within a given Workspace. The path
    to the file(s) to be downloaded, relative to the datastore <datastore_name>, is specified by the parameter
    "prefix".  Azure will search for files within the Datastore whose paths begin with this string.
    If you wish to download multiple files from the same folder, set <prefix> equal to that folder's path
    within the Datastore. If you wish to download a single file, include both the path to the folder it
    resides in, as well as the filename itself. If the relevant file(s) are found, they will be downloaded to
    the folder specified by <output_folder>. If this directory does not already exist, it will be created.
    E.g. if your datastore contains the paths ["foo/bar/1.txt", "foo/bar/2.txt"] and you call this
    function with file_prefix="foo/bar" and output_folder="outputs", you would end up with the
    files ["outputs/foo/bar/1.txt", "outputs/foo/bar/2.txt"]

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to instantiate the workspace.

    :param datastore_name: The name of the Datastore containing the blob to be downloaded. This Datastore itself
        must be an instance of an AzureBlobDatastore.
    :param file_prefix: The prefix to the blob to be downloaded
    :param output_folder: The directory into which the blob should be downloaded
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path to settings for Azure ML Workspace
    :param overwrite: If True, will overwrite any existing file at the same remote path.
        If False, will skip any duplicate file.
    :param show_progress: If True, will show the progress of the file download
    """
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    datastore = workspace.datastores[datastore_name]
    assert isinstance(
        datastore, AzureBlobDatastore
    ), "Invalid datastore type. Can only download from AzureBlobDatastore"  # for mypy
    datastore.download(str(output_folder), prefix=file_prefix, overwrite=overwrite, show_progress=show_progress)
    logging.info(f"Downloaded data to {str(output_folder)}")


def upload_to_datastore(
    datastore_name: str,
    local_data_folder: Path,
    remote_path: Path,
    aml_workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
    overwrite: bool = False,
    show_progress: bool = False,
) -> None:
    """
    Upload a folder to an Azure ML Datastore that is registered within a given Workspace. Note that this will upload
    all files within the folder, but will not copy the folder itself. E.g. if you specify the local_data_dir="foo/bar"
    and that contains the files ["1.txt", "2.txt"], and you specify the remote_path="baz", you would see the
    following paths uploaded to your Datastore: ["baz/1.txt", "baz/2.txt"]

    If not running inside AML and neither a workspace nor the config file are provided, the code will try to locate a
    config.json file in any of the parent folders of the current working directory. If that succeeds, that config.json
    file will be used to instantiate the workspace.

    :param datastore_name: The name of the Datastore to which the blob should be uploaded. This Datastore itself
        must be an instance of an AzureBlobDatastore
    :param local_data_folder: The path to the local directory containing the data to be uploaded
    :param remote_path: The path to which the blob should be uploaded
    :param aml_workspace: Optional Azure ML Workspace object
    :param workspace_config_path: Optional path to settings for Azure ML Workspace
    :param overwrite: If True, will overwrite any existing file at the same remote path.
        If False, will skip any duplicate files and continue to the next.
    :param show_progress: If True, will show the progress of the file download
    """
    if not local_data_folder.is_dir():
        raise TypeError("local_path must be a directory")

    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    datastore = workspace.datastores[datastore_name]
    assert isinstance(
        datastore, AzureBlobDatastore
    ), "Invalid datastore type. Can only upload to AzureBlobDatastore"  # for mypy
    datastore.upload(
        str(local_data_folder), target_path=str(remote_path), overwrite=overwrite, show_progress=show_progress
    )
    logging.info(f"Uploaded data to {str(remote_path)}")


class AmlRunScriptConfig(param.Parameterized):
    """
    Base config for a script that handles Azure ML Runs, which can be retrieved with either a run id, latest_run_file,
    or by giving the experiment name (optionally alongside tags and number of runs to retrieve). A config file path can
    also be presented, to specify the Workspace settings. It is assumed that every AML script would have these
    parameters by default. This class can be inherited from if you wish to add additional command line arguments
    to your script (see HimlDownloadConfig and HimlTensorboardConfig for examples)
    """

    latest_run_file: Path = param.ClassSelector(
        class_=Path,
        default=None,
        instantiate=False,
        doc="Optional path to most_recent_run.txt where the ID of the" "latest run is stored",
    )
    experiment: str = param.String(
        default=None, allow_None=True, doc="The name of the AML Experiment that you wish to download Run files from"
    )
    num_runs: int = param.Integer(
        default=1, allow_None=True, doc="The number of runs to download from the " "named experiment"
    )
    config_file: Path = param.ClassSelector(
        class_=Path, default=None, instantiate=False, doc="Path to config.json where Workspace name is defined"
    )
    tags: Dict[str, Any] = param.Dict()
    run: List[str] = RunIdOrListParam(
        default=None,
        allow_None=True,
        doc="Either single or multiple run id(s). Will be stored as a list"
        " of strings. Also supports run_recovery_ids but this is not "
        "recommended",
    )


def _get_runs_from_script_config(script_config: AmlRunScriptConfig, workspace: Workspace) -> List[Run]:
    """
    Given an AMLRunScriptConfig object, retrieve a run id, given the supplied arguments. For example,
    if "run" has been specified, retrieve the AML Run that corresponds to the supplied run id(s). Alternatively,
    if "experiment" has been specified, retrieve "num_runs" (defaults to 1) latest runs from that experiment. If
    neither is supplied, looks for a file named "most_recent_run.txt" in the current directory and its parents.
    If found, reads the latest run id from there are retrieves the corresponding run. Otherwise, raises a ValueError.

    :param script_config: The AMLRunScriptConfig object which contains the parsed arguments
    :param workspace: an AML Workspace object
    :return: a List of one or more retrieved AML Runs
    """
    if script_config.run is None:
        if script_config.experiment is None:
            # default to latest run file
            latest_run_file = find_file_in_parent_to_pythonpath("most_recent_run.txt")
            if latest_run_file is None:
                raise ValueError("Could not find most_recent_run.txt")
            runs = [get_most_recent_run(latest_run_file, workspace)]
        else:
            # get latest runs from experiment
            runs = get_latest_aml_runs_from_experiment(
                script_config.experiment,
                tags=script_config.tags,
                num_runs=script_config.num_runs,
                aml_workspace=workspace,
            )
    else:
        run_ids: List[str]
        run_ids = script_config.run if isinstance(script_config.run, list) else [script_config.run]  # type: ignore
        runs = [get_aml_run_from_run_id(run_id, aml_workspace=workspace) for run_id in run_ids]
    return runs


def download_checkpoints_from_run_id(
    run_id: str,
    checkpoint_path_or_folder: str,
    output_folder: Path,
    aml_workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
) -> None:
    """
    Given an Azure ML run id, download all files from a given checkpoint directory within that run, to
    the path specified by output_path.
    If running in AML, will take the current workspace. Otherwise, if neither aml_workspace nor
    workspace_config_path are provided, will try to locate a config.json file in any of the
    parent folders of the current working directory.

    :param run_id: The id of the run to download checkpoints from
    :param checkpoint_path_or_folder: The path to the either a single checkpoint file, or a directory of
        checkpoints within the run files. If a folder is provided, all files within it will be downloaded.
    :param output_folder: The path to which the checkpoints should be stored
    :param aml_workspace: Optional AML workspace object
    :param workspace_config_path: Optional workspace config file
    """
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    download_files_from_run_id(
        run_id, output_folder, prefix=checkpoint_path_or_folder, workspace=workspace, validate_checksum=True
    )


def is_running_in_azure_ml(aml_run: Run = RUN_CONTEXT) -> bool:
    """
    Returns True if the given run is inside of an AzureML machine, or False if it is on a machine outside AzureML.
    When called without arguments, this functions returns True if the present code is running in AzureML.
    Note that in runs with "compute_target='local'" this function will also return True. Such runs execute outside
    of AzureML, but are able to log all their metrics, etc to an AzureML run.

    :param aml_run: The run to check. If omitted, use the default run in RUN_CONTEXT
    :return: True if the given run is inside of an AzureML machine, or False if it is a machine outside AzureML.
    """
    return hasattr(aml_run, "experiment")


def is_running_on_azure_agent() -> bool:
    """
    Determine whether the current code is running on an Azure agent by examing the environment variable
    for AGENT_OS, that all Azure hosted agents define.

    :return: True if the code appears to be running on an Azure build agent, and False otherwise.
    """
    return bool(os.environ.get("AGENT_OS", None))


def torch_barrier() -> None:
    """
    This is a barrier to use in distributed jobs. Use it to make all processes that participate in a distributed
    pytorch job to wait for each other. When torch.distributed is not set up or not found, the function exits
    immediately.
    """
    try:
        from torch import distributed
    except ModuleNotFoundError:
        logging.info("Skipping the barrier because PyTorch is not available.")
        return
    if distributed.is_available() and distributed.is_initialized():
        distributed.barrier()


def get_tags_from_hyperdrive_run(run: Run, arg_name: str) -> str:
    """
    Given a child Run that was instantiated as part of a HyperDrive run, retrieve the "hyperparameters" tag
    that AML automatically tags it with, and retrieve a specific tag from within that. The available tags are
    determined by the hyperparameters you specified to perform sampling on. E.g. if you defined AML's
    [Grid Sampling](
    https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#grid-sampling)
    over the space {"learning_rate": choice[1, 2, 3]}, each of your 3 child runs will be tagged with
    hyperparameters: {"learning_rate": 0} and so on


    :param run: An AML run object, representing the child of a HyperDrive run
    :param arg_name: The name of the tag that you want to retrieve - representing one of the hyperparameters you
        specified in sampling.
    :return: A string representing the value of the tag, if found.
    """
    return json.loads(run.tags.get("hyperparameters")).get(arg_name)


def aggregate_hyperdrive_metrics(
    run_id: str,
    child_run_arg_name: str,
    aml_workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    For a given HyperDrive run id, retrieves the metrics from each of its children and then aggregates it.
    Returns a DataFrame where each column is one child run, and each row is a metric logged by that child run.
    For example, for a HyperDrive run with 2 children, where each logs epoch, accuracy and loss, the result
    would look like::

        |              | 0               | 1                  |
        |--------------|-----------------|--------------------|
        | epoch        | [1, 2, 3]       | [1, 2, 3]          |
        | accuracy     | [0.7, 0.8, 0.9] | [0.71, 0.82, 0.91] |
        | loss         | [0.5, 0.4, 0.3] | [0.45, 0.37, 0.29] |

    here each column is one of the splits/ child runs, and each row is one of the metrics you have logged to the run.

    It is possible to log rows and tables in Azure ML by calling run.log_table and run.log_row respectively.
    In this case, the DataFrame will contain a Dictionary entry instead of a list, where the keys are the
    table columns (or keywords provided to log_row), and the values are the table values. E.g.::

        |                | 0                                        | 1                                         |
        |----------------|------------------------------------------|-------------------------------------------|
        | accuracy_table |{'epoch': [1, 2], 'accuracy': [0.7, 0.8]} | {'epoch': [1, 2], 'accuracy': [0.8, 0.9]} |

    It is also possible to log plots in Azure ML by calling run.log_image and passing in a matplotlib plot. In
    this case, the DataFrame will contain a string representing the path to the artifact that is generated by AML
    (the saved plot in the Logs & Outputs pane of your run on the AML portal). E.g.::

        |                | 0                                       | 1                                     |
        |----------------|-----------------------------------------|---------------------------------------|
        | accuracy_plot  | aml://artifactId/ExperimentRun/dcid.... | aml://artifactId/ExperimentRun/dcid...|

    :param run_id: The run id (type: str) of the parent run. The type of this run must be an AML HyperDriveRun
    :param child_run_arg_name: the name of the argument given to each child run to denote its position relative
        to other child runs (e.g. this arg could equal 'child_run_index' - then each of your child runs should expect
        to receive the arg '--child_run_index' with a value <= the total number of child runs)
    :param aml_workspace: If provided this is returned as the AzureML Workspace.
    :param workspace_config_path: If not provided with an AzureML Workspace, then load one given the information in this
        config
    :return: A Pandas DataFrame containing the aggregated metrics from each child run
    """
    workspace = get_workspace(aml_workspace=aml_workspace, workspace_config_path=workspace_config_path)
    run = get_aml_run_from_run_id(run_id, aml_workspace=workspace)
    assert isinstance(run, HyperDriveRun)
    metrics: DefaultDict = defaultdict()
    for child_run in run.get_children():
        child_run_metrics = child_run.get_metrics()
        child_run_tag = get_tags_from_hyperdrive_run(child_run, child_run_arg_name)
        for k, v in child_run_metrics.items():
            if k not in metrics:
                metrics[k] = {}
            metrics[k][child_run_tag] = v

    df = pd.DataFrame.from_dict(metrics, orient="index")
    return df


def download_files_from_hyperdrive_children(
    run: Run, remote_file_paths: str, local_download_folder: Path, hyperparam_name: str = ""
) -> List[str]:
    """
    Download a specified file or folder from each of the children of an Azure ML Hyperdrive run. For each child
    run, create a separate folder within your report folder, based on the value of whatever hyperparameter
    was being sampled. E.g. if you sampled over batch sizes 10, 50 and 100, you'll see 3 folders in your
    report folder, named 10, 50 and 100 respectively. If remote_file_path represents a path to a folder, the
    entire folder and all the files within it will be downloaded

    :param run: An AML Run object whose type equals "hyperdrive"
    :param remote_file_paths: A string of one or more paths to the content in the Datastore associated with your
        run outputs, separated by commas
    :param local_download_folder: The local folder to download the files to
    :param hyperparam_name: The name of one of the hyperparameters that was sampled during the HyperDrive
        run. This is used to ensure files are downloaded into logically-named folders
    :return: A list of paths to the downloaded files
    """
    if len(hyperparam_name) == 0:
        raise ValueError(
            "To download results from a HyperDrive run you must provide the hyperparameter name" "that was sampled over"
        )

    # For each child run we create a directory in the local_download_folder named after value of the
    # hyperparam sampled for this child.
    downloaded_file_paths = []
    for child_run in run.get_children():
        child_run_index = get_tags_from_hyperdrive_run(child_run, hyperparam_name)
        if child_run_index is None:
            raise ValueError("Child run expected to have the tag {child_run_tag}")

        # The artifact will be downloaded into a child folder within local_download_folder
        # strip any special characters from the hyperparam index name
        local_folder_child_run = local_download_folder / re.sub("[^A-Za-z0-9]+", "", str(child_run_index))
        local_folder_child_run.mkdir(exist_ok=True)
        for remote_file_path in remote_file_paths.split(","):
            download_files_from_run_id(child_run.id, local_folder_child_run, prefix=remote_file_path)
            downloaded_file_path = local_folder_child_run / remote_file_path
            if not downloaded_file_path.exists():
                logging.warning(
                    f"Unable to download the file {remote_file_path} from the datastore associated" "with this run."
                )
            else:
                downloaded_file_paths.append(str(downloaded_file_path))

    return downloaded_file_paths


def create_aml_run_object(
    experiment_name: str,
    run_name: Optional[str] = None,
    workspace: Optional[Workspace] = None,
    workspace_config_path: Optional[Path] = None,
    snapshot_directory: Optional[PathOrString] = None,
) -> Run:
    """
    Creates an AzureML Run object in the given workspace, or in the workspace given by the AzureML config file.
    This Run object can be used to write metrics to AzureML, upload files, etc, when the code is not running in
    AzureML. After finishing all operations, use `run.flush()` to write metrics to the cloud, and `run.complete()` or
    `run.fail()`.

    Example:
    >>>run = create_aml_run_object(experiment_name="run_on_my_vm", run_name="try1")
    >>>run.log("foo", 1.23)
    >>>run.flush()
    >>>run.complete()

    :param experiment_name: The AzureML experiment that should hold the run that will be created.
    :param run_name: An optional name for the run (this will be used as the display name in the AzureML UI)
    :param workspace: If provided, use this workspace to create the run in. If not provided, use the workspace
        specified by the `config.json` file in the folder or its parent folder(s).
    :param workspace_config_path: If not provided with an AzureML workspace, then load one given the information in this
        config file.
    :param snapshot_directory: The folder that should be included as the code snapshot. By default, no snapshot
        is created (snapshot_directory=None or snapshot_directory=""). Set this to the folder that contains all the
        code your experiment uses. You can use a file .amlignore to skip specific files or folders, akin to .gitignore
    :return: An AzureML Run object.
    """
    actual_workspace = get_workspace(aml_workspace=workspace, workspace_config_path=workspace_config_path)
    exp = Experiment(workspace=actual_workspace, name=experiment_name)
    if snapshot_directory is None or snapshot_directory == "":
        snapshot_directory = tempfile.mkdtemp()
    return exp.start_logging(name=run_name, snapshot_directory=str(snapshot_directory))  # type: ignore


def aml_workspace_for_unittests() -> Workspace:
    """
    Gets the default AzureML workspace that is used for unit testing. It first tries to locate a workspace config.json
    file in the present folder or its parents, and create a workspace from that if found. If no config.json file
    is found, the workspace details are read from environment variables. Authentication information is also read
    from environment variables.
    """
    config_json = find_file_in_parent_to_pythonpath(WORKSPACE_CONFIG_JSON)
    if config_json is not None:
        return Workspace.from_config(path=str(config_json))
    else:
        workspace_name = get_secret_from_environment(ENV_WORKSPACE_NAME, allow_missing=False)
        subscription_id = get_secret_from_environment(ENV_SUBSCRIPTION_ID, allow_missing=False)
        resource_group = get_secret_from_environment(ENV_RESOURCE_GROUP, allow_missing=False)
        auth = get_authentication()
        return Workspace.get(
            name=workspace_name, auth=auth, subscription_id=subscription_id, resource_group=resource_group
        )


class UnitTestWorkspaceWrapper:
    """
    Wrapper around aml_workspace so that it is lazily loaded only once. Used for unit testing only.
    """

    def __init__(self) -> None:
        """
        Init.
        """
        self._workspace: Workspace = None

    @property
    def workspace(self) -> Workspace:
        """
        Lazily load the aml_workspace.
        """
        if self._workspace is None:
            self._workspace = aml_workspace_for_unittests()
        return self._workspace
