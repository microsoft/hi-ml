#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from argparse import (
    _UNRECOGNIZED_ARGS_ATTR,
    OPTIONAL,
    SUPPRESS,
    ArgumentDefaultsHelpFormatter,
    ArgumentError,
    ArgumentParser,
    Namespace,
)
from dataclasses import dataclass
from enum import Enum
import json
import logging
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import param


logger = logging.getLogger(__name__)

EXPERIMENT_RUN_SEPARATOR = ":"


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


def create_argparser(
    config: param.Parameterized,
    usage: Optional[str] = None,
    description: Optional[str] = None,
    epilog: Optional[str] = None,
) -> ArgumentParser:
    """
    Creates an ArgumentParser with all fields of the given config that are overridable.

    :param config: The config whose parameters should be used to populate the argument parser
    :param usage: Brief information about correct usage that is printed if the script started with "--help". If not
    provided, this is auto-generated from the complete set of arguments.
    :param description: A description of the program that is printed if the script is started with "--help"
    :param epilog: A text that is printed after the argument details if the script is started with "--help"
    :return: ArgumentParser
    """
    assert isinstance(config, param.Parameterized)
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, usage=usage, description=description, epilog=epilog
    )
    _add_overrideable_config_args_to_parser(config, parser)
    return parser


def _enum_from_string(enum_class: Type[Enum]) -> Callable:
    """
    Parse a string as an enum. The string must be a valid member of the enum, but matching is not case sensitive.
    Strings are matched based on the Enum member value, not the member name.

    :param enum_class: Enum class to parse string as.
    :return: A parser function that maps the string to an Enum value.
    :raises ValueError: If the Enum has multiple members that have the same value.
    :raises ValueError: If the value to parse does not match any of the Enum member values.
    """
    # Get a dictionary that maps lower case enum names to enum values
    value_to_member = {}
    for member in enum_class.__members__.values():
        lower_value = str(member.value).lower()
        if lower_value in value_to_member:
            raise ValueError(f"Enum values must be unique when lower cased. Duplicate: {lower_value}")
        value_to_member[lower_value] = member

    correct_values = ", ".join(value_to_member.keys())

    def parse_enum(x: str) -> Enum:
        if x.lower() not in value_to_member:
            raise ValueError(f"Invalid value '{x}' for Enum {enum_class.__name__}. Must be one of {correct_values}")
        return value_to_member[x.lower()]

    return parse_enum


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
        get_type: Callable
        if isinstance(_p, param.Boolean):
            get_type = parse_bool
        elif isinstance(_p, param.Integer):

            def to_int(x: str) -> int:
                return _p.default if x == "" else int(x)

            get_type = to_int
        elif isinstance(_p, param.Number):

            def to_float(x: str) -> float:
                return _p.default if x == "" else float(x)

            get_type = to_float
        elif isinstance(_p, param.String):
            get_type = str
        elif isinstance(_p, param.List):

            def to_list(x: str) -> List[Any]:
                return [_p.class_(item) for item in x.split(",") if item]

            get_type = to_list
        elif isinstance(_p, param.NumericTuple):

            def float_or_int(y: str) -> Union[int, float]:
                return int(y) if isinstance(_p, IntTuple) else float(y)

            def to_tuple(x: str) -> Tuple:
                return tuple([float_or_int(item) for item in x.split(",")])

            get_type = to_tuple
        elif isinstance(_p, param.ClassSelector):
            get_type = _enum_from_string(_p.class_) if issubclass(_p.class_, Enum) else _p.class_
        elif isinstance(_p, CustomTypeParam):
            get_type = _p.from_string

        else:
            raise TypeError(f"Parameter of type {_p} is not supported")

        return get_type

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


def is_private_field_name(name: str) -> bool:
    """
    A private field is any Python class member that starts with an underscore eg: _hello

    :param name: a string representing the name of the class member
    """
    return name.startswith("_")


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
        logger.warning(f"Override {key}={desired} failed: {reason} in class {config.__class__.name}")
