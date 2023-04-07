#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from argparse import ArgumentError, ArgumentParser, Namespace
from enum import Enum
import logging
import sys
from typing import Any, List, Optional, Set, Tuple

import param
from unittest.mock import patch

import pytest
from pytest import CaptureFixture, LogCaptureFixture

from health_azure.argparsing import (
    CustomTypeParam,
    IntTuple,
    ListOrDictParam,
    _add_overrideable_config_args_to_parser,
    _enum_from_string,
    apply_overrides,
    create_argparser,
    get_overridable_parameters,
    parse_args_and_update_config,
    parse_arguments,
    report_on_overrides,
)
from health_azure.utils import set_fields_and_validate
from testazure.test_azure_util import DummyConfig


class ParamEnum(Enum):
    EnumValue1 = ("1",)
    EnumValue2 = "2"


class IllegalCustomTypeNoFromString(param.Parameter):
    def _validate(self, val: Any) -> None:
        super()._validate(val)


class IllegalCustomTypeNoValidate(CustomTypeParam):
    def from_string(self, x: str) -> Any:
        return x


@pytest.mark.fast
def test_create_argparse(dummy_model_config: DummyConfig) -> None:
    with patch("health_azure.argparsing._add_overrideable_config_args_to_parser") as mock_add_args:
        parser = create_argparser(dummy_model_config)
        mock_add_args.assert_called_once()
        assert isinstance(parser, ArgumentParser)


@pytest.mark.fast
def test_add_args(dummy_model_config: DummyConfig) -> None:
    parser = ArgumentParser()
    # assert that calling parse_args on a default ArgumentParser returns an empty Namespace
    args = parser.parse_args([])
    assert args == Namespace()
    # now call _add_overrideable_config_args_to_parser and assert that calling parse_args on the result
    # of that is a non-empty Namepsace
    with patch("health_azure.argparsing.get_overridable_parameters") as mock_get_overridable_parameters:
        mock_get_overridable_parameters.return_value = {"string_param": param.String(default="Hello")}
        parser = _add_overrideable_config_args_to_parser(dummy_model_config, parser)
        assert isinstance(parser, ArgumentParser)
        args = parser.parse_args([])
        assert args != Namespace()
        assert args.string_param == "Hello"


@pytest.mark.fast
def test_parse_args(dummy_model_config: DummyConfig) -> None:
    new_string_arg = "dummy_string"
    new_args = ["--string_param", new_string_arg]
    parser = ArgumentParser()
    parser.add_argument("--string_param", type=str, default=None)
    parser_result = parse_arguments(parser, args=new_args)
    assert parser_result.args.get("string_param") == new_string_arg


class ParamClass(param.Parameterized):
    name: str = param.String(None, doc="Name")
    seed: int = param.Integer(42, doc="Seed")
    flag: bool = param.Boolean(False, doc="Flag")
    not_flag: bool = param.Boolean(True, doc="Not Flag")
    number: float = param.Number(3.14)
    integers: List[int] = param.List(None, class_=int)
    optional_int: Optional[int] = param.Integer(None, doc="Optional int")
    optional_float: Optional[float] = param.Number(None, doc="Optional float")
    floats: List[float] = param.List(None, class_=float)
    tuple1: Tuple[int, float] = param.NumericTuple((1, 2.3), length=2, doc="Tuple")
    int_tuple: Tuple[int, int, int] = IntTuple((1, 1, 1), length=3, doc="Integer Tuple")
    enum: ParamEnum = param.ClassSelector(default=ParamEnum.EnumValue1, class_=ParamEnum, instantiate=False)
    readonly: str = param.String("Nope", readonly=True)
    _non_override: str = param.String("Nope")
    constant: str = param.String("Nope", constant=True)
    strings: List[str] = param.List(default=['some_string'], class_=str)
    other_args = ListOrDictParam(None, doc="List or dictionary of other args")

    def validate(self) -> None:
        pass


@pytest.fixture(scope="module")
def parameterized_config_and_parser() -> Tuple[ParamClass, ArgumentParser]:
    parameterized_config = ParamClass()
    parser = create_argparser(parameterized_config)
    return parameterized_config, parser


@pytest.mark.fast
def test_get_overridable_parameter(parameterized_config_and_parser: Tuple[ParamClass, ArgumentParser]) -> None:
    """
    Test to check overridable parameters are correctly identified.
    """
    parameterized_config = parameterized_config_and_parser[0]
    param_dict = get_overridable_parameters(parameterized_config)
    assert "name" in param_dict
    assert "flag" in param_dict
    assert "not_flag" in param_dict
    assert "seed" in param_dict
    assert "number" in param_dict
    assert "integers" in param_dict
    assert "optional_int" in param_dict
    assert "optional_float" in param_dict
    assert "tuple1" in param_dict
    assert "int_tuple" in param_dict
    assert "enum" in param_dict
    assert "other_args" in param_dict
    assert "strings" in param_dict
    assert "readonly" not in param_dict
    assert "_non_override" not in param_dict
    assert "constant" not in param_dict


@pytest.mark.fast
def test_parser_defaults(parameterized_config_and_parser: Tuple[ParamClass, ArgumentParser]) -> None:
    """
    Check that default values are created as expected, and that the non-overridable parameters
    are omitted.
    """
    parameterized_config = parameterized_config_and_parser[0]
    defaults = vars(create_argparser(parameterized_config).parse_args([]))
    assert defaults["seed"] == 42
    assert defaults["tuple1"] == (1, 2.3)
    assert defaults["int_tuple"] == (1, 1, 1)
    assert defaults["enum"] == ParamEnum.EnumValue1
    assert not defaults["flag"]
    assert defaults["not_flag"]
    assert defaults["strings"] == ['some_string']
    assert "readonly" not in defaults
    assert "constant" not in defaults
    assert "_non_override" not in defaults
    # We can't test if all invalid cases are handled because argparse call sys.exit
    # upon errors.


def check_parsing_succeeds(
    parameterized_config_and_parser: Tuple[ParamClass, ArgumentParser],
    arg: List[str],
    expected_key: str,
    expected_value: Any,
) -> None:
    _, parser = parameterized_config_and_parser
    parser_result = parse_arguments(parser, args=arg)
    assert parser_result.args.get(expected_key) == expected_value


def check_parsing_fails(parameterized_config_and_parser: Tuple[ParamClass, ArgumentParser], arg: List[str]) -> None:
    _, parser = parameterized_config_and_parser
    with pytest.raises(Exception):
        parse_arguments(parser, args=arg, fail_on_unknown_args=True)


@pytest.mark.fast
@pytest.mark.parametrize(
    "args, expected_key, expected_value, expected_pass",
    [
        (["--name=foo"], "name", "foo", True),
        (["--seed", "42"], "seed", 42, True),
        (["--seed", ""], "seed", 42, True),
        (["--number", "2.17"], "number", 2.17, True),
        (["--number", ""], "number", 3.14, True),
        (["--integers", "1,2,3"], "integers", [1, 2, 3], True),
        (["--optional_int", ""], "optional_int", None, True),
        (["--optional_int", "2"], "optional_int", 2, True),
        (["--optional_float", ""], "optional_float", None, True),
        (["--optional_float", "3.14"], "optional_float", 3.14, True),
        (["--tuple1", "1,2"], "tuple1", (1, 2.0), True),
        (["--int_tuple", "1,2,3"], "int_tuple", (1, 2, 3), True),
        (["--enum=2"], "enum", ParamEnum.EnumValue2, True),
        (["--enum=3"], "enum", None, False),
        (["--floats=1,2,3.14"], "floats", [1.0, 2.0, 3.14], True),
        (["--integers=1,2,3"], "integers", [1, 2, 3], True),
        (["--flag"], "flag", True, True),
        (["--no-flag"], None, None, False),
        (["--not_flag"], None, None, False),
        (["--no-not_flag"], "not_flag", False, True),
        (["--not_flag=false", "--no-not_flag"], None, None, False),
        (["--flag=Falsf"], None, None, False),
        (["--flag=Truf"], None, None, False),
        (["--other_args={'learning_rate': 0.5}"], "other_args", {'learning_rate': 0.5}, True),
        (["--other_args=['foo']"], "other_args", ["foo"], True),
        (["--other_args={'learning':3"], None, None, False),
        (["--other_args=['foo','bar'"], None, None, False),
    ],
)
@pytest.mark.fast
def test_create_parser(
    parameterized_config_and_parser: Tuple[ParamClass, ArgumentParser],
    args: List[str],
    expected_key: str,
    expected_value: Any,
    expected_pass: bool,
) -> None:
    """
    Check that parse_args works as expected, with both non default and default values.
    """
    if expected_pass:
        check_parsing_succeeds(parameterized_config_and_parser, args, expected_key, expected_value)
    else:
        check_parsing_fails(parameterized_config_and_parser, args)


@pytest.mark.fast
@pytest.mark.parametrize(
    "flag, expected_value",
    [
        ('on', True),
        ('t', True),
        ('true', True),
        ('y', True),
        ('yes', True),
        ('1', True),
        ('off', False),
        ('f', False),
        ('false', False),
        ('n', False),
        ('no', False),
        ('0', False),
    ],
)
@pytest.mark.fast
def test_parsing_bools(
    parameterized_config_and_parser: Tuple[ParamClass, ArgumentParser], flag: str, expected_value: bool
) -> None:
    """
    Check all the ways of passing in True and False, with and without the first letter capitialized
    """
    check_parsing_succeeds(parameterized_config_and_parser, [f"--flag={flag}"], "flag", expected_value)
    check_parsing_succeeds(parameterized_config_and_parser, [f"--flag={flag.capitalize()}"], "flag", expected_value)
    check_parsing_succeeds(parameterized_config_and_parser, [f"--not_flag={flag}"], "not_flag", expected_value)
    check_parsing_succeeds(
        parameterized_config_and_parser, [f"--not_flag={flag.capitalize()}"], "not_flag", expected_value
    )


@pytest.mark.fast
def test_argparse_usage(capsys: pytest.CaptureFixture) -> None:
    """Test if the auto-generated argument parser prints out defaults and usage information."""

    class SimpleClass(param.Parameterized):
        name: str = param.String(default="name_default", doc="Name description")

    config = SimpleClass()
    parser = create_argparser(config, usage="my_usage", description="my_description", epilog="my_epilog")
    arguments = ["", "--help"]
    with pytest.raises(SystemExit):
        with patch.object(sys, "argv", arguments):
            parser.parse_args()
    stdout: str = capsys.readouterr().out  # type: ignore
    assert "Name description" in stdout
    assert "default: " in stdout
    assert "optional arguments:" in stdout
    assert "--name NAME" in stdout
    assert "usage: my_usage" in stdout
    assert "my_description" in stdout
    assert "my_epilog" in stdout


@pytest.mark.fast
@pytest.mark.parametrize(
    "args, expected_key, expected_value",
    [
        (["--strings=[]"], "strings", ['[]']),
        (["--strings=['']"], "strings", ["['']"]),
        (["--strings=None"], "strings", ['None']),
        (["--strings='None'"], "strings", ["'None'"]),
        (["--strings=','"], "strings", ["'", "'"]),
        (["--strings=''"], "strings", ["''"]),
        (["--strings=,"], "strings", []),
        (["--strings="], "strings", []),
        (["--integers="], "integers", []),
        (["--floats="], "floats", []),
    ],
)
@pytest.mark.fast
def test_override_list(
    parameterized_config_and_parser: Tuple[ParamClass, ArgumentParser],
    args: List[str],
    expected_key: str,
    expected_value: Any,
) -> None:
    """Test different options of overriding a non-empty list parameter to get an empty list"""
    check_parsing_succeeds(parameterized_config_and_parser, args, expected_key, expected_value)


@pytest.mark.fast
def test_argparse_usage_empty(capsys: CaptureFixture) -> None:
    """Test if the auto-generated argument parser prints out defaults and auto-generated usage information."""

    class SimpleClass(param.Parameterized):
        name: str = param.String(default="name_default", doc="Name description")

    config = SimpleClass()
    parser = create_argparser(config)
    arguments = ["", "--help"]
    with pytest.raises(SystemExit):
        with patch.object(sys, "argv", arguments):
            parser.parse_args()
    stdout: str = capsys.readouterr().out  # type: ignore
    assert "usage: " in stdout
    # Check if the auto-generated usage text is present
    assert "[-h] [--name NAME]" in stdout
    assert "optional arguments:" in stdout
    assert "--name NAME" in stdout
    assert "Name description" in stdout
    assert "default: " in stdout


@pytest.mark.fast
def test_apply_overrides(parameterized_config_and_parser: Tuple[ParamClass, ArgumentParser]) -> None:
    """
    Test that overrides are applied correctly, ond only to overridable parameters
    """
    parameterized_config = parameterized_config_and_parser[0]
    with patch("health_azure.argparsing.report_on_overrides") as mock_report_on_overrides:
        overrides = {"name": "newName", "int_tuple": (0, 1, 2)}
        actual_overrides = apply_overrides(parameterized_config, overrides)
        assert actual_overrides == overrides
        assert all([x == i and isinstance(x, int) for i, x in enumerate(parameterized_config.int_tuple)])
        assert parameterized_config.name == "newName"

        # Attempt to change seed and constant, but the latter should be ignored.
        change_seed = {"seed": 123}
        old_constant = parameterized_config.constant
        extra_overrides = {**change_seed, "constant": "Nothing"}  # type: ignore
        changes2 = apply_overrides(parameterized_config, overrides_to_apply=extra_overrides)  # type: ignore
        assert changes2 == change_seed
        assert parameterized_config.seed == 123
        assert parameterized_config.constant == old_constant

        # Check the call count of mock_validate and check it doesn't increase if should_validate is set to False
        # and that setting this flag doesn't affect on the outputs
        # mock_validate_call_count = mock_validate.call_count
        actual_overrides = apply_overrides(parameterized_config, overrides_to_apply=overrides, should_validate=False)
        assert actual_overrides == overrides
        # assert mock_validate.call_count == mock_validate_call_count

        # Check that report_on_overrides has not yet been called, but is called if keys_to_ignore is not None
        # and that setting this flag doesn't affect on the outputs
        assert mock_report_on_overrides.call_count == 0
        actual_overrides = apply_overrides(parameterized_config, overrides_to_apply=overrides, keys_to_ignore={"name"})
        assert actual_overrides == overrides
        assert mock_report_on_overrides.call_count == 1


@pytest.mark.fast
def test_report_on_overrides(
    parameterized_config_and_parser: Tuple[ParamClass, ArgumentParser], caplog: LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    parameterized_config = parameterized_config_and_parser[0]
    old_logs = caplog.messages
    assert len(old_logs) == 0
    # the following overrides are expected to cause logged warnings because
    # a) parameter 'constant' is constant
    # b) parameter 'readonly' is readonly
    # b) parameter 'idontexist' is undefined (not the name of a parameter of ParamClass)
    overrides = {"constant": "dif_value", "readonly": "new_value", "idontexist": (0, 1, 2)}
    keys_to_ignore: Set = set()
    report_on_overrides(parameterized_config, overrides, keys_to_ignore)
    # Expect one warning message per failed override
    new_logs = caplog.messages
    expected_warnings = len(overrides.keys())
    assert len(new_logs) == expected_warnings, f"Expected {expected_warnings} warnings but found: {caplog.records}"


@pytest.mark.fast
@pytest.mark.parametrize("value_idx_0", [1.0, 1])
@pytest.mark.parametrize("value_idx_1", [2.0, 2])
@pytest.mark.parametrize("value_idx_2", [3.0, 3])
def test_int_tuple_validation(
    value_idx_0: Any,
    value_idx_1: Any,
    value_idx_2: Any,
    parameterized_config_and_parser: Tuple[ParamClass, ArgumentParser],
) -> None:
    """
    Test integer tuple parameter is validated correctly.
    """
    parameterized_config = parameterized_config_and_parser[0]
    val = (value_idx_0, value_idx_1, value_idx_2)
    if not all([isinstance(x, int) for x in val]):
        with pytest.raises(ValueError):
            parameterized_config.int_tuple = (value_idx_0, value_idx_1, value_idx_2)
    else:
        parameterized_config.int_tuple = (value_idx_0, value_idx_1, value_idx_2)


class IllegalParamClassNoString(param.Parameterized):
    custom_type_no_from_string = IllegalCustomTypeNoFromString(
        None, doc="This should fail since from_string method is missing"
    )


@pytest.mark.fast
def test_cant_parse_param_type() -> None:
    """
    Assert that a TypeError is raised when trying to add a custom type with no from_string method as an argument
    """
    config = IllegalParamClassNoString()

    with pytest.raises(TypeError) as e:
        create_argparser(config)
        assert "is not supported" in str(e.value)


# Another custom type (from docs/source/conmmandline_tools.md)
class EvenNumberParam(CustomTypeParam):
    """Our custom type param for even numbers"""

    def _validate(self, val: Any) -> None:
        if (not self.allow_None) and val is None:
            raise ValueError("Value must not be None")
        if val % 2 != 0:
            raise ValueError(f"{val} is not an even number")
        super()._validate(val)  # type: ignore

    def from_string(self, x: str) -> int:
        return int(x)


class MyScriptConfig(param.Parameterized):
    simple_string: str = param.String(default="")
    even_number: int = EvenNumberParam(2, doc="your choice of even number", allow_None=False)


@pytest.mark.fast
def test_parse_args_and_apply_overrides() -> None:
    config = MyScriptConfig()
    assert config.even_number == 2
    assert config.simple_string == ""

    new_even_number = config.even_number * 2
    new_string = config.simple_string + "something_new"
    config_w_results = parse_args_and_update_config(
        config, ["--even_number", str(new_even_number), "--simple_string", new_string]
    )
    assert config_w_results.even_number == new_even_number
    assert config_w_results.simple_string == new_string

    # parsing args with unaccepted values should cause an exception to be raised
    odd_number = new_even_number + 1
    with pytest.raises(ValueError) as e:
        parse_args_and_update_config(config, args=["--even_number", f"{odd_number}"])
        assert "not an even number" in str(e.value)

    none_number = "None"
    with pytest.raises(ArgumentError):
        parse_args_and_update_config(config, args=["--even_number", f"{none_number}"])

    # Mock from_string to check test _validate
    def mock_from_string_none(a: Any, b: Any) -> None:
        return None  # type: ignore

    with patch.object(EvenNumberParam, "from_string", new=mock_from_string_none):
        # Check that _validate fails with None value
        with pytest.raises(ValueError) as e:
            parse_args_and_update_config(config, ["--even_number", f"{none_number}"])
            assert "must not be None" in str(e.value)


@pytest.mark.fast
def test_parse_illegal_params() -> None:
    with pytest.raises(TypeError) as e:
        ParamClass(readonly="abc")
    assert "cannot be modified" in str(e.value)


@pytest.mark.fast
def test_config_add_and_validate() -> None:
    config = ParamClass()
    assert config.name.startswith("ParamClass")
    set_fields_and_validate(config, {"name": "foo"})
    assert config.name == "foo"

    assert hasattr(config, "new_property") is False
    set_fields_and_validate(config, {"new_property": "bar"})
    assert hasattr(config, "new_property") is True
    assert config.new_property == "bar"


@pytest.fixture(scope="module")
def dummy_model_config() -> DummyConfig:
    string_param = "dummy"
    int_param = 1
    return DummyConfig(param1=string_param, param2=int_param)


@pytest.mark.fast
def test_add_and_validate(dummy_model_config: DummyConfig) -> None:
    new_string_param = "new_dummy"
    new_int_param = 2
    new_args = {"string_param": new_string_param, "int_param": new_int_param}
    set_fields_and_validate(dummy_model_config, new_args)

    assert dummy_model_config.string_param == new_string_param
    assert dummy_model_config.int_param == new_int_param


@pytest.mark.fast
def test_duplicate_enum() -> None:
    """Test parsing of enum values where the values are not unique when lower cased."""

    class DuplicateEnum(Enum):
        A = "abc"
        B = "ABC"

    with pytest.raises(ValueError, match="Enum values must be unique when lower cased. Duplicate: abc"):
        _enum_from_string(DuplicateEnum)


@pytest.mark.fast
def test_enum_from_string() -> None:
    """Test converting from strings to Enum cases."""

    class MyEnum(Enum):
        A = 1
        B = 2

    parser = _enum_from_string(MyEnum)
    with pytest.raises(ValueError, match="Invalid value 'A' for Enum MyEnum. Must be one of 1, 2"):
        parser("A")

    assert parser("1") == MyEnum.A
    assert parser("2") == MyEnum.B


@pytest.mark.fast
def test_parse_enum_from_param() -> None:
    class MyEnum(Enum):
        A = 1
        B = 2

    class MyParam(param.Parameterized):
        my_enum = param.ClassSelector(class_=MyEnum, default=MyEnum.A)

    config = MyParam()
    parse_args_and_update_config(config, ["--my_enum", "2"])
    assert config.my_enum == MyEnum.B

    with pytest.raises(ArgumentError, match="argument --my_enum: invalid parse_enum value: 'A'"):
        parse_args_and_update_config(config, ["--my_enum", "A"])
