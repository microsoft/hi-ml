#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import param
import pytest
from _pytest.logging import LogCaptureFixture

from health_azure.traverse import (
    _object_to_dict,
    get_all_writable_attributes,
    is_basic_type,
    object_to_dict,
    object_to_yaml,
    yaml_to_dict,
    write_dict_to_object,
    write_yaml_to_object,
    _write_dict_to_object,
)


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    optimizer: Optional[str] = "Adam"


@dataclass
class TransformConfig:
    blur_sigma: float = 0.1
    blur_p: float = 0.2


@dataclass
class FullConfig:
    optimizer: OptimizerConfig = OptimizerConfig()
    transforms: TransformConfig = TransformConfig()


@dataclass
class TripleNestedConfig:
    float1: float = 1.0
    int1: int = 1
    nested: FullConfig = FullConfig()


@dataclass
class ConfigWithList:
    list_field: Union[List, Dict] = field(default_factory=list)


class ParamsConfig(param.Parameterized):
    p1: int = param.Integer(default=1)
    p2: str = param.String(default="foo")
    p3: float = param.Number(default=2.0)


class ParamsConfigWithReadonly(param.Parameterized):
    p1: int = param.Integer(default=1, readonly=True)
    p2: str = param.String(default="foo", constant=True)
    p3: float = param.Number(default=2.0)
    _p4: float = param.Number(default=4.0)


class MyEnum(enum.Enum):
    foo = "foo_value"
    bar = "bar_value"


@dataclass
class ConfigWithEnum:
    f1: MyEnum


def test_traverse1() -> None:
    config = TransformConfig()
    d = object_to_dict(config)
    assert d == {"blur_sigma": 0.1, "blur_p": 0.2}


def test_traverse2() -> None:
    config = FullConfig()
    d = object_to_dict(config)
    assert d == {
        "optimizer": {"learning_rate": 1e-3, "optimizer": "Adam"},
        "transforms": {"blur_sigma": 0.1, "blur_p": 0.2},
    }


def test_traverse_params() -> None:
    config = ParamsConfig()
    d = object_to_dict(config)
    assert d == {"p1": 1, "p2": "foo", "p3": 2.0}


def test_traverse_params_readonly() -> None:
    """
    Private, constant and readonly fields of a Params object should be skipped.
    """
    config = ParamsConfigWithReadonly()
    d = get_all_writable_attributes(config)
    assert d == {"p3": 2.0}


def test_traverse_list() -> None:
    """Lists of basic datatypes should be serialized"""
    list_config = ConfigWithList(list_field=[1, 2.5, "foo"])
    assert object_to_yaml(list_config) == """list_field:
- 1
- 2.5
- foo
"""
    list_config.list_field = [1, dict()]
    with pytest.raises(ValueError) as ex:
        object_to_dict(list_config)
    assert "only allowed to contain basic types" in str(ex)


def test_is_basic() -> None:
    assert is_basic_type(1)
    assert is_basic_type(1.0)
    assert is_basic_type("foo")
    assert not is_basic_type(None)
    assert not is_basic_type(dict())
    assert not is_basic_type([])


def test_traverse_dict() -> None:
    """Fields with dictionaries with basic datatypes should be serialized"""
    list_config = ConfigWithList(list_field={1: "foo", "bar": 2.0})
    assert object_to_yaml(list_config) == """list_field:
  1: foo
  bar: 2.0
"""
    # Invalid dictionaries contain non-basic types either as keys or as values
    for invalid in [{"foo": dict()}, {(1, 2): "foo"}]:
        list_config.list_field = invalid  # type: ignore
        with pytest.raises(ValueError) as ex:
            object_to_dict(list_config)
        assert "Dictionaries can only contain basic types" in str(ex)


def test_traverse_enum() -> None:
    """
    Enum objects have not non-private fields, and should return an empty value dictionary.
    """
    config = MyEnum.foo
    attributes = get_all_writable_attributes(config)
    assert len(attributes) == 0
    # Enum objects should be printed out by their case name
    d = _object_to_dict(config)
    assert d == "foo"


def test_traverse_none() -> None:
    """
    Attributes that are None should be preserved as such in YAML.
    """
    config = OptimizerConfig()
    config.optimizer = None
    assert _object_to_dict(None) is None
    assert _object_to_dict(config) == {"learning_rate": 1e-3, "optimizer": None}
    assert object_to_yaml(config) == """learning_rate: 0.001
optimizer: null
"""


def test_to_yaml_rountrip() -> None:
    config = FullConfig()
    yaml = object_to_yaml(config)
    print("\n" + yaml)
    dict = yaml_to_dict(yaml)
    assert dict == object_to_dict(config)


def test_params_roundtrip() -> None:
    config = ParamsConfig()
    yaml = object_to_yaml(config)
    print("\n" + yaml)
    dict = yaml_to_dict(yaml)
    assert dict == object_to_dict(config)


@pytest.mark.parametrize("my_list", [[1, "foo"], [], {1: 2.0}, {}])
def test_list_dict_roundtrip(my_list: Any) -> None:
    """Test if configs with lists and dicts can be converted to YAML and back"""
    my_list = [1, "foo"]
    config = ConfigWithList(list_field=my_list)
    yaml = object_to_yaml(config)
    print("\n" + yaml)
    # Initialize the new object with a non-empty list, to see if the empty gets correctly written
    config2 = ConfigWithList(list_field=["not_in_args"])
    write_yaml_to_object(config2, yaml)
    assert config2.list_field == my_list


def test_write_flat() -> None:
    obj = OptimizerConfig()
    learning_rate = 3.0
    optimizer = "Foo"
    d = {"learning_rate": learning_rate, "optimizer": optimizer}
    write_dict_to_object(obj, d)
    assert obj.learning_rate == learning_rate
    assert obj.optimizer == optimizer


def test_write_nested() -> None:
    obj = TripleNestedConfig()
    yaml = object_to_yaml(obj)
    print("\n" + yaml)
    float1 = 2.0
    int1 = 2
    blur_p = 7.0
    blur_sigma = 8.0
    learning_rate = 3.0
    optimizer = "Foo"
    d1 = {"transforms": {"blur_sigma": blur_sigma, "blur_p": blur_p},
          "optimizer": {"learning_rate": learning_rate, "optimizer": optimizer}}
    d = {"float1": float1, "int1": int1, "nested": d1}
    write_dict_to_object(obj, d)
    assert obj.float1 == float1
    assert obj.int1 == int1
    assert obj.nested.optimizer.optimizer == optimizer
    assert obj.nested.optimizer.learning_rate == learning_rate
    assert obj.nested.transforms.blur_p == blur_p
    assert obj.nested.transforms.blur_sigma == blur_sigma

    from_obj = object_to_dict(obj)
    assert from_obj == d

    yaml = object_to_yaml(obj)

    print("\n" + yaml)
    obj2 = TripleNestedConfig()
    write_yaml_to_object(obj2, yaml_string=yaml)
    print("\n" + repr(obj2))
    from_yaml_obj = object_to_dict(obj2)
    assert from_yaml_obj == d


def test_yaml_and_write_roundtrip() -> None:
    obj = ParamsConfig()
    obj.p1 = 2
    obj.p2 = "nothing"
    obj.p3 = 3.14
    yaml = object_to_yaml(obj)
    print("\n" + yaml)
    obj2 = ParamsConfig()
    write_yaml_to_object(obj2, yaml_string=yaml)
    assert obj2.p1 == obj.p1
    assert obj2.p2 == obj.p2
    assert obj2.p3 == obj.p3


def test_to_yaml_datatypes() -> None:
    """
    Ensure that string fields that look like numbers are treated correctly.
    """
    config = OptimizerConfig()
    config.optimizer = "2"
    yaml = object_to_yaml(config)
    print("\n" + yaml)
    dict = yaml_to_dict(yaml)
    assert dict == object_to_dict(config)


def test_object_to_yaml_floats() -> None:
    """
    Check that floating point numbers that could be mistaken for integers are in YAML unambiguously.
    """
    config = TransformConfig()
    config.blur_p = 1.0
    yaml = object_to_yaml(config)
    assert yaml == """blur_p: 1.0
blur_sigma: 0.1
"""


def test_write_dict_errors1(caplog: LogCaptureFixture) -> None:
    """
    Check type mismatches between object and YAML contents
    :return:
    """
    # First test the private writer method
    config = TransformConfig()
    dict = {"blur_p": 1, "blur_sigma": 0.1}
    errors = _write_dict_to_object(config, dict)
    assert len(errors) == 1
    assert "Attribute blur_p" in errors[0]
    assert "Skipped" in errors[0]
    assert "Current value has type float" in errors[0]
    assert "trying to write int" in errors[0]

    # The same error message should be raised when calling the full method, with either strict or non-strict
    config = TransformConfig()
    with caplog.at_level(level=logging.WARNING):
        with pytest.raises(ValueError) as ex:
            write_dict_to_object(config, dict, strict=True)
    assert "Found 1 problems" in str(ex)
    assert errors[0] in caplog.text

    config = TransformConfig()
    with caplog.at_level(level=logging.WARNING):
        write_dict_to_object(config, dict, strict=False)
    assert errors[0] in caplog.text


def test_write_dict_errors2(caplog: LogCaptureFixture) -> None:
    """
    Check type mismatches between object and YAML contents, and that field names are correctly handled
    :return:
    """
    caplog.set_level(logging.WARNING)
    config = FullConfig()
    dict = object_to_dict(config)
    dict["transforms"]["blur_p"] = "foo"
    dict["optimizer"]["learning_rate"] = "bar"
    write_dict_to_object(config, dict, strict=False)
    assert "Found 2 problems" in caplog.text
    assert "Attribute transforms.blur_p" in caplog.text
    assert "Attribute optimizer.learning_rate" in caplog.text


def test_write_dict_errors3() -> None:
    """
    Check handling of cases where the object has more fields than present in the dictionary
    :return:
    """
    config = TransformConfig()
    dict = {"blur_p": 1.0}
    issues = _write_dict_to_object(config, dict)
    assert len(issues) == 1
    assert "Present in the object, but missing in the dictionary" in issues[0]
    assert "Attribute blur_sigma" in issues[0]


def test_write_enums() -> None:
    """
    Test handling of fields that are enum types
    """
    config = ConfigWithEnum(f1=MyEnum.bar)
    d = object_to_dict(config)
    assert d == {"f1": "bar"}
    # Now change F1 to be another enum value, write back the previous dictionary, and check if F1 is back at the
    # original value
    config.f1 = MyEnum.foo
    write_dict_to_object(config, d)
    assert config.f1 == MyEnum.bar


def test_write_enums_errors() -> None:
    """
    Error handling for enum fields
    """
    config = ConfigWithEnum(f1=MyEnum.bar)
    # Trying to write a floating point number, but we expect a string case name
    issues = _write_dict_to_object(config, {"f1": 1.0})
    assert len(issues) == 1
    assert "Enum field" in issues[0]
    assert "got value of type float" in issues[0]
    # Referencing an enum case that does not exist
    issues = _write_dict_to_object(config, {"f1": "no_such_case"})
    assert len(issues) == 1
    assert "Enum type MyEnum has no case no_such_case" in issues[0]
    # This should work fine
    issues = _write_dict_to_object(config, {"f1": MyEnum.bar.name})
    assert len(issues) == 0


def test_write_null() -> None:
    """
    Round-trip test for writing fields that are None.
    """
    config = OptimizerConfig()
    config.optimizer = None
    dict = object_to_dict(config)
    yaml = object_to_yaml(config)
    config.optimizer = "foo"
    issues = _write_dict_to_object(config, dict)
    assert len(issues) == 0
    assert config.optimizer is None
    config.optimizer = "foo"
    write_yaml_to_object(config, yaml)
    assert config.optimizer is None
