#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from io import StringIO
from typing import Any, Dict, Union, List, Optional

import param
from ruamel import yaml


def is_basic_field(o: Any) -> bool:
    """
    Returns True if the given object is an instance of a basic simple datatype: string, integer, float.
    """
    return isinstance(o, (str, int, float))


def get_all_writable_attributes(o: Any) -> Dict[str, Any]:
    """
    Returns all writable attributes of an object, by resorting to the "vars" method. For object that derive
    from param.Parameterized, it returns all params that are not constant and not readonly.

    :param o: The object to inspect.
    :return: A dictionary mapping from attribute name to its value.
    """
    if isinstance(o, param.Parameterized):
        result = {}
        for param_name, p in o.params().items():
            if not p.constant and not p.readonly:
                result[param_name] = getattr(o, param_name)
        return result
    try:
        return vars(o)
    except TypeError:
        raise ValueError("This function cannot be used on objects that do not support the 'vars' operation")


def _object_to_dict(o: Any) -> Union[str, Dict[str, Any]]:
    """
    Converts an object to a dictionary mapping from attribute name to value. That value can be a dictionary recursively,
    if the attribute is not a simple datatype.

    :param o: The object to inspect.
    :return: Returns the argument if the object is a basic datatype, otherwise a dictionary mapping from attribute
    name to value.
    """
    if is_basic_field(o):
        return o
    try:
        fields = get_all_writable_attributes(o)
        return {field: _object_to_dict(value) for field, value in fields.items()}
    except ValueError:
        return repr(o)


def object_to_dict(o: Any) -> Dict[str, Any]:
    """
    Converts an object to a dictionary mapping from attribute name to value. That value can be a dictionary recursively,
    if the attribute is not a simple datatype.
    This function only works on objects that are not basic datatype (i.e., classes)
    :param o: The object to inspect.
    :return: Returns the argument if the object is a basic datatype, otherwise a dictionary mapping from attribute
    name to value.
    :raises ValueError: If the argument is a b
    """
    if is_basic_field(o):
        raise ValueError("This function cannot be used on objects that are basic datatypes.")
    fields = get_all_writable_attributes(o)
    return {field: _object_to_dict(value) for field, value in fields.items()}


def object_to_yaml(o: Any) -> str:
    """
    Converts an object to a YAML string representation. This is done by recursively traversing all attributes and
    writing them out to YAML if they are basic datatypes.

    :param o: The object to inspect.
    :return: A string in YAML format.
    """
    return yaml.safe_dump(object_to_dict(o), default_flow_style=False)


def yaml_to_dict(s: str) -> Dict[str, Any]:
    """
    Interprets a string as YAML and returns the contents as a dictionary.

    :param s: The YAML string to parse.
    :return: A dictionary where the keys are the YAML field names, and values are either the YAML leaf node values,
    or dictionaries again.
    """
    stream = StringIO(s)
    return yaml.safe_load(stream=stream)


def _write_dict_to_object(o: Any, d: Dict[str, Any],
                          strict: bool = False,
                          traversed_fields: Optional[List] = None) -> List[str]:
    """
    Writes a dictionary of values into an object, assuming that the attributes of the object and the dictionary keys
    are in sync. For example, to write a dictionary {"foo": 1, "bar": "baz"} to an object, the object needs to have
    attributes "foo" and "bar".

    :param strict: If True, any mismatch of field names will raise a ValueError. If False, only a warning will be
    printed.
    :param o: The object to write to.
    :param d: A dictionary mapping from attribute names to values or dictionaries recursively.
    :return: A list of error messages collected.
    """
    issues: List[str] = []
    traversed = traversed_fields or []

    def report_issue(name, message: str) -> None:
        full_field_name = ".".join(traversed + [name])
        issues.append(f"Attribute {full_field_name}: {message}")

    existing_attrs = get_all_writable_attributes(o)
    for name, value in existing_attrs.items():
        if name in d:
            value_to_write = d[name]
            t_value = type(value)
            t_value_to_write = type(value_to_write)
            if is_basic_field(value) and is_basic_field(value_to_write):
                if t_value != t_value_to_write:
                    report_issue(name, f"Skipped. Current value has type {t_value.__name__}, but trying to "
                                       f"write {t_value_to_write.__name__}")
                setattr(o, name, value_to_write)
            elif not is_basic_field(value) and isinstance(value_to_write, Dict):
                new_issues = _write_dict_to_object(getattr(o, name), value_to_write, traversed_fields=traversed)
                issues.extend(new_issues)
            else:
                report_issue(name, f"Skipped. Current value has type {t_value.__name__}, but trying to "
                                   f"write {t_value_to_write.__name__}")
        else:
            report_issue(name, "Present in the object, but missing in the dictionary.")

    return issues


def write_dict_to_object(o: Any, d: Dict[str, Any],
                         strict: bool = False) -> None:
    """
    Writes a dictionary of values into an object, assuming that the attributes of the object and the dictionary keys
    are in sync. For example, to write a dictionary {"foo": 1, "bar": "baz"} to an object, the object needs to have
    attributes "foo" and "bar".

    :param strict: If True, any mismatch of field names will raise a ValueError. If False, only a warning will be
    printed. Note that the object may have been modified even if an error is raised.
    :param o: The object to write to.
    :param d: A dictionary mapping from attribute names to values or dictionaries recursively.
    """
    issues = _write_dict_to_object(o, d, strict=strict)
    message = f"Unable to complete writing to the object: Found {len(issues)} problems:"
    full_message = "\n".join([message] + issues)
    if strict:
        raise ValueError(full_message)
    else:
        logging.warning(full_message)


def write_yaml_to_object(o: Any, yaml_string: str) -> None:
    d = yaml_to_dict(yaml_string)
    write_dict_to_object(o, d)
