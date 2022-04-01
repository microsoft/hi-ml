#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import enum
import logging
from io import StringIO
from typing import Any, Dict, Iterable, Union, List, Optional

import param
from ruamel import yaml


def is_basic_type(o: Any) -> bool:
    """
    Returns True if the given object is an instance of a basic simple datatype: string, integer, float.
    """
    return isinstance(o, (str, int, float))


def is_enum(o: Any) -> bool:
    """
    Returns True if the given object is a subclass of enum.Enum.

    :param o: The object to inspect.
    :return: True if the object is an enum, False otherwise.
    """
    return isinstance(o, enum.Enum)


def get_all_writable_attributes(o: Any) -> Dict[str, Any]:
    """
    Returns all writable attributes of an object, by resorting to the "vars" method. For object that derive
    from param.Parameterized, it returns all params that are not constant and not readonly.

    :param o: The object to inspect.
    :return: A dictionary mapping from attribute name to its value.
    """

    def _is_private(s: str) -> bool:
        return s.startswith("_")

    result = {}
    if isinstance(o, param.Parameterized):
        for param_name, p in o.params().items():
            if _is_private(param_name):
                logging.debug(f"get_all_writable_attributes: Skipping private field {param_name}")
            elif p.constant:
                logging.debug(f"get_all_writable_attributes: Skipping constant field {param_name}")
            elif p.readonly:
                logging.debug(f"get_all_writable_attributes: Skipping readonly field {param_name}")
            else:
                result[param_name] = getattr(o, param_name)
        return result
    try:
        for name, value in vars(o).items():
            if _is_private(name):
                logging.debug(f"get_all_writable_attributes: Skipping private field {name}")
            else:
                result[name] = value
        return result
    except TypeError:
        raise ValueError("This function can only be used on objects that support the 'vars' operation")


def all_basic_types(o: Iterable) -> bool:
    """Checks if all entries of the iterable are of a basic datatype (int, str, float).

    :param o: The iterable that should be checked.
    :return: True if all entries of the iterable are of a basic datatype.
    """
    for item in o:
        if not is_basic_type(item):
            return False
    return True


def _object_to_dict(o: Any) -> Union[None, int, float, str, List, Dict]:
    """
    Converts an object to a dictionary mapping from attribute name to value. That value can be a dictionary recursively,
    if the attribute is not a simple datatype. Lists and dictionaries are returned as-is.

    :param o: The object to inspect.
    :return: Returns the argument if the object is a basic datatype, otherwise a dictionary mapping from attribute
    name to value.
    """
    if is_basic_type(o):
        return o
    if isinstance(o, enum.Enum):
        return o.name
    if isinstance(o, list):
        if not all_basic_types(o):
            raise ValueError(f"Lists are only allowed to contain basic types (int, float, str), but got: {o}")
        return o
    if isinstance(o, dict):
        if not all_basic_types(o.keys()):
            raise ValueError(f"Dictionaries can only contain basic types (int, float, str) as keys, but got: {o}")
        if not all_basic_types(o.values()):
            raise ValueError(f"Dictionaries can only contain basic types (int, float, str) as values, but got: {o}")
        return o
    if o is None:
        return o
    try:
        fields = get_all_writable_attributes(o)
        return {field: _object_to_dict(value) for field, value in fields.items()}
    except ValueError as ex:
        raise ValueError(f"Unable to traverse object {o}: {ex}")


def object_to_dict(o: Any) -> Dict[str, Any]:
    """
    Converts an object to a dictionary mapping from attribute name to value. That value can be a dictionary recursively,
    if the attribute is not a simple datatype.
    This function only works on objects that are not basic datatype (i.e., classes). Private fields (name starting with
    underscores), or ones that appear to be constant or readonly are omitted. For attributes that are Enums, the
    case name is returned as a string.

    :param o: The object to inspect.
    :return: Returns the argument if the object is a basic datatype, otherwise a dictionary mapping from attribute
    name to value.
    :raises ValueError: If the argument is a basic datatype (int, str, float)
    """
    if is_basic_type(o):
        raise ValueError("This function can only be used on objects that are basic datatypes.")
    fields = get_all_writable_attributes(o)
    result = {}
    for field, value in fields.items():
        logging.debug(f"object_to_dict: Processing {field}")
        result[field] = _object_to_dict(value)
    return result


def object_to_yaml(o: Any) -> str:
    """
    Converts an object to a YAML string representation. This is done by recursively traversing all attributes and
    writing them out to YAML if they are basic datatypes.

    :param o: The object to inspect.
    :return: A string in YAML format.
    """
    return yaml.safe_dump(object_to_dict(o), default_flow_style=False)  # type: ignore


def yaml_to_dict(s: str) -> Dict[str, Any]:
    """
    Interprets a string as YAML and returns the contents as a dictionary.

    :param s: The YAML string to parse.
    :return: A dictionary where the keys are the YAML field names, and values are either the YAML leaf node values,
    or dictionaries again.
    """
    stream = StringIO(s)
    return yaml.safe_load(stream=stream)


def _write_dict_to_object(o: Any, d: Dict[str, Any], traversed_fields: Optional[List] = None) -> List[str]:
    """
    Writes a dictionary of values into an object, assuming that the attributes of the object and the dictionary keys
    are in sync. For example, to write a dictionary {"foo": 1, "bar": "baz"} to an object, the object needs to have
    attributes "foo" and "bar".

    :param o: The object to write to.
    :param d: A dictionary mapping from attribute names to values or dictionaries recursively.
    :return: A list of error messages collected.
    """
    issues: List[str] = []
    traversed = traversed_fields or []

    def report_issue(name: str, message: str) -> None:
        full_field_name = ".".join(traversed + [name])
        issues.append(f"Attribute {full_field_name}: {message}")

    def try_set_field(name: str, value_to_write: Any) -> None:
        try:
            setattr(o, name, value_to_write)
        except Exception as ex:
            report_issue(name, f"Unable to set value {value_to_write}: {ex}")

    existing_attrs = get_all_writable_attributes(o)
    for name, value in existing_attrs.items():
        if name in d:
            value_to_write = d[name]
            t_value = type(value)
            t_value_to_write = type(value_to_write)
            if is_basic_type(value) and is_basic_type(value_to_write):
                if t_value != t_value_to_write:
                    report_issue(
                        name,
                        f"Skipped. Current value has type {t_value.__name__}, but trying to "
                        f"write {t_value_to_write.__name__}",
                    )
                try_set_field(name, value_to_write)
            elif isinstance(value, enum.Enum):
                if isinstance(value_to_write, str):
                    try:
                        enum_case = getattr(t_value, value_to_write)
                    except Exception:
                        report_issue(name, f"Skipped. Enum type {t_value.__name__} has no case {value_to_write}")
                    else:
                        try_set_field(name, enum_case)
                else:
                    report_issue(
                        name,
                        "Skipped. This is an Enum field. Can only write string values to that field "
                        f"(case name), but got value of type {t_value_to_write.__name__}",
                    )
            elif value is None or value_to_write is None:
                # We can't do much type checking if we get Nones. This is a potential source of errors.
                try_set_field(name, value_to_write)
            elif isinstance(value, List) and isinstance(value_to_write, List):
                try_set_field(name, value_to_write)
            elif isinstance(value, Dict) and isinstance(value_to_write, Dict):
                try_set_field(name, value_to_write)
            elif not is_basic_type(value) and isinstance(value_to_write, Dict):
                # For anything that is not a basic datatype, we expect that we get a dictionary of fields
                # recursively.
                new_issues = _write_dict_to_object(
                    getattr(o, name), value_to_write, traversed_fields=traversed + [name]
                )
                issues.extend(new_issues)
            else:
                report_issue(
                    name,
                    f"Skipped. Current value has type {t_value.__name__}, but trying to "
                    f"write {t_value_to_write.__name__}",
                )
        else:
            report_issue(name, "Present in the object, but missing in the dictionary.")

    return issues


def write_dict_to_object(o: Any, d: Dict[str, Any], strict: bool = True) -> None:
    """
    Writes a dictionary of values into an object, assuming that the attributes of the object and the dictionary keys
    are in sync. For example, to write a dictionary {"foo": 1, "bar": "baz"} to an object, the object needs to have
    attributes "foo" and "bar".

    :param strict: If True, any mismatch of field names will raise a ValueError. If False, only a warning will be
    printed. Note that the object may have been modified even if an error is raised.
    :param o: The object to write to.
    :param d: A dictionary mapping from attribute names to values or dictionaries recursively.
    """
    issues = _write_dict_to_object(o, d)
    if len(issues) == 0:
        return
    message = f"Unable to complete writing to the object: Found {len(issues)} problems. Please inspect console log."
    for issue in issues:
        logging.warning(issue)
    if strict:
        raise ValueError(message)
    else:
        logging.warning(message)


def write_yaml_to_object(o: Any, yaml_string: str, strict: bool = False) -> None:
    """
    Writes a serialized object in YAML format back into an object, assuming that the attributes of the object and
    the YAML field names are in sync.

    :param strict: If True, any mismatch of field names will raise a ValueError. If False, only a warning will be
    printed. Note that the object may have been modified even if an error is raised.
    :param o: The object to write to.
    :param yaml_string: A YAML formatted string with attribute names and values.
    """
    d = yaml_to_dict(yaml_string)
    write_dict_to_object(o, d, strict=strict)
