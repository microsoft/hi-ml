#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from io import StringIO
from typing import Any, Dict, Union
from ruamel import yaml


def is_basic_field(o: Any) -> bool:
    return isinstance(o, (str, int, float))


def _object_to_dict(o: Any) -> Union[str, Dict[str, Any]]:
    if is_basic_field(o):
        return o
    try:
        fields = vars(o)
        return {field: _object_to_dict(value) for field, value in fields.items()}
    except TypeError:
        return repr(o)


def object_to_dict(o: Any) -> Dict[str, Any]:
    if is_basic_field(o):
        raise ValueError("This function cannot be used on objects that are basic datatypes.")
    try:
        fields = vars(o)
    except TypeError:
        raise ValueError("This function cannot be used on objects that do not support the 'vars' operation")
    return {field: _object_to_dict(value) for field, value in fields.items()}


def object_to_yaml(o: Any) -> str:
    return yaml.safe_dump(object_to_dict(o), default_flow_style=False)


def yaml_to_dict(s: str) -> Dict[str, Any]:
    stream = StringIO(s)
    return yaml.safe_load(stream=stream)
