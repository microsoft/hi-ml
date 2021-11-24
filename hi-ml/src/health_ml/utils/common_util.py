from enum import Enum, unique
from typing import Any, List, Optional


@unique
class ModelExecutionMode(Enum):
    """
    Model execution mode
    """
    TRAIN = "Train"
    TEST = "Test"
    VAL = "Val"


def check_properties_are_not_none(obj: Any, ignore: Optional[List[str]] = None) -> None:
    """
    Checks to make sure the provided object has no properties that have a None value assigned.
    """
    if ignore is not None:
        none_props = [k for k, v in vars(obj).items() if v is None and k not in ignore]
        if len(none_props) > 0:
            raise ValueError("Properties had None value: {}".format(none_props))
