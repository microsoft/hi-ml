#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Iterable, List, Optional


def any_smaller_or_equal_than(items: Iterable[Any], scalar: float) -> bool:
    """
    Returns True if any of the elements of the list is smaller than the given scalar number.
    """
    return any(item < scalar for item in items)


def any_pairwise_larger(items1: Any, items2: Any) -> bool:
    """
    Returns True if any of the elements of items1 is larger than the corresponding element in items2.
    The two lists must have the same length.
    """
    if len(items1) != len(items2):
        raise ValueError(f"Arguments must have the same length. len(items1): {len(items1)}, len(items2): {len(items2)}")
    for i in range(len(items1)):
        if items1[i] > items2[i]:
            return True
    return False


def check_is_any_of(message: str, actual: Optional[str], valid: Iterable[Optional[str]]) -> None:
    """
    Raises an exception if 'actual' is not any of the given valid values.
    :param message: The prefix for the error message.
    :param actual: The actual value.
    :param valid: The set of valid strings that 'actual' is allowed to take on.
    :return:
    """
    if actual not in valid:
        all_valid = ", ".join(["<None>" if v is None else v for v in valid])
        raise ValueError("{} must be one of [{}], but got: {}".format(message, all_valid, actual))


def check_properties_are_not_none(obj: Any, ignore: Optional[List[str]] = None) -> None:
    """
    Checks to make sure the provided object has no properties that have a None value assigned.
    """
    if ignore is not None:
        none_props = [k for k, v in vars(obj).items() if v is None and k not in ignore]
        if len(none_props) > 0:
            raise ValueError("Properties had None value: {}".format(none_props))
