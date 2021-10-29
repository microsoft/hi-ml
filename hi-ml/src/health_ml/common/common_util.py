#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Iterable


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
