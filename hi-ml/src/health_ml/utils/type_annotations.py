#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Tuple, TypeVar, Union

T = TypeVar('T')
PathOrString = Union[Path, str]
TupleFloat2 = Tuple[float, float]
DictStrFloat = Dict[str, float]
DictStrFloatOrFloatList = Dict[str, Union[float, List[float]]]
