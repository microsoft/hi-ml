from typing import Any, Collection, Mapping

import numpy as np
import torch


def assert_dicts_equal(d1: Mapping, d2: Mapping, exclude_keys: Collection[Any] = (),
                       rtol: float = 1e-5, atol: float = 1e-8) -> None:
    assert isinstance(d1, Mapping)
    assert isinstance(d2, Mapping)
    keys1 = [key for key in d1 if key not in exclude_keys]
    keys2 = [key for key in d2 if key not in exclude_keys]
    assert keys1 == keys2
    for key in keys1:
        msg = f"Dictionaries differ for key '{key}': {d1[key]} vs {d2[key]}"
        if isinstance(d1[key], torch.Tensor):
            assert torch.allclose(d1[key], d2[key], rtol=rtol, atol=atol, equal_nan=True), msg
        elif isinstance(d1[key], np.ndarray):
            assert np.allclose(d1[key], d2[key], rtol=rtol, atol=atol, equal_nan=True), msg
        else:
            assert d1[key] == d2[key], msg
