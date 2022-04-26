from typing import Optional, Tuple
import numpy as np


def interp_index_1d(index: int, values: np.ndarray) -> float:
    return values[0] + (values[-1] - values[0]) * index / (len(values) - 1)


def search_1d(array1d: np.ndarray, value: float, ascending: Optional[bool] = None, first: bool = True) -> int:
    if ascending is None:
        diff = np.abs(array1d - value)
        return diff.argmin()  # type: ignore
    elif ascending:
        side = 'left' if first else 'right'
        return array1d.searchsorted(value, side=side)  # type: ignore
    else:
        reversed_side = 'right' if first else 'left'
        return array1d.shape[0] - array1d[::-1].searchsorted(value, side=reversed_side)


def sliced_search_2d(array2d: np.ndarray, value: float, frac_x: Optional[float] = None, frac_y: Optional[float] = None,
                     ascending: Optional[bool] = None) -> Tuple[int, int]:
    if not ((frac_x is None) ^ (frac_y is None)):
        raise ValueError(f"Exactly one of frac_x ({frac_x}) or frac_y ({frac_y}) must be provided")

    if frac_y is not None:
        if not 0 <= frac_y <= 1:
            raise ValueError(f"Fraction must be between 0 and 1, got {frac_y}")
        i = int(frac_y * array2d.shape[0])
        j = search_1d(array2d[i, :], value, ascending, first=False)
    elif frac_x is not None:
        if not 0 <= frac_x <= 1:
            raise ValueError(f"Fraction must be between 0 and 1, got {frac_x}")
        j = int(frac_x * array2d.shape[1])
        i = search_1d(array2d[:, j], value, ascending, first=False)
    return i, j


def is_sorted_1d(array1d: np.ndarray, ascending: bool) -> bool:
    if ascending:
        return np.all(array1d[:-1] <= array1d[1:])  # type: ignore
    else:
        return np.all(array1d[:-1] >= array1d[1:])  # type: ignore
