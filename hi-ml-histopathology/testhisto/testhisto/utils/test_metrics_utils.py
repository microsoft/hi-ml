#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import math
import random
from typing import List, Optional

import numpy as np
import pytest
import torch
from torch.functional import Tensor

from health_ml.utils.common_utils import is_gpu_available

from histopathology.utils.metrics_utils import select_k_tiles
from histopathology.utils.naming import ResultsKey
from histopathology.utils.heatmap_utils import location_selected_tiles


def set_random_seed(random_seed: int, caller_name: Optional[str] = None) -> None:
    """
    Set the seed for the random number generators of python, numpy, torch.random, and torch.cuda for all gpus.
    :param random_seed: random seed value to set.
    :param caller_name: name of the caller for logging purposes.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if is_gpu_available():
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(random_seed)  # type: ignore
    prefix = ""
    if caller_name is not None:
        prefix = caller_name + ": "
    logging.debug(f"{prefix}Random seed set to: {random_seed}")


def assert_equal_lists(pred: List, expected: List) -> None:
    assert len(pred) == len(expected)
    for i, slide in enumerate(pred):
        for j, value in enumerate(slide):
            if type(value) in [int, float]:
                assert math.isclose(value, expected[i][j], rel_tol=1e-06)
            elif isinstance(value, List):
                for k, idx in enumerate(value):
                    if type(idx) in [int, float]:
                        assert math.isclose(idx, expected[i][j][k], rel_tol=1e-06)
                    elif type(idx) == Tensor:
                        assert math.isclose(idx.item(), expected[i][j][k].item(), rel_tol=1e-06)
            else:
                raise TypeError("Unexpected list composition")


test_dict = {ResultsKey.SLIDE_ID: [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
             ResultsKey.IMAGE_PATH: [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
             ResultsKey.PROB: [Tensor([0.5]), Tensor([0.7]), Tensor([0.4]), Tensor([1.0])],
             ResultsKey.TRUE_LABEL: [0, 1, 1, 1],
             ResultsKey.BAG_ATTN:
                 [Tensor([[0.1, 0.0, 0.2, 0.15]]),
                  Tensor([[0.10, 0.18, 0.15, 0.13]]),
                  Tensor([[0.25, 0.23, 0.20, 0.21]]),
                  Tensor([[0.33, 0.31, 0.37, 0.35]])],
             ResultsKey.TILE_X:
                 [Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424])],
             ResultsKey.TILE_Y:
                 [Tensor([200, 424, 200, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424])]
             }


@pytest.mark.fast
def test_select_k_tiles() -> None:
    top_tn = select_k_tiles(test_dict, n_slides=1, label=0, n_tiles=2, select=('lowest_pred', 'highest_att'))
    assert_equal_lists(top_tn, [(1, 0.5, [3, 4], [Tensor([0.2]), Tensor([0.15])])])

    nslides = 2
    ntiles = 2
    top_fn = select_k_tiles(test_dict, n_slides=nslides, label=1, n_tiles=ntiles, select=('lowest_pred', 'highest_att'))
    bottom_fn = select_k_tiles(test_dict, n_slides=nslides, label=1, n_tiles=ntiles,
                               select=('lowest_pred', 'lowest_att'))
    assert_equal_lists(top_fn, [(3, 0.4, [1, 2], [Tensor([0.25]), Tensor([0.23])]),
                                (2, 0.7, [2, 3], [Tensor([0.18]), Tensor([0.15])])])
    assert_equal_lists(bottom_fn, [(3, 0.4, [3, 4], [Tensor([0.20]), Tensor([0.21])]),
                                   (2, 0.7, [1, 4], [Tensor([0.10]), Tensor([0.13])])])

    top_tp = select_k_tiles(test_dict, n_slides=nslides, label=1, n_tiles=ntiles,
                            select=('highest_pred', 'highest_att'))
    bottom_tp = select_k_tiles(test_dict, n_slides=nslides, label=1, n_tiles=ntiles,
                               select=('highest_pred', 'lowest_att'))
    assert_equal_lists(top_tp, [(4, 1.0, [3, 4], [Tensor([0.37]), Tensor([0.35])]),
                                (2, 0.7, [2, 3], [Tensor([0.18]), Tensor([0.15])])])
    assert_equal_lists(bottom_tp, [(4, 1.0, [2, 1], [Tensor([0.31]), Tensor([0.33])]),
                                   (2, 0.7, [1, 4], [Tensor([0.10]), Tensor([0.13])])])


@pytest.mark.fast
@pytest.mark.parametrize("level", [0, 1, 2])
def test_location_selected_tiles(level: int) -> None:
    set_random_seed(0)
    slide = 1
    location_bbox = [100, 100]
    slide_image = np.random.rand(3, 1000, 2000)

    coords = []
    slide_ids = [item[0] for item in test_dict[ResultsKey.SLIDE_ID]]  # type: ignore
    slide_idx = slide_ids.index(slide)
    for tile_idx in range(len(test_dict[ResultsKey.IMAGE_PATH][slide_idx])):  # type: ignore
        tile_coords = np.transpose(
            np.array([test_dict[ResultsKey.TILE_X][slide_idx][tile_idx].cpu().numpy(),  # type: ignore
                      test_dict[ResultsKey.TILE_Y][slide_idx][tile_idx].cpu().numpy()]))  # type: ignore
        coords.append(tile_coords)

    coords = np.array(coords)
    tile_coords_transformed = location_selected_tiles(tile_coords=coords,
                                                      location_bbox=location_bbox,
                                                      level=level)
    tile_xs, tile_ys = tile_coords_transformed.T
    level_dict = {0: 1, 1: 4, 2: 16}
    factor = level_dict[level]
    assert min(tile_xs) >= 0
    assert max(tile_xs) <= slide_image.shape[2] // factor
    assert min(tile_ys) >= 0
    assert max(tile_ys) <= slide_image.shape[1] // factor
