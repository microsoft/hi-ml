#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np

from health_ml.utils.box_utils import Box
from health_cpath.preprocessing.create_panda_tiles_dataset import generate_tiles


def test_generate_slide_tiles() -> None:
    image_size = 12
    bg_value = 255
    fg_value = 128
    image_level_0 = np.full((image_size, image_size), bg_value, np.uint8)
    image_level_0[4:6, 0:4] = fg_value
    image_level_0[6:8, 2:4] = fg_value
    image_level_0[8:10, 2:6] = fg_value
    image_level_1 = np.array((
        (fg_value, fg_value, bg_value),
        (bg_value, fg_value, bg_value),
        (bg_value, fg_value, fg_value),
    ), np.uint8)
    mask_level_1 = (image_level_1 == fg_value)
    # Add channel dimensions
    image_level_1 = np.array(3 * (image_level_1,))
    mask_level_1 = mask_level_1[np.newaxis]

    # Level 1 including background:
    # 0 0 0 0 0 0
    # 0 0 0 0 0 0
    # 1 1 0 0 0 0
    # 0 1 0 0 0 0
    # 0 1 1 0 0 0
    # 0 0 0 0 0 0

    location_level_0 = 4, 0
    # Level 1 (foreground only, as loaded by the loader)
    # 1 1 0
    # 0 1 0
    # 0 1 1

    tile_size = 2

    # Padded level 1
    # 1 1 0 0
    # 0 1 0 0
    # 0 1 1 0
    # 0 0 0 0

    # Tiles
    # 1 1 | 0 0
    # 0 1 | 0 0
    # - - - - -
    # 0 1 | 1 0
    # 0 0 | 0 0

    sample = {
        'image': image_level_1,
        'mask': mask_level_1,
        'location': location_level_0,
        'scale': 2,
    }

    min_occupancy: float = 0
    image_tiles, mask_tiles, tile_boxes, occupancies, num_discarded = generate_tiles(sample, tile_size, min_occupancy)
    assert len(image_tiles) == len(mask_tiles) == 4
    assert Box(x=0, y=4, w=4, h=4) in tile_boxes
    assert Box(x=4, y=4, w=4, h=4) in tile_boxes
    assert Box(x=0, y=8, w=4, h=4) in tile_boxes
    assert Box(x=4, y=8, w=4, h=4) in tile_boxes
    assert num_discarded == 0
    np.testing.assert_allclose(sorted(occupancies), (0, 0.25, 0.25, 0.75))

    min_occupancy = 0.1
    image_tiles, mask_tiles, tile_boxes, occupancies, num_discarded = generate_tiles(sample, tile_size, min_occupancy)
    assert len(image_tiles) == len(mask_tiles) == 3
    assert Box(x=0, y=4, w=4, h=4) in tile_boxes
    assert Box(x=0, y=8, w=4, h=4) in tile_boxes
    assert Box(x=4, y=8, w=4, h=4) in tile_boxes
    assert num_discarded == 1
    np.testing.assert_allclose(sorted(occupancies), (0.25, 0.25, 0.75))

    min_occupancy = 0.5
    image_tiles, mask_tiles, tile_boxes, occupancies, num_discarded = generate_tiles(sample, tile_size, min_occupancy)
    assert len(image_tiles) == len(mask_tiles) == 1
    assert Box(x=0, y=4, w=4, h=4) in tile_boxes
    assert num_discarded == 3
    np.testing.assert_allclose(sorted(occupancies), (0.75,))

    min_occupancy = 0.9
    image_tiles, mask_tiles, tile_boxes, occupancies, num_discarded = generate_tiles(sample, tile_size, min_occupancy)
    assert len(image_tiles) == len(mask_tiles) == 0
    assert num_discarded == 4
