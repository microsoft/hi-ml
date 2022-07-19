#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
from health_cpath.preprocessing.create_tiles_dataset import generate_tiles


def test_generate_slide_tiles() -> None:
    bg_value = 255
    fg_value = 128

    image_level_1 = np.array((
        (fg_value, fg_value, bg_value),
        (bg_value, fg_value, bg_value),
        (fg_value, fg_value, fg_value),
    ), np.uint8)

    # Add channel dimensions
    image_level_1 = np.array(3 * (image_level_1,))

    # Level 1 (foreground only, as loaded by the loader)
    # 1 1 0
    # 0 1 0
    # 1 1 1

    # Padded level 1 (gets padded automatically)
    # 1 1 0 0
    # 0 1 0 0
    # 1 1 1 0
    # 0 0 0 0

    tile_size = 2
    # Tiles
    # 1 1 | 0 0
    # 0 1 | 0 0
    # - - - - -
    # 1 1 | 1 0
    # 0 0 | 0 0

    foreground_threshold = 200
    occupancy_threshold = 0.49  # keeps the top left and bottom left
    image_tiles, tile_locations, occupancies, n_discarded = generate_tiles(image_level_1,
                                                                           tile_size,
                                                                           foreground_threshold,
                                                                           occupancy_threshold)
    assert np.all(image_tiles[0] == np.array([[[128, 128], [255, 128]],
                                              [[128, 128], [255, 128]],
                                              [[128, 128], [255, 128]]]))
    assert np.all(image_tiles[1] == np.array([[[128, 128], [255, 255]],
                                              [[128, 128], [255, 255]],
                                              [[128, 128], [255, 255]]]))
    assert np.all(tile_locations[0] == np.array([0, 0]))
    assert np.all(tile_locations[1] == np.array([0, 2]))
    assert occupancies[0] == 0.75
    assert occupancies[1] == 0.5
    assert n_discarded == 2
