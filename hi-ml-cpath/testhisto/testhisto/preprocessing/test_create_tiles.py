#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
import numpy as np
from PIL import Image

from health_cpath.preprocessing.create_tiles_dataset import generate_tiles, get_tile_id, save_image
from health_ml.utils.box_utils import Box


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

    # Padded level 1 (gets padded automatically by generate_tiles)
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
    assert np.array_equal(image_tiles[0], np.array([[[128, 128], [255, 128]],
                                                    [[128, 128], [255, 128]],
                                                    [[128, 128], [255, 128]]]))
    assert np.array_equal(image_tiles[1], np.array([[[128, 128], [255, 255]],
                                                    [[128, 128], [255, 255]],
                                                    [[128, 128], [255, 255]]]))
    assert np.array_equal(tile_locations[0], np.array([0, 0]))
    assert np.array_equal(tile_locations[1], np.array([0, 2]))
    assert occupancies[0] == 0.75
    assert occupancies[1] == 0.5
    assert n_discarded == 2

    foreground_threshold = 200
    occupancy_threshold = 0.51  # keeps the top left
    image_tiles, tile_locations, occupancies, n_discarded = generate_tiles(image_level_1,
                                                                           tile_size,
                                                                           foreground_threshold,
                                                                           occupancy_threshold)
    assert np.array_equal(image_tiles[0], np.array([[[128, 128], [255, 128]],
                                                    [[128, 128], [255, 128]],
                                                    [[128, 128], [255, 128]]]))
    assert np.array_equal(tile_locations[0], np.array([0, 0]))
    assert occupancies[0] == 0.75
    assert n_discarded == 3

    foreground_threshold = 100  # discards everything
    occupancy_threshold = 0.49
    image_tiles, tile_locations, occupancies, n_discarded = generate_tiles(image_level_1,
                                                                           tile_size,
                                                                           foreground_threshold,
                                                                           occupancy_threshold)
    assert image_tiles.size == 0
    assert tile_locations.size == 0
    assert occupancies.size == 0
    assert n_discarded == 4


def test_get_tile_id() -> None:
    test_slide_id = 'f34esdsaf3'
    test_box = Box(1, 2, 3, 4)
    tile_id = get_tile_id(test_slide_id, test_box)

    assert tile_id == 'f34esdsaf3_left_00001_top_00002_right_00004_bottom_00006'


def test_save_image(tmp_path: Path) -> None:
    tmp_path = Path(str(tmp_path) + '.png')
    test_image = np.zeros((3, 5, 5))
    save_image(test_image, tmp_path)
    assert tmp_path.exists()

    test_image = np.array(Image.open(tmp_path))
    assert np.array_equal(test_image.shape, (5, 5, 3))
