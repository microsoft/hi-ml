#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import List
import numpy as np


def location_selected_tiles(tile_coords: np.ndarray,
                            location_bbox: List[int],
                            scale_factor: int = 1,
                            should_upscale_coords: bool = True) -> np.ndarray:
    """ Return the scaled and shifted tile co-ordinates for selected tiles in the slide.

    :param tile_coords: XY tile coordinates, assumed to be spaced by multiples of `tile_size`
    (shape: [N, 2]) in original resolution.
    :param location_bbox: Location of the bounding box on the slide in original resolution.
    :param scale_factor: Scale factor to be applied to the tile coordinates.
    :param should_upscale_coords: If True, the tile coordinates are upscaled by the scale factor before substructing
        the offset. If False, the tile coordinates are not upscaled and the offset is upscaled by the scale factor.
    """

    y_tr, x_tr = location_bbox
    if should_upscale_coords:
        tile_coords = tile_coords * scale_factor
    tile_xs, tile_ys = tile_coords.T
    tile_xs = tile_xs - x_tr
    tile_ys = tile_ys - y_tr
    tile_xs = tile_xs // scale_factor
    tile_ys = tile_ys // scale_factor

    sel_coords = np.transpose([tile_xs.tolist(), tile_ys.tolist()])

    return sel_coords
