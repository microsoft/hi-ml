#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""This script is specific to PANDA and is kept only for retrocompatibility.
`create_tiles_dataset.py` is the new supported way to process slide datasets.
"""
import functools
import sys
import shutil
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, Union, List

import numpy as np
from monai.data import Dataset
from tqdm import tqdm
from health_ml.utils.box_utils import Box

from health_cpath.preprocessing import tiling
from health_cpath.utils.naming import SlideKey, TileKey
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.preprocessing.loading import LoadMaskROId, WSIBackend
from health_cpath.preprocessing.create_tiles_dataset import get_tile_id, save_image, merge_dataset_csv_files

CSV_COLUMNS = (
    'slide_id',
    'tile_id',
    'image',
    'mask',
    'left',
    'top',
    'right',
    'bottom',
    'occupancy',
    'data_provider',
    'slide_isup_grade',
    'slide_gleason_score',
    'num_discarded',
)
TMP_SUFFIX = "_tmp"


def select_tile(mask_tile: np.ndarray, occupancy_threshold: float) \
        -> Union[Tuple[bool, float], Tuple[np.ndarray, np.ndarray]]:
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")
    # mask_tile has shape (N, C, H, W)
    foreground_mask = mask_tile > 0
    occupancy = foreground_mask.mean(axis=(-2, -1))
    selected = occupancy >= occupancy_threshold  # if the threshold is 0, all tiles should be selected
    # selected has shape (N, 1)
    return selected[:, 0], occupancy[:, 0]


def generate_tiles(sample: dict, tile_size: int, occupancy_threshold: float) \
        -> Tuple[np.ndarray, np.ndarray, List[Box], np.ndarray, int]:
    image_tiles, tile_locations = tiling.tile_array_2d(sample['image'], tile_size=tile_size,
                                                       constant_values=255)
    assert tile_locations.ndim == 2
    mask_tiles, _ = tiling.tile_array_2d(sample['mask'], tile_size=tile_size, constant_values=0)

    selected: np.ndarray
    occupancies: np.ndarray
    selected, occupancies = select_tile(mask_tiles, occupancy_threshold)  # type: ignore
    num_selected = selected.sum()
    num_tiles = len(image_tiles)
    num_discarded = num_tiles - num_selected
    percentage_discarded = 100 * num_discarded / num_tiles
    print(f"Discarded {num_discarded}/{num_tiles} tiles ({percentage_discarded:.2f} %)")

    image_tiles = image_tiles[selected]
    mask_tiles = mask_tiles[selected]
    tile_locations = tile_locations[selected]
    occupancies = occupancies[selected]

    top, left = sample['location']
    offset = np.array((left, top))

    abs_tile_locations = (sample['scale'] * tile_locations + offset).astype(int)
    tile_size_scaled = int(tile_size * sample['scale'])
    tile_boxes = [Box(x, y, tile_size_scaled, tile_size_scaled) for x, y in abs_tile_locations]

    return image_tiles, mask_tiles, tile_boxes, occupancies, num_discarded


# TODO refactor this to separate metadata identification from saving. We might want the metadata
# even if the saving fails
def save_tile(sample: dict, image_tile: np.ndarray, mask_tile: np.ndarray,
              tile_box: Box, output_dir: Path) -> dict:
    slide_id = sample[SlideKey.SLIDE_ID]
    tile_id = get_tile_id(slide_id, tile_box)
    image_tile_filename = f"train_images/{tile_id}.png"
    mask_tile_filename = f"train_label_masks/{tile_id}_mask.png"

    save_image(image_tile, output_dir / image_tile_filename)
    save_image(mask_tile, output_dir / mask_tile_filename)

    slide_metadata = sample[SlideKey.METADATA]
    tile_metadata = {
        TileKey.SLIDE_ID.value: slide_id,
        TileKey.TILE_ID.value: tile_id,
        TileKey.IMAGE.value: image_tile_filename,
        TileKey.MASK.value: mask_tile_filename,
        TileKey.TILE_LEFT.value: tile_box.x,
        TileKey.TILE_TOP.value: tile_box.y,
        TileKey.TILE_RIGHT.value: tile_box.x + tile_box.w,
        TileKey.TILE_BOTTOM.value: tile_box.y + tile_box.h,
        'data_provider': slide_metadata['data_provider'],
        'slide_isup_grade': slide_metadata['isup_grade'],
        'slide_gleason_score': slide_metadata['gleason_score'],
    }
    return tile_metadata


def process_slide(sample: dict, level: int, margin: int, tile_size: int, occupancy_threshold: int,
                  output_dir: Path, tile_progress: bool = False, filter_slide: str = '') -> None:
    slide_id = sample[SlideKey.SLIDE_ID]
    if filter_slide not in slide_id:
        return
    slide_dir: Path = output_dir / (slide_id + "/")
    print(f">>> Slide dir {slide_dir}")
    if slide_dir.exists():  # already processed slide - skip
        print(f">>> Skipping {slide_dir} - already processed")
        return
    else:
        mask_key = SlideKey.MASK  # it should be read from the dataset attribute instead, but we assume it's the same
        mask_path = Path(sample[mask_key])
        if not mask_path.is_file():
            print(f'Mask for slide {slide_id} not found')
            return

        slide_dir.mkdir(parents=True)

        dataset_csv_path = slide_dir / "dataset.csv"
        dataset_csv_file = dataset_csv_path.open('w')
        dataset_csv_file.write(','.join(CSV_COLUMNS) + '\n')  # write CSV header

        print(f"Loading slide {slide_id} ...")
        loader = LoadMaskROId(backend=WSIBackend.CUCIM, level=level, margin=margin)
        try:
            sample = loader(sample)  # load 'image' and 'mask' from disk
            failed = False
        except RuntimeError as e:  # happens when masks are empty
            print(f'Error loading slide {slide_id}, maybe due to an empty mask:\n{e}')
            failed = True

        if failed:
            print(f'Error loading slide {slide_id}')
            dataset_csv_file.close()
            shutil.rmtree(slide_dir)
            return

        print(f"Tiling slide {slide_id} ...")
        image_tiles, mask_tiles, tile_boxes, occupancies, num_discarded = \
            generate_tiles(sample, tile_size, occupancy_threshold)
        n_tiles = image_tiles.shape[0]

        for i in tqdm(range(n_tiles), f"Tiles ({slide_id[:6]}…)", unit="img", disable=not tile_progress):
            tile_metadata = save_tile(
                sample,
                image_tiles[i],
                mask_tiles[i],
                tile_boxes[i],
                slide_dir,
            )
            relative_slide_dir = Path(slide_dir.name)
            tile_metadata[TileKey.OCCUPANCY] = occupancies[i]
            tile_metadata[TileKey.IMAGE] = relative_slide_dir / tile_metadata[TileKey.IMAGE]
            tile_metadata[TileKey.MASK] = relative_slide_dir / tile_metadata[TileKey.MASK]
            tile_metadata[TileKey.NUM_DISCARDED] = num_discarded
            dataset_row = ','.join(str(tile_metadata[column]) for column in CSV_COLUMNS)
            dataset_csv_file.write(dataset_row + '\n')

        dataset_csv_file.close()


def main(panda_dir: Union[str, Path], root_output_dir: str, level: int, tile_size: int,
         margin: int, occupancy_threshold: float, parallel: bool = False, overwrite: bool = False,
         filter_slide: str = '') -> None:

    # Ignoring some types here because mypy is getting confused with the MONAI Dataset class
    # to select a subsample use keyword n_slides
    dataset = Dataset(PandaDataset(panda_dir))  # type: ignore
    output_dir = Path(root_output_dir)

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=not overwrite)

    print(f"Command: \"{' '.join(sys.argv)}\"")
    print(f"Creating dataset of level-{level} {tile_size}x{tile_size} PANDA tiles at: {output_dir}")

    func = functools.partial(process_slide, level=level, margin=margin, tile_size=tile_size,
                             occupancy_threshold=occupancy_threshold, output_dir=output_dir,
                             tile_progress=not parallel, filter_slide=filter_slide)

    if parallel:
        import multiprocessing

        pool = multiprocessing.Pool()
        map_func = pool.imap_unordered  # type: ignore
    else:
        map_func = map  # type: ignore

    list(tqdm(map_func(func, dataset), desc="Slides", unit="img", total=len(dataset)))  # type: ignore

    if parallel:
        pool.close()

    print("Merging slide files in a single file")
    merge_dataset_csv_files(output_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--panda-dir",
        type=str,
        default="/tmp/datasets/PANDA",
        help="Folder with the PANDA dataset. For example, '/tmp/datasets/PANDA'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/datasetdrive/PANDA_20X_level_0_224"
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min-occupancy",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--no-parallel",
        action="store_false",
        default=True,
        dest="parallel",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--filter-slide",
        type=str,
        default="",  # filtering for "b896" gives 4 slides, good for debugging
        help="Process only slides whose ID contain this substring. Useful for debugging"
    )
    args = parser.parse_args()

    main(
        panda_dir=args.panda_dir,
        root_output_dir=args.output_dir,
        level=args.level,
        tile_size=args.tile_size,
        margin=args.margin,
        occupancy_threshold=args.min_occupancy,
        parallel=args.parallel,
        overwrite=args.overwrite,
        filter_slide=args.filter_slide,
    )
