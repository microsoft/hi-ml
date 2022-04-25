#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""This script is specific to PANDA and is kept only for retrocompatibility.
`create_tiles_dataset.py` is the new supported way to process slide datasets.
"""
import functools
import os
import sys
import logging
import shutil
import datetime
from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, Union, List

import PIL
import numpy as np
import coloredlogs
from monai.data import Dataset
from monai.data.image_reader import WSIReader
from tqdm import tqdm
from health_ml.utils.box_utils import Box

from histopathology.preprocessing import tiling
from histopathology.utils.naming import SlideKey, TileKey
from histopathology.datasets.panda_dataset import PandaDataset, LoadPandaROId

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

logging.basicConfig(format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def select_tile(mask_tile: np.ndarray, occupancy_threshold: float) \
        -> Union[Tuple[bool, float], Tuple[np.ndarray, np.ndarray]]:
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")
    # mask_tile has shape (N, C, H, W)
    foreground_mask = mask_tile > 0
    occupancy = foreground_mask.mean(axis=(-2, -1))
    selected = occupancy > occupancy_threshold
    # selected has shape (N, 1)
    return selected[:, 0], occupancy[:, 0]


def get_tile_descriptor(tile_box: Box) -> str:
    left, top = tile_box.x, tile_box.y
    right, bottom = left + tile_box.w, top + tile_box.h
    return f"left_{left:05d}_top_{top:05d}_right_{right:05d}_bottom_{bottom:05d}"


def get_tile_id(slide_id: str, tile_box: Box) -> str:
    return f"{slide_id}_{get_tile_descriptor(tile_box)}"


def save_image(array_chw: np.ndarray, path: Path) -> PIL.Image:
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image


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
    logging.info(f"Discarded {num_discarded}/{num_tiles} tiles ({percentage_discarded:.2f} %)")

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
    }
    pandas_metadata = {key: slide_metadata[key] for key in PandaDataset.METADATA_COLUMNS}
    tile_metadata.update(pandas_metadata)
    return tile_metadata


def process_slide(sample: dict, level: int, margin: int, tile_size: int, occupancy_threshold: int,
                  output_dir: Path, tile_progress: bool = False, filter_slide: str = '') -> None:
    slide_id = sample[SlideKey.SLIDE_ID]
    if filter_slide not in slide_id:
        return
    slide_dir: Path = output_dir / (slide_id + "/")
    logging.info(f">>> Slide dir {slide_dir}")
    if slide_dir.exists():  # already processed slide - skip
        logging.info(f">>> Skipping {slide_dir} - already processed")
        return
    else:
        slide_dir.mkdir(parents=True)

        dataset_csv_path = slide_dir / "dataset.csv"
        dataset_csv_file = dataset_csv_path.open('w')
        dataset_csv_file.write(','.join(CSV_COLUMNS) + '\n')  # write CSV header

        logging.info(f"Loading slide {slide_id} ...")
        reader = WSIReader(backend="cucim")
        loader = LoadPandaROId(reader, level=level, margin=margin)
        try:
            sample = loader(sample)  # load 'image' and 'mask' from disk
        except (ValueError, RuntimeError) as e:
            logging.error(f'Error loading slide {slide_id}:\n{e}')
            dataset_csv_file.close()
            shutil.rmtree(slide_dir)
            return
        logging.info(f"Tiling slide {slide_id} ...")
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
            tile_metadata['occupancy'] = occupancies[i]
            tile_metadata['image'] = os.path.join(slide_dir.name, tile_metadata['image'])
            tile_metadata['mask'] = os.path.join(slide_dir.name, tile_metadata['mask'])
            tile_metadata['num_discarded'] = num_discarded
            dataset_row = ','.join(str(tile_metadata[column]) for column in CSV_COLUMNS)
            dataset_csv_file.write(dataset_row + '\n')

        dataset_csv_file.close()


def merge_dataset_csv_files(dataset_dir: Path) -> Path:
    full_csv = dataset_dir / "dataset.csv"
    # TODO change how we retrieve these filenames, probably because mounted, the operation is slow
    #  and it seems to find many more files
    # print("List of files")
    # print([str(file) + '\n' for file in dataset_dir.glob("*/dataset.csv")])
    with full_csv.open('w') as full_csv_file:
        # full_csv_file.write(','.join(CSV_COLUMNS) + '\n')  # write CSV header
        first_file = True
        for slide_csv in tqdm(dataset_dir.glob("*/dataset.csv"), desc="Merging dataset.csv", unit='file'):
            logging.info(f"Merging slide {slide_csv}")
            content = slide_csv.read_text()
            if not first_file:
                content = content[content.index('\n') + 1:]  # discard header row for all but the first file
            full_csv_file.write(content)
            first_file = False
    return full_csv


def main(panda_dir: Union[str, Path], root_output_dir: Union[str, Path], level: int, tile_size: int,
         margin: int, occupancy_threshold: float, parallel: bool = False, overwrite: bool = False,
         filter_slide: str = '') -> None:

    # Ignoring some types here because mypy is getting confused with the MONAI Dataset class
    # to select a subsample use keyword n_slides
    dataset = Dataset(PandaDataset(panda_dir))  # type: ignore

    output_dir = Path(root_output_dir) / f"panda_tiles_level{level}_{tile_size}"
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=not overwrite)

    time_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile = open(output_dir / f"{time_string}.log", 'w')
    coloredlogs.install(level=logging.DEBUG, stream=logfile)
    logging.info(f"Command: \"{' '.join(sys.argv)}\"")
    logging.info(f"Creating dataset of level-{level} {tile_size}x{tile_size} PANDA tiles at: {output_dir}")

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

    logging.info("Merging slide files in a single file")
    merge_dataset_csv_files(output_dir)
    logfile.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--container-dir",
        type=str,
        required=True,
        help="Folder with the 'datasets' container",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--occupancy",
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
        default=False,
    )
    parser.add_argument(
        "--filter-slide",
        type=str,
        default="",
        help="Process only slides whose ID contain this substring. Useful for debugging"
    )
    args = parser.parse_args()

    panda_dir = Path(args.container_dir) / "panda"
    main(
        panda_dir=panda_dir,
        root_output_dir=args.output_dir,
        level=args.level,
        tile_size=args.size,
        margin=args.margin,
        occupancy_threshold=args.occupancy,
        parallel=args.parallel,
        overwrite=args.overwrite,
        filter_slide=args.filter_slide,
    )
