#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import functools
import shutil
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import PIL
from monai.data import Dataset
from tqdm import tqdm
from health_ml.utils.box_utils import Box

from health_cpath.datasets.base_dataset import SlidesDataset
from health_cpath.preprocessing import tiling
from health_cpath.preprocessing.loading import LoadROId, WSIBackend, segment_foreground
from health_cpath.utils.naming import SlideKey, TileKey


def select_tiles(foreground_mask: np.ndarray, occupancy_threshold: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Exclude tiles that are mostly background based on estimated occupancy.

    :param foreground_mask: Boolean array of shape (*, H, W).
    :param occupancy_threshold: Tiles with lower occupancy (between 0 and 1) will be discarded.
    :return: A tuple containing which tiles were selected and the estimated occupancies. These will
    be boolean and float arrays of shape (*,), or scalars if `foreground_mask` is a single tile.
    """
    if occupancy_threshold < 0. or occupancy_threshold > 1.:
        raise ValueError("Tile occupancy threshold must be between 0 and 1")
    occupancy = foreground_mask.mean(axis=(-2, -1))
    return (occupancy > occupancy_threshold).squeeze(), occupancy.squeeze()  # type: ignore


def get_tile_descriptor(tile_box: Box) -> str:
    """Format the box tile coordinates into a tile descriptor."""
    left, top = tile_box.x, tile_box.y
    right, bottom = left + tile_box.w, top + tile_box.h
    return f"left_{left:05d}_top_{top:05d}_right_{right:05d}_bottom_{bottom:05d}"


def get_tile_id(slide_id: str, tile_box: Box) -> str:
    """Format the slide ID and box tile coordinates into a unique tile ID."""
    return f"{slide_id}_{get_tile_descriptor(tile_box)}"


def save_image(array_chw: np.ndarray, path: Path) -> PIL.Image:
    """Save an image array in (C, H, W) format to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array_hwc = np.moveaxis(array_chw, 0, -1).astype(np.uint8).squeeze()
    pil_image = PIL.Image.fromarray(array_hwc)
    pil_image.convert('RGB').save(path)
    return pil_image


def generate_tiles(slide_image: np.ndarray, tile_size: int, foreground_threshold: float,
                   occupancy_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Split the foreground of an input slide image into tiles.

    :param slide_image: The RGB image array in (C, H, W) format.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :return: A tuple containing the image tiles (N, C, H, W), tile coordinates (N, 2), occupancies
    (N,), and total number of discarded empty tiles.
    """
    image_tiles, tile_locations = tiling.tile_array_2d(slide_image, tile_size=tile_size,
                                                       constant_values=255)
    foreground_mask, _ = segment_foreground(image_tiles, foreground_threshold)

    selected, occupancies = select_tiles(foreground_mask, occupancy_threshold)
    n_discarded = (~selected).sum()
    print(f"Percentage tiles discarded: {n_discarded / len(selected) * 100:.2f}")

    image_tiles = image_tiles[selected]
    tile_locations = tile_locations[selected]
    occupancies = occupancies[selected]

    return image_tiles, tile_locations, occupancies, n_discarded


def get_tile_info(sample: Dict[SlideKey, Any], occupancy: float, tile_box: Box,
                  rel_slide_dir: Path) -> dict:
    """Map slide information and tiling outputs into tile-specific information dictionary.

    :param sample: Slide dictionary.
    :param occupancy: Estimated tile foreground occuppancy.
    :param tile_box: Tile box coordinates.
    :param rel_slide_dir: Directory where tiles are saved, relative to dataset root.
    :return: Tile information dictionary.
    """
    slide_id = sample[SlideKey.SLIDE_ID]
    descriptor = get_tile_descriptor(tile_box)
    rel_image_path = f"{rel_slide_dir}/{descriptor}.png"

    tile_info = {
        TileKey.SLIDE_ID: slide_id,
        TileKey.TILE_ID: get_tile_id(slide_id, tile_box),
        TileKey.IMAGE: rel_image_path,
        TileKey.LABEL: sample[SlideKey.LABEL],
        TileKey.TILE_LEFT.value: tile_box.x,
        TileKey.TILE_TOP.value: tile_box.y,
        TileKey.TILE_RIGHT.value: tile_box.x + tile_box.w,
        TileKey.TILE_BOTTOM.value: tile_box.y + tile_box.h,
        TileKey.OCCUPANCY: occupancy,
        TileKey.SLIDE_METADATA: {key: value
                                 for key, value in sample[SlideKey.METADATA].items()}
    }
    return tile_info


def format_csv_row(tile_info: Dict[TileKey, Any], keys_to_save: Iterable[TileKey],
                   metadata_keys: Iterable[str]) -> str:
    """Format tile information dictionary as a row to write to a dataset CSV tile.

    :param tile_info: Tile information dictionary.
    :param keys_to_save: Which main keys to include in the row, and in which order.
    :param metadata_keys: Likewise for metadata keys.
    :return: The formatted CSV row.
    """
    tile_slide_metadata = tile_info.pop(TileKey.SLIDE_METADATA)

    fields = [str(tile_info[key]) for key in keys_to_save]
    fields.extend(str(tile_slide_metadata[key]) for key in metadata_keys)
    fields = ['"' + value + '"' if ',' in value else value for value in fields]  # if field contains a , add extra " "

    dataset_row = ','.join(fields)
    return dataset_row


def process_slide(sample: Dict[SlideKey, Any], level: int, margin: int, tile_size: int,
                  foreground_threshold: Optional[float], occupancy_threshold: float, output_dir: Path,
                  tile_progress: bool = False) -> None:
    """Load and process a slide, saving tile images and information to a CSV file.

    :param sample: Slide information dictionary, returned by the input slide dataset.
    :param level: Magnification level at which to process the slide.
    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    If `None` (default), an optimal threshold will be estimated automatically.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :param output_dir: Root directory for the output dataset; outputs for a single slide will be
    saved inside `output_dir/slide_id/`.
    :param tile_progress: Whether to display a progress bar in the terminal.
    """
    slide_metadata: Dict[str, Any] = sample[SlideKey.METADATA]
    keys_to_save = (TileKey.SLIDE_ID, TileKey.TILE_ID, TileKey.IMAGE, TileKey.LABEL,
                    TileKey.TILE_LEFT, TileKey.TILE_TOP, TileKey.TILE_RIGHT, TileKey.TILE_BOTTOM, TileKey.OCCUPANCY)
    metadata_keys = tuple(key for key in slide_metadata)
    csv_columns: Tuple[str, ...] = (*keys_to_save, *metadata_keys)

    slide_id: str = sample[SlideKey.SLIDE_ID]
    rel_slide_dir = Path(slide_id)
    slide_dir = output_dir / rel_slide_dir
    print(f">>> Slide dir {slide_dir}")
    if slide_dir.exists():  # already processed slide - skip
        print(f">>> Skipping {slide_dir} - already processed")
        return
    else:
        try:
            slide_dir.mkdir(parents=True)

            dataset_csv_path = slide_dir / "dataset.csv"
            dataset_csv_file = dataset_csv_path.open('w')
            dataset_csv_file.write(','.join(csv_columns) + '\n')  # write CSV header

            n_failed_tiles = 0
            failed_tiles_csv_path = slide_dir / "failed_tiles.csv"
            failed_tiles_file = failed_tiles_csv_path.open('w')
            failed_tiles_file.write('tile_id' + '\n')

            print(f"Loading slide {slide_id} ...")
            loader = LoadROId(backend=WSIBackend.CUCIM, level=level, margin=margin,
                              foreground_threshold=foreground_threshold)
            sample = loader(sample)  # load 'image' from disk

            print(f"Tiling slide {slide_id} ...")
            image_tiles, rel_tile_locations, occupancies, _ = \
                generate_tiles(sample[SlideKey.IMAGE], tile_size,
                               sample[SlideKey.FOREGROUND_THRESHOLD],
                               occupancy_threshold)

            tile_locations = (sample[SlideKey.SCALE] * rel_tile_locations
                              + sample[SlideKey.ORIGIN]).astype(int)  # noqa: W503
            tile_size_scaled = int(tile_size * sample[SlideKey.SCALE])
            tile_boxes = [Box(x, y, tile_size_scaled, tile_size_scaled) for x, y in tile_locations]

            n_tiles = image_tiles.shape[0]

            print(f"Saving tiles for slide {slide_id} ...")
            for i in tqdm(range(n_tiles), f"Tiles ({slide_id[:6]}…)", unit="img", disable=not tile_progress):
                try:
                    tile_info = get_tile_info(sample, occupancies[i], tile_boxes[i], rel_slide_dir)
                    save_image(image_tiles[i], output_dir / tile_info[TileKey.IMAGE])
                    dataset_row = format_csv_row(tile_info, keys_to_save, metadata_keys)
                    dataset_csv_file.write(dataset_row + '\n')
                except Exception as e:
                    n_failed_tiles += 1
                    descriptor = get_tile_descriptor(tile_boxes[i])
                    failed_tiles_file.write(descriptor + '\n')
                    traceback.print_exc()
                    warnings.warn(f"An error occurred while saving tile "
                                  f"{get_tile_id(slide_id, tile_boxes[i])}: {e}")

            dataset_csv_file.close()
            failed_tiles_file.close()
            if n_failed_tiles > 0:
                # TODO what we want to do with slides that have some failed tiles?
                print(f"{slide_id} is incomplete. {n_failed_tiles} tiles failed.")
            print(f"Finished processing slide {slide_id}")
        except Exception as e:
            traceback.print_exc()
            warnings.warn(f"An error occurred while processing slide {slide_id}: {e}")


def merge_dataset_csv_files(dataset_dir: Path) -> Path:
    """Combines all "*/dataset.csv" files into a single "dataset.csv" file in the given directory."""
    full_csv = dataset_dir / "dataset.csv"
    # TODO change how we retrieve these filenames, probably because mounted, the operation is slow
    #  and it seems to find many more files
    # print("List of files")
    # print([str(file) + '\n' for file in dataset_dir.glob("*/dataset.csv")])
    with full_csv.open('w') as full_csv_file:
        # full_csv_file.write(','.join(CSV_COLUMNS) + '\n')  # write CSV header
        first_file = True
        for slide_csv in tqdm(dataset_dir.glob("*/dataset.csv"), desc="Merging dataset.csv", unit='file'):
            print(f"Merging slide {slide_csv}")
            content = slide_csv.read_text()
            if not first_file:
                content = content[content.index('\n') + 1:]  # discard header row for all but the first file
            full_csv_file.write(content)
            first_file = False
    return full_csv


def main(slides_dataset: SlidesDataset, root_output_dir: Union[str, Path],
         level: int, tile_size: int, margin: int, foreground_threshold: Optional[float],
         occupancy_threshold: float, parallel: bool = False, overwrite: bool = False,
         n_slides: Optional[int] = 0, only: Optional[str] = None) -> None:
    """Process a slides dataset to produce a tiles dataset.

    :param slides_dataset: Input tiles dataset object.
    :param root_output_dir: The root directory of the output tiles dataset.
    :param level: Magnification level at which to process the slide.
    :param tile_size: Lateral dimensions of each tile, in pixels.
    :param margin: Margin around the foreground bounding box, in pixels at lowest resolution.
    :param foreground_threshold: Luminance threshold (0 to 255) to determine tile occupancy.
    If `None` (default), an optimal threshold will be estimated automatically.
    :param occupancy_threshold: Threshold (between 0 and 1) to determine empty tiles to discard.
    :param parallel: Whether slides should be processed in parallel with multiprocessing.
    :param overwrite: Whether to overwrite an existing output tiles dataset. If `True`, will delete
    and recreate `root_output_dir`, otherwise will resume by skipping already processed slides.
    :param n_slides: If given, limit the total number of slides for debugging.
    :param only: Id of a single slide for debugging.
    """

    # Ignoring some types here because mypy is getting confused with the MONAI Dataset class
    # to select a subsample use keyword n_slides
    if only:
        slides_dataset.dataset_df = slides_dataset.dataset_df.filter(items=[only], axis=0)
    dataset = Dataset(slides_dataset)  # type: ignore

    if n_slides > 0:  # type: ignore
        dataset = dataset[:n_slides]

    output_dir = Path(root_output_dir)

    print(f"Creating dataset of level-{level} {tile_size}x{tile_size} "
          f"{slides_dataset.__class__.__name__} tiles at: {output_dir}")

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=not overwrite)

    func = functools.partial(process_slide, level=level, margin=margin, tile_size=tile_size,
                             foreground_threshold=foreground_threshold,
                             occupancy_threshold=occupancy_threshold, output_dir=output_dir,
                             tile_progress=not parallel)

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
    from health_cpath.datasets.tcga_prad_dataset import TcgaPradDataset

    # Example set up for an existing slides dataset:
    main(slides_dataset=TcgaPradDataset("/tmp/datasets/TCGA-PRAD_20220712"),
         root_output_dir="/tmp/datasets/TCGA-PRAD_10X_tiles_level1_224",
         n_slides=2,
         only=None,
         level=0,
         tile_size=224,
         margin=0,
         foreground_threshold=None,
         occupancy_threshold=0.05,
         parallel=True,
         overwrite=True)
