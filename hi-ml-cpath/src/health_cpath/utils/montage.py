#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import functools
import logging
import multiprocessing
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torchvision
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from monai.data.image_reader import WSIReader

from health_azure.argparsing import apply_overrides, parse_arguments
from health_cpath.preprocessing.loading import WSIBackend
from health_cpath.utils.montage_config import MontageConfig, create_montage_argparser
from health_cpath.utils.naming import SlideKey
from health_cpath.datasets.base_dataset import DEFAULT_DATASET_CSV, SlidesDataset
from health_ml.utils.type_annotations import TupleInt3


MONTAGE_FILE = "montage.png"
DatasetOrDataframe = Union[SlidesDataset, pd.DataFrame]
DatasetRecord = Dict[SlideKey, Any]

logger = logging.getLogger(__name__)


def add_text(image: Image, text: str, y: float = 0.9, color: TupleInt3 = (27, 77, 40), fontsize_step: int = 2) -> None:
    """Add text to a PIL image.

    :param image: Image object to which text needs to be added.
    :param text: The text that needs to be added.
    :param y: Float between 0-1 specifying the vertical position of the text (default=0.9).
    :param color: A 3-tuple indicating the fill color of the text (default = (27, 77, 40)).
    :param fontsize_step: Steps of font size to reduce if the text size is more than image size (default=2).
    """
    # This font is usually found in a path like /usr/share/fonts/truetype/dejavu
    font_path = Path('DejaVuSans.ttf')
    fontsize = 48
    draw = ImageDraw.Draw(image)
    image_size_x, image_size_y = image.size
    font = ImageFont.truetype(str(font_path), fontsize)
    text_size_x, text_size_y = draw.textsize(text, font=font)
    while text_size_x >= image_size_x:
        fontsize -= fontsize_step
        font = ImageFont.truetype(str(font_path), fontsize)
        text_size_x, text_size_y = draw.textsize(text, font=font)
    start_x = image_size_x // 2 - text_size_x // 2
    start_y = image_size_y * y - text_size_y // 2
    xy = start_x, start_y
    draw.text(xy, text, fill=color, font=font, align='center')


def load_slide_as_pil(reader: WSIReader, slide_file: Path, level: int = 0) -> Image:
    """Load a WSI as a PIL image.

    :param reader: The WSI reader for loading the slide.
    :param slide_file: The file to read from.
    :param level: Resolution downsampling level (default=0).
    :return: PIL image object corresponding to the WSI image.
    """
    image = reader.read(slide_file)
    try:
        image_array, _ = reader.get_data(image, level=level)
    except ValueError:
        logger.warning(f"Level {level} not available for {slide_file}, using level 0 instead.")
        image_array, _ = reader.get_data(image, level=0)
    array = image_array.numpy().transpose(1, 2, 0)
    to_pil = torchvision.transforms.ToPILImage()
    array_pil = to_pil(array)
    return array_pil


def _make_thumbnail(
    sample: DatasetRecord,
    reader: WSIReader,
    level: int,
    slide_size: Tuple[int, int],
    images_dir: Path,
    masks_dir: Optional[Path] = None,
    image_suffix: str = '.png',
    default_mask_color: TupleInt3 = (119, 161, 120),
) -> None:
    """Make thumbnails of the slides in slides dataset.

    :param sample: The slide dataset object dictionary for which thumbnail needs to be created.
    :param reader: The WSI reader for loading the slide.
    :param slide_size: The tuple of slide size (width, height).
    :param images_dir: The path to the `images` directory where WSI thumbnails will be stored.
    :param level: Resolution downsampling level.
    :param masks_dir: Optional path to `masks` directory where mask thumbnails will be stored.
        If `None` (default), masks thumbnails will not be created.
    :param image_suffix: Suffix of image thumbnails (default=`.png`).
    :param default_mask_color: Color of the masks (default = (119, 161, 120)).
    """
    try:
        image_pil = load_slide_as_pil(reader, sample[SlideKey.IMAGE], level)
        image_pil = image_pil.resize(slide_size)
        slide_id = sample.get(SlideKey.SLIDE_ID, "")
        # Slide IDs can be purely numeric, in those cases need to convert to str
        text_to_add = str(slide_id)
        if SlideKey.LABEL in sample:
            label = str(sample[SlideKey.LABEL])
            text_to_add += ": " + str(label)
        if text_to_add:
            add_text(image_pil, text_to_add)
        if masks_dir is not None and SlideKey.MASK in sample:
            masks_dir.mkdir(exist_ok=True)
            try:
                mask_pil = load_slide_as_pil(reader, sample[SlideKey.MASK], level=level)
                mask_pil = mask_pil.resize(slide_size, Image.NEAREST)
                mask_pil = Image.fromarray(np.asarray(mask_pil) * 255)  # for visualization
            except ValueError:
                mask_pil = Image.new("RGB", slide_size, default_mask_color)
            finally:
                mask_path = masks_dir / f"{slide_id}.png"
                mask_pil.save(mask_path)
        image_path = images_dir / f"{slide_id}{image_suffix}"
        image_pil.save(image_path)
    except Exception as ex:
        slide_id = sample.get(SlideKey.SLIDE_ID, "(no slide ID found)")
        logging.warning(f"Unable to process slide with ID '{slide_id}': {ex}")


def make_thumbnails(
    records: List[DatasetRecord],
    slide_size: Tuple[int, int],
    images_dir: Path,
    level: int,
    masks_dir: Optional[Path] = None,
    num_parallel: int = 0,
    image_suffix: str = '.png',
    backend: str = WSIBackend.CUCIM,
) -> None:
    """Make thumbnails of the slides in slides dataset.

    :param records: A list of dataframe records. The records must contain at least the columns `slide_id` and `image`.
    :param slide_size: The tuple of slide size (width, height).
    :param images_dir: The path to the `images` directory where WSI thumbnails will be stored.
    :param level: Resolution downsampling level.
    :param masks_dir: Optional path to `masks` directory where mask thumbnails will be stored (default=None).
    :param num_parallel: Number of parallel processes for thumbnail creation. Use 0 to disable parallel.
    :param image_suffix: Suffix of image thumbnails (default=`.png`).
    :param backend: The backend to use for reading the WSI (default=`cucim`).
    """
    images_dir.mkdir(exist_ok=True, parents=True)
    reader = WSIReader(backend=backend)
    func = functools.partial(
        _make_thumbnail,
        reader=reader,
        level=level,
        slide_size=slide_size,
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_suffix=image_suffix,
    )
    if num_parallel > 0:
        pool = multiprocessing.Pool(num_parallel)
        map_func = pool.imap_unordered  # type: ignore
    else:
        map_func = map  # type: ignore
    progress = tqdm(map_func(func, records), total=len(records))
    list(progress)  # type: ignore
    if num_parallel > 0:
        pool.close()


def make_montage_from_dir(
    images_dir: Path, num_cols: int, masks_dir: Optional[Path] = None, image_suffix: str = '.png'
) -> Image:
    """Create the montage image from the thumbnails.

    :param images_dir: The path to the `images` directory where WSI thumbnails will be stored.
    :param num_cols: Number of columns in the montage.
    :param masks_dir: Optional path to `masks` directory where mask thumbnails will be stored (default=None).
    :param image_suffix: Suffix of image thumbnails (default=`.png`).
    :return: PIL image of the montage.
    """
    image_paths = sorted(images_dir.glob(f'*{image_suffix}'))
    if len(image_paths) == 0:
        raise ValueError(f"No thumbnail images found in {images_dir}")
    images_arrays = []
    for image_path in tqdm(image_paths):
        image_pil = Image.open(image_path)
        images_arrays.append(np.asarray(image_pil))
    images_array = np.asarray(images_arrays)
    if masks_dir is not None:
        mask_paths = sorted(masks_dir.glob('*.png'))
        # Don't process masks if there are no files present, even if the mask directory has been passed as an argument.
        if len(mask_paths) > 0:
            if len(mask_paths) != len(image_paths):
                raise ValueError("Number of masks is different from number of images.")
            masks_arrays = []
            for mask_path in tqdm(mask_paths):
                mask_pil = Image.open(mask_path)
                masks_arrays.append(np.asarray(mask_pil))
            masks_array = np.asarray(masks_arrays)
            images_array = np.concatenate((images_array, masks_array), axis=-2)
    images_tensor = torch.from_numpy(images_array).permute(0, 3, 1, 2)
    grid_tensor = torchvision.utils.make_grid(images_tensor, nrow=num_cols)
    grid_pil = torchvision.transforms.ToPILImage()(grid_tensor)
    return grid_pil


def dataset_from_folder(root_folder: Path, glob_pattern: str = "**/*") -> pd.DataFrame:
    """Create slides dataset all files in a folder. The function searches for all files in the `root_folder` and its
    subfolders, and creates a dataframe with the following columns: `image`: The absolute path of the file,
    column `slide_id`: Either the file name only if that is unique, or otherwise the path of the file relative to
    the `root_folder`.

    :param root_folder: A directory with (image) files.
    :param glob_pattern: The glob pattern to match the image files (default=`**/*`, using all files recursively in all
        subfolders).
    :return: Slides dataset.
    """
    if not root_folder.is_dir():
        raise ValueError(f"Root folder '{root_folder}' does not exist or is not a directory.")
    # Find all image files in the folder, exclude folders in the result
    image_paths = list(sorted(f for f in root_folder.glob(glob_pattern) if f.is_file()))
    file_names_only = [path.name for path in image_paths]
    # Check if file names alone are enough to make the dataset unique.
    if len(file_names_only) != len(set(file_names_only)):
        # There are duplicates when going only by file names. Hence, use full paths relative to the root folder.
        image_ids = [str(path.relative_to(root_folder)) for path in image_paths]
    else:
        # File names are unique. Hence, use them as slide IDs.
        image_ids = file_names_only
    # Mounted datasets can show up with '&' appearing escape. Clean up the image IDs if so.
    # We expect that the exclusion list shows '&' for the offending slides.
    escaped_amp = "%26"
    if any(escaped_amp in image_id for image_id in image_ids):
        logging.info(f"Some image IDs contain '{escaped_amp}', replacing that with '&'.")
        image_ids = [id.replace(escaped_amp, "&") for id in image_ids]
    return pd.DataFrame({SlideKey.SLIDE_ID: image_ids, SlideKey.IMAGE: map(str, image_paths)})


def restrict_dataset(dataset: pd.DataFrame, column: str, items: List[str], include: bool) -> pd.DataFrame:
    """Exclude or include slides from a dataset, based on values in a column.
    For example, to exclude slides with label `0` from the dataset, use:
    restrict_dataset(dataset, column='label', items=['0'], include=False).

    The items are matched with the column values using `isin` operator. The code also handles
    the case where the column in question is the dataset index.
    If the items in question are not present in the column, the result is an empty dataset (if include=True)
    or the original dataset (if include=False).

    :param dataset: Slides dataset.
    :param column: The name of the column on which the inclusion/exclusion name.
    :param items: The values that the column should match.
    :param include: If True, modify the dataset to only include the rows where the column matches a value in the
        item list in `items`. If False, modify the dataset to exclude those rows.
    :return: Filtered dataset.
    """
    if column in dataset:
        matching_rows = dataset[column].isin(items)
        if not include:
            matching_rows = ~matching_rows
        return dataset[matching_rows]
    elif dataset.index.name == column:
        # Drop or loc operations on an index column when the values do not exist raise an error. Hence, restrict
        # to existing values first.
        items = list(set(items).intersection(set(dataset.index)))
        if include:
            return dataset.loc[items]
        else:
            return dataset.drop(items)
    else:
        raise ValueError(f"Column {column} not found in dataset.")


def dataset_to_records(dataset: DatasetOrDataframe) -> List[DatasetRecord]:
    """Converts a SlidesDataset or a plain dataframe to a list of dictionaries.

    :param dataset: Slides dataset or a plain dataframe.
    """
    if isinstance(dataset, pd.DataFrame):
        return dataset.to_dict(orient='records')
    elif isinstance(dataset, SlidesDataset):
        # SlidesData overrides __getitem__, use that to convert to records
        return [dataset[i] for i in range(len(dataset))]
    else:
        raise ValueError(f"Can't convert {type(dataset)} to a list of records.")


def make_montage(
    records: List[DatasetRecord],
    out_path: Path,
    width: int = 60_000,
    level: int = 2,
    image_suffix: str = '.png',
    masks: bool = True,
    temp_dir: Optional[Union[Path, str]] = None,
    cleanup: bool = False,
    num_parallel: int = 0,
    backend: str = "cucim",
) -> None:
    """Make the montage of WSI thumbnails from a slides dataset.

    :param records: A list of dataframe records. The records must contain at least the columns `slide_id` and `image`.
    :param out_path: The output path where the montage image will be stored.
    :param width: The width of the montage (default=60000).
    :param level: Resolution downsampling level at which the WSI will be read (default=2).
    :param image_suffix: Suffix of image thumbnails (default=`.png`).
    :param masks: Flag to denote if masks need to be included (default=True).
    :param temp_dir: Optional path to temporary directory that stores the slide thumbnails.
        If `None`(default), a temporary directory will be created in `tmp` folder.
    :param cleanup: Flag to determine whether to clean the temporary directory containing thumbnails
        after the montage is created (default=False).
    :param num_parallel: Number of parallel processes for thumbnail creation. Use 0 or 1 to disable parallel.
    :param backend: The backend to use for reading the WSI (default=`cucim`).
    """
    # There might be some blanks at the bottom right
    # rows * cols <= N
    # We are going to stack the slides and their masks slide by side, so we need 2 * cols
    # 2 * cols / rows = 16 / 9; rows = 2 * cols * 9 / 16
    # cols * 2 * cols * 9 / 16 <= N; cols <= (N / 2 * 16 / 9)**(1 / 2)
    num_slides = len(records)
    multiplier = 2 if masks else 1
    num_cols = int(np.sqrt(num_slides / multiplier * 16 / 9))
    logging.info(f"Creating montage from {num_slides} slides with {num_cols} columns.")
    slide_width = (width // num_cols) // multiplier
    slide_size = slide_width, slide_width // 2
    temp_dir = tempfile.mkdtemp() if temp_dir is None else temp_dir
    temp_dir = Path(temp_dir)
    image_thumbnail_dir = temp_dir / "images"
    mask_thumbnail_dir = temp_dir / "masks" if masks else None

    if image_thumbnail_dir.is_dir():
        logging.info(f"Skipping thumbnail creation because folder already exists: {image_thumbnail_dir}")
    else:
        logging.info(f"Starting thumbnail creation with thumbnail size {slide_size}")
        make_thumbnails(
            records=records,
            slide_size=slide_size,
            images_dir=image_thumbnail_dir,
            level=level,
            masks_dir=mask_thumbnail_dir,
            image_suffix=image_suffix,
            num_parallel=num_parallel,
            backend=backend,
        )
    try:
        logging.info("Starting montage creation")
        montage_pil = make_montage_from_dir(
            image_thumbnail_dir, num_cols, masks_dir=mask_thumbnail_dir, image_suffix=image_suffix
        )
    except Exception as ex:
        raise ValueError(f"Failed to create montage from {image_thumbnail_dir}: {ex}")
    logger.info(f"Saving montage to {out_path}")
    montage_pil.save(out_path)
    if out_path.suffix != '.jpg':
        jpeg_out = out_path.with_suffix('.jpg')
        montage_pil.save(jpeg_out, format='JPEG', quality=90)
    if cleanup:
        shutil.rmtree(temp_dir)


class MontageCreation(MontageConfig):
    def read_list(self, csv_file_path: Optional[Path]) -> List[str]:
        """Reads a list of slide IDs from a file."""
        if csv_file_path:
            df = pd.read_csv(csv_file_path)
            column_to_read = df.columns[0]
            if len(df.columns) > 1:
                logger.warning(f"More than one column in file, using first column: {column_to_read}")
            return df[column_to_read].tolist()
        else:
            return []

    def read_exclusion_list(self) -> List[str]:
        """Read the list of slide IDs that should be excluded from the montage."""
        if self.exclude_by_slide_id:
            slides_to_exclude = self.read_list(self.exclude_by_slide_id)
            logger.info(f"Excluding {len(slides_to_exclude)} slides from montage. First 3: {slides_to_exclude[:3]}")
            logger.info(
                "Exclusion list will be matched against the Slide ID column (for predefined datasets) or the "
                "filename."
            )
            return slides_to_exclude
        else:
            return []

    def read_inclusion_list(self) -> List[str]:
        """Read the list of slide IDs that should be included in the montage."""
        if self.include_by_slide_id:
            slides_to_include = self.read_list(self.include_by_slide_id)
            logger.info(f"Restricting montage to {len(slides_to_include)} slides. First 3: {slides_to_include[:3]}")
            logger.info(
                "Inclusion list will be matched against the Slide ID column (for predefined datasets) or the "
                "filename."
            )
            return slides_to_include
        else:
            return []

    def read_dataset(self, input_folder: Path) -> DatasetOrDataframe:
        """Read the dataset that should be used for creating the montage. If a glob pattern has been provided, then
        all the image files specified by that pattern will be used for the montage. Otherwise, a file `dataset.csv`
        is expected in the input folder. The `dataset.csv` will be used to create an instance of `SlidesDataset`.

        :param input_folder: The folder where the dataset is located.
        :return: A SlidesDataset or dataframe object that contains the dataset."""
        if self.image_glob_pattern:
            logger.info(f"Trying to create a dataset from files that match: {self.image_glob_pattern}")
            try:
                dataset = dataset_from_folder(input_folder, glob_pattern=self.image_glob_pattern)
            except Exception as ex:
                raise ValueError(f"Unable to create dataset from files in folder {input_folder}: {ex}")
            if len(dataset) == 0:
                raise ValueError(f"No images found in folder {input_folder} with pattern {self.image_glob_pattern}")
            return dataset
        else:
            logger.info(f"Trying to load the dataset as a SlidesDataset from folder {input_folder}")
            try:
                dataset = SlidesDataset(root=input_folder)
            except Exception as ex:
                logging.error("Unable to load dataset.")
                file = input_folder / DEFAULT_DATASET_CSV
                # Print the whole directory tree to check where the problem is.
                while str(file) != str(file.root):
                    logging.debug(f"File: {file}, exists: {file.exists()}")
                    file = file.parent
                raise ValueError(
                    f"Unable to load dataset. Check if the file {DEFAULT_DATASET_CSV} "
                    f"exists, or provide a file name pattern via --image_glob_pattern. Error: {ex}"
                )
            return dataset

    def create_montage(self, input_folder: Path) -> None:
        """Creates a montage from the dataset in the input folder. The method reads the dataset, creates an output
        folder, handles the inclusion and exclusion lists, and then calls the method that creates the montage.

        :param input_folder: The folder where the dataset is located.
        :raises ValueError: If both an inclusion and exclusion list have been provided.
        """
        dataset = self.read_dataset(input_folder)
        self.output_path.mkdir(parents=True, exist_ok=True)
        if self.include_by_slide_id and self.exclude_by_slide_id:
            raise ValueError("You cannot provide both an inclusion and exclusion list.")
        if self.include_by_slide_id:
            items = self.read_inclusion_list()
            exclude_items = False
        elif self.exclude_by_slide_id:
            items = self.read_exclusion_list()
            exclude_items = True
        else:
            items = []
            exclude_items = True
        self.montage_from_included_and_excluded_slides(
            dataset=dataset,
            items=items,
            exclude_items=exclude_items,
        )

    def montage_from_included_and_excluded_slides(
        self,
        dataset: DatasetOrDataframe,
        items: Optional[List[str]] = None,
        exclude_items: bool = True,
        restrict_by_column: str = "",
    ) -> Optional[Path]:
        """Creates a montage of included and excluded slides from the dataset.

        :param dataset: Slides dataset or a plain dataframe.
        :param items: A list values for SlideID that should be included/excluded from the montage.
        :param exclude_items: If True, exclude the list in `items` from the montage. If False, include
            only those in the montage.
        :param restrict_by_column: The column name that should be used for inclusion/exclusion lists
            (default=dataset.slide_id_column).
        :return: A path to the created montage, or None if no images were available for creating the montage.
        """
        if isinstance(dataset, pd.DataFrame):
            df_original = dataset
        else:
            df_original = dataset.dataset_df
        logging.info(f"Input dataset contains {len(df_original)} records.")

        if restrict_by_column == "":
            if isinstance(dataset, pd.DataFrame):
                restrict_by_column = SlideKey.SLIDE_ID.value
            else:
                restrict_by_column = dataset.slide_id_column
        if items:
            if exclude_items:
                logging.info(f"Using dataset column '{restrict_by_column}' to exclude slides")
                include = False
            else:
                logging.info(f"Using dataset column '{restrict_by_column}' to restrict the set of slides")
                include = True
            df_restricted = restrict_dataset(df_original, column=restrict_by_column, items=items, include=include)
            logging.info(f"Updated dataset contains {len(df_restricted)} records")
        else:
            df_restricted = df_original

        montage_result = self.output_path / MONTAGE_FILE
        logging.info(f"Creating montage in {montage_result}")
        if isinstance(dataset, pd.DataFrame):
            records_restricted = dataset_to_records(df_restricted)
        else:
            dataset.dataset_df = df_restricted
            records_restricted = dataset_to_records(dataset)
            # We had to modify the dataset in place, hence restore the original dataset to avoid odd side-effects.
            dataset.dataset_df = df_original

        if len(records_restricted) > 0:
            make_montage(
                records=records_restricted,
                out_path=montage_result,
                width=self.width,
                level=self.level,
                masks=False,
                cleanup=True,
                num_parallel=self.parallel,
                backend=self.backend,
            )
            return montage_result
        else:
            logging.info("No slides to include in montage, skipping.")
            return None


def create_config_from_args() -> MontageConfig:
    """Creates a configuration object for montage creation from the commandline arguments.

    :return: An object that describes all options for the montage creation.
    """
    parser = create_montage_argparser()
    config = MontageCreation()
    parser_results = parse_arguments(parser, args=sys.argv[1:], fail_on_unknown_args=True)
    _ = apply_overrides(config, parser_results.args)
    return config
