#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import shutil
import logging
import tempfile
import functools
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Union, Optional

import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from monai.config import KeysCollection
from monai.data.image_reader import ImageReader, WSIReader
from monai.transforms import MapTransform

from health_ml.utils import box_utils
from health_ml.utils.type_annotations import TupleInt3

from health_cpath.utils.naming import SlideKey
from health_cpath.utils.viz_utils import add_text
from health_cpath.datasets.base_dataset import SlidesDataset

try:
    from cucim import CuImage
except ImportError:  # noqa: E722
    logging.warning("cucim library not available, code may fail")


class PandaDataset(SlidesDataset):
    """Dataset class for loading files from the PANDA challenge dataset.

    Iterating over this dataset returns a dictionary following the `SlideKey` schema plus meta-data
    from the original dataset (`'data_provider'`, `'isup_grade'`, and `'gleason_score'`).

    Ref.: https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview
    """
    SLIDE_ID_COLUMN = 'image_id'
    IMAGE_COLUMN = 'image'
    MASK_COLUMN = 'mask'

    METADATA_COLUMNS = ('data_provider', 'isup_grade', 'gleason_score')

    DEFAULT_CSV_FILENAME = "train.csv"

    def __init__(self,
                 root: Union[str, Path],
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None,
                 label_column: str = "isup_grade",
                 n_classes: int = 6) -> None:
        super().__init__(root, dataset_csv, dataset_df, validate_columns=False, label_column=label_column,
                         n_classes=n_classes)
        # PANDA CSV does not come with paths for image and mask files
        slide_ids = self.dataset_df.index
        self.dataset_df[self.IMAGE_COLUMN] = "train_images/" + slide_ids + ".tiff"
        self.dataset_df[self.MASK_COLUMN] = "train_label_masks/" + slide_ids + "_mask.tiff"
        self.validate_columns()

    def _make_thumbnail(
            self,
            sample: dict,
            reader: WSIReader,
            level: int,
            slide_size: int,
            images_dir: Path,
            masks_dir: Optional[Path] = None,
            image_suffix: str = '.png',
            default_mask_color: TupleInt3 = (119, 161, 120),
            ) -> None:  # noqa: E123
        slide_id = sample[SlideKey.SLIDE_ID]
        image_pil = self.load_slide_as_pil(reader, sample, SlideKey.IMAGE, level=level)
        image_pil = image_pil.resize(slide_size)
        add_text(image_pil, slide_id)
        if masks_dir is not None:
            try:
                mask_pil = self.load_slide_as_pil(reader, sample, SlideKey.MASK, level=level)
                mask_pil = mask_pil.resize(slide_size, Image.NEAREST)
                mask_pil = Image.fromarray(np.asarray(mask_pil) * 255)  # for visualization
            except ValueError:
                mask_pil = Image.new("RGB", slide_size, default_mask_color)
            finally:
                mask_path = masks_dir / f"{slide_id}.png"
                mask_pil.save(mask_path)
        image_path = images_dir / f"{slide_id}{image_suffix}"
        image_pil.save(image_path)

    def make_thumbnails(
            self,
            slide_size: int,
            images_dir: Path,
            level: int,
            masks_dir: Optional[Path] = None,
            parallel: bool = True,
            image_suffix: str = '.png',
            ) -> None:  # noqa: E123
        images_dir.mkdir(exist_ok=True, parents=True)
        if masks_dir is not None:
            masks_dir.mkdir(exist_ok=True)
        reader = WSIReader(backend="cucim")
        func = functools.partial(
            self._make_thumbnail,
            reader=reader,
            level=level,
            slide_size=slide_size,
            images_dir=images_dir,
            masks_dir=masks_dir,
            image_suffix=image_suffix,
        )
        if parallel:
            pool = multiprocessing.Pool()
            map_func = pool.imap_unordered  # type: ignore
        else:
            map_func = map  # type: ignore
        progress = tqdm(map_func(func, self), total=len(self))
        list(progress)  # type: ignore

        if parallel:
            pool.close()

    @staticmethod
    def make_montage_from_dir(
            images_dir: Path,
            num_cols: int,
            masks_dir: Optional[Path] = None,
            image_suffix: str = '.png',
            ) -> Image:  # noqa: E123
        image_paths = sorted(images_dir.glob(f'*{image_suffix}'))
        images_arrays = []
        for image_path in tqdm(image_paths):
            image_pil = Image.open(image_path)
            images_arrays.append(np.asarray(image_pil))
        images_array = np.asarray(images_arrays)
        if masks_dir is not None:
            mask_paths = sorted(masks_dir.glob('*.png'))
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

    def make_montage(
            self,
            out_path: Path,
            width: int = 60_000,
            level: int = 2,
            image_suffix: str = '.png',
            masks: bool = True,
            temp_dir: Optional[Path] = None,
            cleanup: bool = False,
            ) -> None:  # noqa: E123
        # There might be some blanks at the bottom right
        # rows * cols <= N
        # We are going to stack the slides and their masks slide by side, so we need 2 * cols
        # 2 * cols / rows = 16 / 9; rows = 2 * cols * 9 / 16
        # cols * 2 * cols * 9 / 16 <= N; cols <= (N / 2 * 16 / 9)**(1 / 2)
        num_slides = len(self)
        multiplier = 2 if masks else 1
        num_cols = int(np.sqrt(num_slides / multiplier * 16 / 9))
        slide_width = (width // num_cols) // multiplier
        slide_size = slide_width, slide_width
        temp_dir = tempfile.mkdtemp() if temp_dir is None else temp_dir
        temp_dir = Path(temp_dir)
        images_dir = temp_dir / "images"
        masks_dir = temp_dir / "masks" if masks else None
        if not temp_dir.is_dir():
            self.make_thumbnails(slide_size, images_dir, level, masks_dir=masks_dir, image_suffix=image_suffix)
        montage_pil = self.make_montage_from_dir(images_dir, num_cols, masks_dir=masks_dir, image_suffix=image_suffix)
        montage_pil.save(out_path)
        if cleanup:
            shutil.rmtree(temp_dir)


# MONAI's convention is that dictionary transforms have a 'd' suffix in the class name
class ReadImaged(MapTransform):
    """Basic transform to read image files."""

    def __init__(self, reader: ImageReader, keys: KeysCollection,
                 allow_missing_keys: bool = False, **kwargs: Any) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.reader = reader
        self.kwargs = kwargs

    def __call__(self, data: Dict) -> Dict:
        for key in self.keys:
            if key in data or not self.allow_missing_keys:
                data[key] = self.reader.read(data[key], **self.kwargs)
        return data


class LoadPandaROId(MapTransform):
    """Transform that loads a pathology slide and mask, cropped to the mask bounding box (ROI).

    Operates on dictionaries, replacing the file paths in `image_key` and `mask_key` with the
    respective loaded arrays, in (C, H, W) format. Also adds the following meta-data entries:
    - `'location'` (tuple): top-right coordinates of the bounding box
    - `'size'` (tuple): width and height of the bounding box
    - `'level'` (int): chosen magnification level
    - `'scale'` (float): corresponding scale, loaded from the file
    """

    def __init__(self, reader: WSIReader, image_key: str = 'image', mask_key: str = 'mask',
                 level: int = 0, margin: int = 0, **kwargs: Any) -> None:
        """
        :param reader: And instance of MONAI's `WSIReader`.
        :param image_key: Image key in the input and output dictionaries.
        :param mask_key: Mask key in the input and output dictionaries.
        :param level: Magnification level to load from the raw multi-scale files.
        :param margin: Amount in pixels by which to enlarge the estimated bounding box for cropping.
        """
        super().__init__([image_key, mask_key], allow_missing_keys=False)
        self.reader = reader
        self.image_key = image_key
        self.mask_key = mask_key
        self.level = level
        self.margin = margin
        self.kwargs = kwargs

    def _get_bounding_box(self, mask_obj: 'CuImage') -> box_utils.Box:
        # Estimate bounding box at the lowest resolution (i.e. highest level)
        highest_level = mask_obj.resolutions['level_count'] - 1
        scale = mask_obj.resolutions['level_downsamples'][highest_level]
        mask, _ = self.reader.get_data(mask_obj, level=highest_level)  # loaded as RGB PIL image

        foreground_mask = mask[0] > 0  # PANDA segmentation mask is in 'R' channel

        bbox = box_utils.get_bounding_box(foreground_mask)
        padded_bbox = bbox.add_margin(self.margin)
        scaled_bbox = scale * padded_bbox
        return scaled_bbox

    def __call__(self, data: Dict) -> Dict:
        mask_obj: CuImage = self.reader.read(data[self.mask_key])
        image_obj: CuImage = self.reader.read(data[self.image_key])

        level0_bbox = self._get_bounding_box(mask_obj)

        # cuCIM/OpenSlide take absolute location coordinates in the level 0 reference frame,
        # but relative region size in pixels at the chosen level
        scale = mask_obj.resolutions['level_downsamples'][self.level]
        scaled_bbox = level0_bbox / scale
        get_data_kwargs = dict(
            location=(level0_bbox.y, level0_bbox.x),
            size=(scaled_bbox.h, scaled_bbox.w),
            level=self.level,
        )
        mask, _ = self.reader.get_data(mask_obj, **get_data_kwargs)  # type: ignore
        data[self.mask_key] = mask[:1]  # PANDA segmentation mask is in 'R' channel
        data[self.image_key], _ = self.reader.get_data(image_obj, **get_data_kwargs)  # type: ignore
        data.update(get_data_kwargs)
        data['scale'] = scale

        mask_obj.close()
        image_obj.close()
        return data
