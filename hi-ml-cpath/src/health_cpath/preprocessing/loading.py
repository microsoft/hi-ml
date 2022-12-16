#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import param
import numpy as np
import skimage.filters

from enum import Enum
from health_ml.utils import box_utils
from health_cpath.utils.naming import SlideKey
from monai.data.wsi_reader import WSIReader
from monai.transforms import MapTransform, LoadImaged
from typing import Any, Callable, Dict, Optional, Tuple
from health_azure.logging import print_message_with_rank_pid


def get_luminance(slide: np.ndarray) -> np.ndarray:
    """Compute a grayscale version of the input slide.

    :param slide: The RGB image array in (*, C, H, W) format.
    :return: The single-channel luminance array as (*, H, W).
    """
    # TODO: Consider more sophisticated luminance calculation if necessary
    return slide.mean(axis=-3)  # type: ignore


def segment_foreground(slide: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """Segment the given slide by thresholding its luminance.

    :param slide: The RGB image array in (*, C, H, W) format.
    :param threshold: Pixels with luminance below this value will be considered foreground.
    If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.
    :return: A tuple containing the boolean output array in (*, H, W) format and the threshold used.
    """
    luminance = get_luminance(slide)
    if threshold is None:
        threshold = skimage.filters.threshold_otsu(luminance)
    return luminance < threshold, threshold


class ROIType(str, Enum):
    """Options for the ROI selection. Either a bounding box defined by foreground or a mask can be used."""
    FOREGROUND = 'foreground'
    MASK = 'segmentation_mask'
    WHOLE = 'whole_slide'


class WSIBackend(str, Enum):
    """Options for the WSI reader backend."""
    OPENSLIDE = 'OpenSlide'
    CUCIM = 'cuCIM'


class BaseLoadROId:
    """Abstract base class for loading a region of interest (ROI) from a slide. The ROI is defined by a bounding box."""

    def __init__(
        self, backend: str = WSIBackend.CUCIM, image_key: str = SlideKey.IMAGE, level: int = 1, margin: int = 0,
        backend_args: Dict = {}
    ) -> None:
        """
        :param backend: The WSI reader backend to use. One of 'OpenSlide' or 'cuCIM'. Default: 'cuCIM'.
        :param image_key: Image key in the input and output dictionaries. Default: 'image'.
        :param level: Magnification level to load from the raw multi-scale file, 0 is the highest resolution. Default: 1
            which loads the second highest resolution.
        :param margin: Amount in pixels by which to enlarge the estimated bounding box for cropping. Default: 0.
        :param backend_args: Additional arguments to pass to the WSI reader backend. Default: {}.
        """
        self.reader = WSIReader(backend=backend, **backend_args)
        self.image_key = image_key
        self.level = level
        self.margin = margin

    def _get_foreground_mask(self, slide_obj: Any, level: int) -> np.ndarray:
        """Estimate foreground mask at the given level of the slide."""
        raise NotImplementedError

    def _get_whole_slide_bbox(self, slide_obj: Any, level: int) -> box_utils.Box:
        """Return a bounding box that covers the whole slide at the given level."""
        h, w = self.reader.get_size(slide_obj, level=level)
        return box_utils.Box(0, 0, w, h)

    def _get_bounding_box(self, slide_obj: Any, slide_id: int) -> box_utils.Box:
        """Estimate bounding box at the lowest resolution (i.e. highest level) of the slide."""
        highest_level = self.reader.get_level_count(slide_obj) - 1
        scale = self.reader.get_downsample_ratio(slide_obj, highest_level)
        foreground_mask = self._get_foreground_mask(slide_obj, level=highest_level)
        try:
            bbox = box_utils.get_bounding_box(foreground_mask)
        except RuntimeError as e:
            print_message_with_rank_pid(f"Failed to estimate bounding box for slide {slide_id}: {e}")
            bbox = self._get_whole_slide_bbox(slide_obj, level=highest_level)
        return scale * bbox.add_margin(self.margin)

    def __call__(self, data: Dict) -> Dict:
        raise NotImplementedError


class LoadROId(MapTransform, BaseLoadROId):
    """Transform that loads a pathology slide, cropped to an estimated bounding box (ROI) of the foreground tissue.

    Operates on dictionaries, replacing the file path in `image_key` with the loaded array in
    (C, H, W) format. Also adds the following entries:
    - `SlideKey.ORIGIN` (tuple): top-right coordinates of the bounding box
    - `SlideKey.SCALE` (float): corresponding scale, loaded from the file
    - `SlideKey.FOREGROUND_THRESHOLD` (float): threshold used to segment the foreground
    """

    def __init__(
        self, image_key: str = SlideKey.IMAGE, foreground_threshold: Optional[float] = None, **kwargs: Any
    ) -> None:
        """
        :param image_key: Image key in the input and output dictionaries. Default: 'image'.
        :param foreground_threshold: Pixels with luminance below this value will be considered foreground.
        If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.
        :param kwargs: Additional arguments for `BaseLoadROId`.
        """
        MapTransform.__init__(self, [image_key], allow_missing_keys=False)
        BaseLoadROId.__init__(self, image_key=image_key, **kwargs)
        self.foreground_threshold = foreground_threshold

    def _get_foreground_mask(self, slide_obj: Any, level: int) -> np.ndarray:
        """Estimate foreground mask at the highest resolution (i.e. lowest level) of the slide based on luminance."""
        slide, _ = self.reader.get_data(slide_obj, level=level)
        foreground_mask, threshold = segment_foreground(slide, self.foreground_threshold)
        self.foreground_threshold = threshold
        return foreground_mask

    def __call__(self, data: Dict) -> Dict:
        try:
            image_obj = self.reader.read(data[self.image_key])
            level0_bbox = self._get_bounding_box(image_obj, data[SlideKey.SLIDE_ID])

            # cuCIM/OpenSlide takes absolute location coordinates in the level 0 reference frame,
            # but relative region size in pixels at the chosen level
            origin = (level0_bbox.y, level0_bbox.x)
            scale = self.reader.get_downsample_ratio(image_obj, self.level)
            scaled_bbox = level0_bbox / scale

            data[self.image_key], _ = self.reader.get_data(image_obj, location=origin, level=self.level,
                                                           size=(scaled_bbox.h, scaled_bbox.w))
            data[SlideKey.ORIGIN] = origin
            data[SlideKey.SCALE] = scale
            data[SlideKey.FOREGROUND_THRESHOLD] = self.foreground_threshold
        finally:
            image_obj.close()
        return data


class LoadMaskROId(MapTransform, BaseLoadROId):
    """Transform that loads a pathology slide and mask, cropped to the mask bounding box (ROI) defined by the mask.

    Operates on dictionaries, replacing the file paths in `image_key` and `mask_key` with the
    respective loaded arrays, in (C, H, W) format. Also adds the following meta-data entries:
    - `'location'` (tuple): top-right coordinates of the bounding box
    - `'size'` (tuple): width and height of the bounding box
    - `'level'` (int): chosen magnification level
    - `'scale'` (float): corresponding scale, loaded from the file
    """

    def __init__(self, image_key: str = SlideKey.IMAGE, mask_key: str = SlideKey.MASK, **kwargs: Any) -> None:
        """
        :param image_key: Image key in the input and output dictionaries. Default: 'image'.
        :param mask_key: Mask key in the input and output dictionaries. Default: 'mask'.
        :param kwargs: Additional arguments for `BaseLoadROId`.
        """
        MapTransform.__init__(self, [image_key, mask_key], allow_missing_keys=False)
        BaseLoadROId.__init__(self, image_key=image_key, **kwargs)
        self.mask_key = mask_key

    def _get_foreground_mask(self, mask_obj: Any, level: int) -> np.ndarray:
        """Load foreground mask at the given level of the slide."""
        mask, _ = self.reader.get_data(mask_obj, level=level)  # loaded as RGB PIL image
        foreground_mask = mask[0] > 0  # PANDA segmentation mask is in 'R' channel
        return foreground_mask

    def __call__(self, data: Dict) -> Dict:
        try:
            mask_obj = self.reader.read(data[self.mask_key])
            image_obj = self.reader.read(data[self.image_key])

            level0_bbox = self._get_bounding_box(mask_obj, data[SlideKey.SLIDE_ID])

            # cuCIM/OpenSlide take absolute location coordinates in the level 0 reference frame,
            # but relative region size in pixels at the chosen level
            scale = self.reader.get_downsample_ratio(mask_obj, self.level)
            scaled_bbox = level0_bbox / scale
            origin = (level0_bbox.y, level0_bbox.x)
            get_data_kwargs = dict(
                location=origin,
                size=(scaled_bbox.h, scaled_bbox.w),
                level=self.level,
            )
            mask, _ = self.reader.get_data(mask_obj, **get_data_kwargs)  # type: ignore
            data[self.mask_key] = mask[:1]  # PANDA segmentation mask is in 'R' channel
            data[self.image_key], _ = self.reader.get_data(image_obj, **get_data_kwargs)  # type: ignore
            data.update(get_data_kwargs)
            data[SlideKey.SCALE] = scale
            data[SlideKey.ORIGIN] = origin
        finally:
            mask_obj.close()
            image_obj.close()
        return data


class LoadingParams(param.Parameterized):
    """Parameters for loading a whole slide image."""

    level: int = param.Integer(
        default=1,
        doc="Magnification level to load from the raw multi-scale files. Default: 1.")
    margin: int = param.Integer(
        default=0, doc="Amount in pixels by which to enlarge the estimated bounding box for cropping")
    backend: WSIBackend = param.ClassSelector(
        default=WSIBackend.CUCIM,
        class_=WSIBackend,
        doc="WSI reader backend. Default: cuCIM.")
    roi_type: ROIType = param.ClassSelector(
        default=ROIType.WHOLE,
        class_=ROIType,
        doc="ROI type to use for cropping the slide. Default: `ROIType.WHOLE`. no cropping is performed.")
    image_key: str = param.String(
        default=SlideKey.IMAGE,
        doc="Key for the image in the data dictionary.")
    mask_key: str = param.String(
        default=SlideKey.MASK,
        doc="Key for the mask in the data dictionary. This only applies to `LoadMaskROId`.")
    foreground_threshold: Optional[float] = param.Number(
        default=None,
        bounds=(0, 255.),
        allow_None=True,
        doc="Threshold for foreground mask. If None, the threshold is selected automatically with otsu thresholding."
        "This only applies to `LoadROId`.")

    def set_roi_type_to_foreground(self) -> None:
        """Set the ROI type to foreground. This is useful for plotting even if we load whole slides during
        training. This help us reduce the size of thrumbnails to only meaningful tissue. We only hardcode it to
        foreground in the WHOLE case. Otherwise, keep it as is if a mask is available."""
        if self.roi_type == ROIType.WHOLE:
            self.roi_type = ROIType.FOREGROUND

    def get_load_roid_transform(self) -> Callable:
        """Returns a transform to load a slide and mask, cropped to the mask bounding box (ROI) defined by either the
        mask or the foreground."""
        if self.roi_type == ROIType.WHOLE:
            return LoadImaged(keys=self.image_key, reader=WSIReader, image_only=True, level=self.level,  # type: ignore
                              backend=self.backend, dtype=np.uint8, **self.get_additionl_backend_args())
        elif self.roi_type == ROIType.FOREGROUND:
            return LoadROId(backend=self.backend, image_key=self.image_key, level=self.level,
                            margin=self.margin, foreground_threshold=self.foreground_threshold,
                            backend_args=self.get_additionl_backend_args())
        elif self.roi_type == ROIType.MASK:
            return LoadMaskROId(backend=self.backend, image_key=self.image_key,
                                mask_key=self.mask_key, level=self.level, margin=self.margin,
                                backend_args=self.get_additionl_backend_args())
        else:
            raise ValueError(f"Unknown ROI type: {self.roi_type}. Choose from {list(ROIType)}.")

    def get_additionl_backend_args(self) -> Dict[str, Any]:
        """Returns a dictionary of additional arguments for the WSI reader backend. Multi processing is
        enabled since monai 1.0.0 by specifying num_workers > 0 with CuCIM backend only.
        This function can be overridden in BaseMIL to add additional arguments for the backend."""
        return dict()

    def should_upscale_coordinates(self) -> bool:
        """Returns True if the coordinates should be upscaled to the level 0 reference frame. This is the case when
        we load the whole slide without cropping for tiling on the fly. We need to upscale the coordinates to the
        highest level to match the coordinates of the tiles pipeline."""
        return self.level > 0 and self.roi_type == ROIType.WHOLE
