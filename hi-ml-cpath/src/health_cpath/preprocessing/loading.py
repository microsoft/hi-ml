#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
import logging
import param
import numpy as np
import skimage.filters

from health_ml.utils import box_utils
from health_cpath.utils.naming import SlideKey
from monai.data.wsi_reader import WSIReader
from monai.transforms import MapTransform
from openslide import OpenSlide
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

try:
    from cucim import CuImage
except ImportError:  # noqa: E722
    logging.warning("cucim library not available, code may fail")


_OpenSlideOrCuImage = TypeVar('_OpenSlideOrCuImage', 'CuImage', OpenSlide)


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


class WSIBackend(str, Enum):
    """Options for the WSI reader backend."""
    OPENSLIDE = 'OpenSlide'
    CUCIM = 'cuCIM'


class CuImageMixin:
    """Mixin class for CuImage support in WSIReader."""

    def _get_size_at_level(self, slide_obj: 'CuImage', level: int) -> Tuple[int, int]:
        size = slide_obj.resolutions['level_dimensions'][level]
        return size[::-1]  # CuImage returns (width, height), we need to reverse it

    def _get_highest_level(self, slide_obj: 'CuImage') -> int:
        return slide_obj.resolutions['level_count'] - 1

    def _get_scale_at_level(self, slide_obj: 'CuImage', level: int) -> float:
        return slide_obj.resolutions['level_downsamples'][level]


class OpenSlideMixin:
    """Mixin class for OpenSlide support in WSIReader."""

    def _get_size_at_level(self, slide_obj: OpenSlide, level: int) -> Tuple[int, int]:
        size = slide_obj.level_dimensions[level]
        return size[::-1]  # OpenSlide returns (width, height), we need to reverse it

    def _get_highest_level(self, slide_obj: OpenSlide) -> int:
        return slide_obj.level_count - 1

    def _get_scale_at_level(self, slide_obj: OpenSlide, level: int) -> float:
        return slide_obj.level_downsamples[level]


class BaseLoadROId:
    """Abstract base class for loading a region of interest (ROI) from a slide. The ROI is defined by a bounding box."""
    def __init__(
        self, backend: str, image_key: str = SlideKey.IMAGE, level: int = 2, margin: int = 0, backend_args: Dict = {}
    ) -> None:
        """
        :param backend: The WSI reader backend to use. One of 'OpenSlide' or 'cuCIM'.
        :param image_key: Image key in the input and output dictionaries.
        :param level: Magnification level to load from the raw multi-scale file.
        :param margin: Amount in pixels by which to enlarge the estimated bounding box for cropping.
        :param backend_args: Additional arguments to pass to the WSI reader backend.
        """
        self.reader = WSIReader(backend=backend, **backend_args)
        self.image_key = image_key
        self.level = level
        self.margin = margin

    def _get_size_at_level(self, slide_obj: _OpenSlideOrCuImage, level: int) -> Tuple[int, int]:
        """Get the size of the slide at the given level. This is specific to the slide type (OpenSlide or CuImage).
        It will be implemented by the mixin classes."""
        raise NotImplementedError

    def _get_highest_level(self, slide_obj: _OpenSlideOrCuImage) -> int:
        """Get the highest magnification level of the slide. This is specific to the slide type (OpenSlide or CuImage).
        It will be implemented by the mixin classes."""
        raise NotImplementedError

    def _get_scale_at_level(self, slide_obj: _OpenSlideOrCuImage, level: int) -> float:
        """Get the scale of the slide at the given level. This is specific to the slide type (OpenSlide or CuImage).
        It will be implemented by the mixin classes."""
        raise NotImplementedError

    def _get_bounding_box(self, slide_obj: _OpenSlideOrCuImage) -> box_utils.Box:
        raise NotImplementedError

    def __call__(self, data: Dict) -> Dict:
        raise NotImplementedError


class LoadROId(MapTransform, BaseLoadROId, Generic[_OpenSlideOrCuImage]):
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
        :param image_key: Image key in the input and output dictionaries.
        :param foreground_threshold: Pixels with luminance below this value will be considered foreground.
        If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.
        :param kwargs: Additional arguments for `BaseLoadROId`.
        """
        MapTransform.__init__(self, [image_key], allow_missing_keys=False)
        BaseLoadROId.__init__(self, image_key=image_key, **kwargs)
        self.foreground_threshold = foreground_threshold

    def _load_slide_at_level(self, slide_obj: _OpenSlideOrCuImage, level: int) -> np.ndarray:
        """Load full slide array at the given magnification level.

        This is a manual workaround for a MONAI bug (https://github.com/Project-MONAI/MONAI/issues/3415)
        fixed in a currently unreleased PR (https://github.com/Project-MONAI/MONAI/pull/3417).

        :param reader: A MONAI `WSIReader` using cuCIM backend.
        :param slide_obj: The cuCIM image object returned by `reader.read(<image_file>)`.
        :param level: Index of the desired magnification level as defined in the `slide_obj` headers.
        :return: The loaded image array in (C, H, W) format.
        """
        size = self._get_size_at_level(slide_obj, level)
        slide, _ = self.reader.get_data(slide_obj, size=size, level=level)  # loaded as RGB PIL image
        return slide

    def _get_bounding_box(self, slide_obj: _OpenSlideOrCuImage) -> box_utils.Box:
        """Estimate bounding box at the lowest resolution (i.e. highest level) of the slide."""
        highest_level = self._get_highest_level(slide_obj)
        scale = self._get_scale_at_level(slide_obj, highest_level)
        slide = self._load_slide_at_level(slide_obj, level=highest_level)

        foreground_mask, threshold = segment_foreground(slide, self.foreground_threshold)
        self.foreground_threshold = threshold
        bbox = scale * box_utils.get_bounding_box(foreground_mask).add_margin(self.margin)
        return bbox

    def __call__(self, data: Dict) -> Dict:
        image_obj: _OpenSlideOrCuImage = self.reader.read(data[self.image_key])

        level0_bbox = self._get_bounding_box(image_obj)

        # cuCIM/OpenSlide takes absolute location coordinates in the level 0 reference frame,
        # but relative region size in pixels at the chosen level
        origin = (level0_bbox.y, level0_bbox.x)
        scale = self._get_scale_at_level(image_obj, self.level)
        scaled_bbox = level0_bbox / scale

        data[self.image_key], _ = self.reader.get_data(image_obj, location=origin, level=self.level,
                                                       size=(scaled_bbox.h, scaled_bbox.w))
        data[SlideKey.ORIGIN] = origin
        data[SlideKey.SCALE] = scale
        data[SlideKey.FOREGROUND_THRESHOLD] = self.foreground_threshold

        image_obj.close()
        return data


class LoadMaskROId(MapTransform, BaseLoadROId, Generic[_OpenSlideOrCuImage]):
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
        :param image_key: Image key in the input and output dictionaries.
        :param mask_key: Mask key in the input and output dictionaries.
        :param kwargs: Additional arguments for `BaseLoadROId`.
        """
        MapTransform.__init__(self, [image_key, mask_key], allow_missing_keys=False)
        BaseLoadROId.__init__(self, image_key=image_key, **kwargs)
        self.mask_key = mask_key

    def _get_bounding_box(self, mask_obj: _OpenSlideOrCuImage) -> box_utils.Box:
        """Estimate bounding box at the lowest resolution (i.e. highest level) of the mask."""
        highest_level = self._get_highest_level(mask_obj)
        scale = self._get_scale_at_level(mask_obj, highest_level)
        mask, _ = self.reader.get_data(mask_obj, level=highest_level)  # loaded as RGB PIL image

        foreground_mask = mask[0] > 0  # PANDA segmentation mask is in 'R' channel

        bbox = box_utils.get_bounding_box(foreground_mask)
        padded_bbox = bbox.add_margin(self.margin)
        scaled_bbox = scale * padded_bbox
        return scaled_bbox

    def __call__(self, data: Dict) -> Dict:
        mask_obj: _OpenSlideOrCuImage = self.reader.read(data[self.mask_key])
        image_obj: _OpenSlideOrCuImage = self.reader.read(data[self.image_key])

        level0_bbox = self._get_bounding_box(mask_obj)

        # cuCIM/OpenSlide take absolute location coordinates in the level 0 reference frame,
        # but relative region size in pixels at the chosen level
        scale = self._get_scale_at_level(mask_obj, self.level)
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

        mask_obj.close()
        image_obj.close()
        return data


class CucimLoadROId(CuImageMixin, LoadROId['CuImage']):
    """Transform that loads a pathology slide, cropped to the foreground bounding box (ROI). This transform uses cuCIM
    backend to load the whole slide image (WSI). Note that cucim is a requires a GPU to run."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(backend=WSIBackend.CUCIM, **kwargs)


class OpenSlideLoadROId(OpenSlideMixin, LoadROId['OpenSlide']):
    """Transform that loads a pathology slide, cropped to the foreground bounding box (ROI). This transform uses
    OpenSlide to load the whole slide image (WSI)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(backend=WSIBackend.OPENSLIDE, **kwargs)


class CucimLoadMaskROId(CuImageMixin, LoadMaskROId['CuImage']):
    """Transform that loads a pathology slide and mask, cropped to the mask bounding box (ROI) defined by the mask.
    This transform uses cuCIM backend to load the whole slide image (WSI). Note that cucim is a requires a GPU to
    run."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(backend=WSIBackend.CUCIM, **kwargs)


class OpenSlideLoadMaskROId(OpenSlideMixin, LoadMaskROId['OpenSlide']):
    """Transform that loads a pathology slide and mask, cropped to the mask bounding box (ROI) defined by the mask.
    This transform uses OpenSlides backend to load the whole slide image (WSI)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(backend=WSIBackend.OPENSLIDE, **kwargs)


class LoadingParams(param.Parameterized):
    """Parameters for loading a whole slide image."""

    image_key: str = param.String(
        default=SlideKey.IMAGE,
        doc="Key for the image in the data dictionary.")
    mask_key: str = param.String(
        default=SlideKey.MASK,
        doc="Key for the mask in the data dictionary. This only applies to `LoadMaskROId`.")
    level: int = param.Integer(
        default=0,
        doc="Magnification level to load from the raw multi-scale files")
    margin: int = param.Integer(
        default=0, doc="Amount in pixels by which to enlarge the estimated bounding box for cropping")
    backend: WSIBackend = param.ClassSelector(
        default=WSIBackend.CUCIM,
        class_=WSIBackend,
        doc="WSI reader backend.")
    roi_type: ROIType = param.ClassSelector(
        default=ROIType.FOREGROUND,
        class_=ROIType,
        doc="ROI type to use for cropping the slide.")
    foreground_threshold: Optional[float] = param.Number(
        default=None,
        bounds=(0, 255.),
        allow_None=True,
        doc="Threshold for foreground mask. If None, the threshold is selected automatically with otsu thresholding."
        "This only applies to `LoadROId`.")

    @property
    def load_roid_transforns_dict(self) -> Dict[Tuple[WSIBackend, ROIType], Callable]:
        """Returns a dictionary of transforms to load a slide and mask, cropped to the mask bounding box (ROI) defined
        by either the mask or the foreground."""
        return {
            (WSIBackend.CUCIM, ROIType.FOREGROUND): CucimLoadROId,
            (WSIBackend.CUCIM, ROIType.MASK): CucimLoadMaskROId,
            (WSIBackend.OPENSLIDE, ROIType.FOREGROUND): OpenSlideLoadROId,
            (WSIBackend.OPENSLIDE, ROIType.MASK): OpenSlideLoadMaskROId,
        }

    def get_load_roid_transform(self) -> Callable:
        """Returns a transform to load a slide and mask, cropped to the mask bounding box (ROI) defined by either the
        mask or the foreground."""
        return self.load_roid_transforns_dict[(self.backend, self.roi_type)](**self.get_transform_args())

    def get_transform_args(self) -> Dict[str, Any]:
        """Returns a dictionary of arguments for the transform."""
        args = dict(
            image_key=self.image_key,
            level=self.level,
            margin=self.margin,
            backend_args=self.get_additionl_backend_args()
        )
        if self.roi_type == ROIType.FOREGROUND:
            args.update(dict(foreground_threshold=self.foreground_threshold))
        elif self.roi_type == ROIType.MASK:
            args.update(dict(mask_key=self.mask_key))
        return args

    def get_additionl_backend_args(self) -> Dict[str, Any]:
        """Returns a dictionary of additional arguments for the WSI reader backend. Multi processing is
        enabled since monai 1.0.0 by specifying num_workers > 0 with CuCIM backend only.
        """
        return dict()
