#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Mapping, Sequence, Tuple, Union, Callable, Dict

import torch
import numpy as np
import PIL
from PIL import PngImagePlugin
from monai.config.type_definitions import KeysCollection
from monai.transforms import MapTransform, Randomizable, Transform
from monai.utils.enums import WSIPatchKeys
from monai.data.meta_tensor import MetaTensor
from health_azure.logging import elapsed_timer
from health_ml.utils.box_utils import Box
from torchvision.transforms.functional import to_tensor

from health_cpath.models.encoders import TileEncoder
from health_cpath.preprocessing.create_tiles_dataset import get_tile_id
from health_cpath.utils.naming import SlideKey, TileKey

PathOrString = Union[Path, str]


def load_pil_image(image_path: PathOrString) -> PIL.Image.Image:
    """Load a PIL image in RGB format from the given path"""
    with PngImagePlugin.PngImageFile(image_path) as pil_png:
        image = np.asarray(pil_png)
    return image


def load_image_as_tensor(image_path: PathOrString, scale_intensity: bool = True) -> torch.Tensor:
    """Load an image as a tensor from the given path

    :param image_path: path to the image
    :param scale_intensity: if True, use `to_tensor` from torchvision which scales the image pixel intensities to
    [0, 1] by defaul as [C, H, W] tensors. Otherwise, only transpose the image to [C, H, W] format and return it as a
    torch tensor.
    """

    pil_image = load_pil_image(image_path)  # pil_image is in channels last format [H, W, C]
    if scale_intensity:
        return to_tensor(pil_image)  # to_tensor scales the image pixel intensities to [0, 1] as [C, H, W] tensors
    else:
        return torch.from_numpy(pil_image.transpose((2, 0, 1))).contiguous()  # only transpose to [C, H, W]


def load_image_stack_as_tensor(image_paths: Sequence[PathOrString],
                               progress: bool = False,
                               scale_intensity: bool = True) -> torch.Tensor:
    """Load a batch of images of the same size as a tensor from the given paths

    :param image_paths: paths to the images
    :param progress: if True, show a progress bar
    :param scale_intensity: if True, use `to_tensor` from torchvision which scales the image pixel intensities to
    [0, 1] by defaul as [C, H, W] tensors. Otherwise, only transpose the image to [C, H, W] format and return it as a
    torch tensor.
    """

    loading_generator = (load_image_as_tensor(path, scale_intensity) for path in image_paths)
    if progress:
        from tqdm import tqdm
        loading_generator = tqdm(loading_generator, desc="Loading image stack",
                                 total=len(image_paths), leave=False)
    image_tensors = list(loading_generator)
    return torch.stack(image_tensors, dim=0)


def transform_dict_adaptor(function: Callable, k_input: str = None, k_output: str = None) -> Callable:
    """Adapt transformations to work with an input dictionary (rather than a tensor).
       We can't reuse monai.transforms.adaptors because it is only compatible with transformations that accept
       a dict as input.

    :param function: a transformation function
    :param k_input: key of the input dictionary that contains the object
        to which function should be applied
    :param k_output: key of the input dictionary where to place the function output. If None the ouput of
        the transformation is returned

    :return: adapted transformation
    """
    def _inner(ditems: dict) -> Dict:
        if k_input is None:
            dinputs = ditems
        else:
            dinputs = ditems[k_input]
        ret = function(dinputs)
        if k_output is None:
            ditems = ret
        else:
            if isinstance(ret, type(ditems[k_output])):
                ditems[k_output] = ret
            else:
                raise ValueError("The transformation is not expect to change the type."
                                 "Check input and output are used correctly ")
        return ditems
    return _inner


class LoadTiled(MapTransform):
    """Dictionary transform to load an individual image tile as a tensor from an input path"""

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        """
        :param keys: Key(s) for the image path(s) in the input dictionary.
        :param allow_missing_keys: If `False` (default), raises an exception when an input
        dictionary is missing any of the specified keys.
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping) -> Mapping:
        out_data = dict(data)  # create shallow copy
        for key in self.key_iterator(out_data):
            out_data[key] = load_image_as_tensor(data[key])
        return out_data


class LoadTilesBatchd(MapTransform):
    """Dictionary transform to load a batch of image tiles as a tensor from a list of input paths"""

    # Cannot reuse MONAI readers because they support stacking only images with no channels
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False,
                 progress: bool = False, scale_intensity: bool = True) -> None:
        """
        :param keys: Key(s) for the image path(s) in the input dictionary.
        :param allow_missing_keys: If `False` (default), raises an exception when an input
        dictionary is missing any of the specified keys.
        :param progress: Whether to display a tqdm progress bar.
        :param scale_intensity: if True, use `to_tensor` from torchvision which scales the image pixel intensities to
        [0, 1] by defaul as [C, H, W] tensors. Otherwise, only transpose the image to [C, H, W] format and return it as
        a torch tensor.
        """

        super().__init__(keys, allow_missing_keys)
        self.progress = progress
        self.scale_intensity = scale_intensity

    def __call__(self, data: Mapping) -> Mapping:
        out_data = dict(data)  # create shallow copy
        for key in self.key_iterator(out_data):
            out_data[key] = load_image_stack_as_tensor(
                data[key], progress=self.progress, scale_intensity=self.scale_intensity
            )
        return out_data


class EncodeTilesBatchd(MapTransform):
    """Dictionary transform to extract features from a batch tensor of image tiles"""

    def __init__(self,
                 keys: KeysCollection,
                 encoder: TileEncoder,
                 allow_missing_keys: bool = False,
                 chunk_size: int = 0) -> None:
        """
        :param keys: Key(s) for the image tensor(s) in the input dictionary.
        :param encoder: The tile encoder to use for feature extraction.
        :param allow_missing_keys: If `False` (default), raises an exception when an input
        dictionary is missing any of the specified keys.
        :param chunk_size: if > 0, extracts features in chunks of size chunk_size.
        """
        super().__init__(keys, allow_missing_keys)
        self.encoder = encoder
        self.chunk_size = chunk_size

    @torch.no_grad()
    def _encode_tiles(self, images: torch.Tensor) -> torch.Tensor:
        device = next(self.encoder.parameters()).device
        if self.chunk_size > 0:
            embeddings = []
            chunks = torch.split(images, self.chunk_size)
            # TODO parallelize encoding - keep metadata and images aligned
            for chunk in chunks:
                chunk_embeddings = self._encode_images(chunk, device)
                embeddings.append(chunk_embeddings)
            return torch.cat(embeddings)
        else:
            return self._encode_images(images, device)

    def _encode_images(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        images = images.to(device)
        embeddings = self.encoder(images)
        del images
        torch.cuda.empty_cache()
        return embeddings

    def __call__(self, data: Mapping) -> Mapping:
        out_data = dict(data)  # create shallow copy
        for key in self.key_iterator(out_data):
            out_data[key] = self._encode_tiles(data[key])
        return out_data


def take_indices(data: Sequence, indices: np.ndarray) -> Sequence:
    if isinstance(data, (np.ndarray, torch.Tensor)):
        return data[indices]  # type: ignore
    elif isinstance(data, Sequence):
        return [data[i] for i in indices]
    else:
        raise ValueError(f"Data of type {type(data)} is not indexable")


class Subsampled(MapTransform, Randomizable):
    """Dictionary transform to randomly subsample the data down to a fixed maximum length"""

    def __init__(self, keys: KeysCollection, max_size: int,
                 allow_missing_keys: bool = False) -> None:
        """
        :param keys: Key(s) for all batch elements that must be subsampled.
        :param max_size: Each specified array, tensor, or sequence will be subsampled uniformly at
        random down to `max_size` along their first dimension. If shorter, the elements are merely
        shuffled.
        :param allow_missing_keys: If `False` (default), raises an exception when an input
        dictionary is missing any of the specified keys.
        """
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.max_size = max_size
        self._indices: np.ndarray

    def randomize(self, total_size: int) -> None:
        subsample_size = min(self.max_size, total_size)
        self._indices = self.R.choice(total_size, size=subsample_size, replace=False)

    def __call__(self, data: Mapping) -> Mapping:
        out_data = dict(data)  # create shallow copy
        size = len(data[self.keys[0]])
        self.randomize(size)
        for key in self.key_iterator(out_data):
            out_data[key] = take_indices(data[key], self._indices)
        return out_data


class ExtractCoordinatesd(MapTransform):
    """Extract the coordinates of the tiles returned as meta data by monai transforms to hi-ml-cpath format where
    the coordinates are represented as TileKey.TILE_LEFT, TileKey.TILE_TOP, TileKey.TILE_RIGHT, TileKey.TILE_BOTTOM."""

    def __init__(self, image_key: str, tile_size: int) -> None:
        self.tile_size = tile_size
        self.image_key = image_key

    def extract_coordinates(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the coordinates of the tiles from the metadata."""
        assert isinstance(data[self.image_key], MetaTensor), f"Expected MetaTensor, got {type(data[self.image_key])}"
        ys, xs = data[self.image_key].meta[WSIPatchKeys.LOCATION]
        return ys, xs

    def extract_scale_factor(self, data: Dict) -> int:
        """Extract the scale factor of the tiles from the metadata to rescale the coordinates to highest resolution."""
        return int(data.get(SlideKey.SCALE, 1))

    def extract_offset(self, data: Dict) -> Tuple[int, int]:
        """Extract the offset of the tiles from the metadata to translate to (0, 0) origin."""
        return data.get(SlideKey.ORIGIN, (0, 0))

    def set_coordinates(self, data: Dict, xs: np.ndarray, ys: np.ndarray) -> None:
        """Set the coordinates of the tiles in the metadata."""
        # Extract the scale factor and offset to rescale the coordinates to highest resolution
        scale_factor = self.extract_scale_factor(data)
        offset_y, offset_x = self.extract_offset(data)
        # We set the coordinates of the tiles as top left and bottom right coordinates
        data[TileKey.TILE_LEFT] = torch.tensor((xs * scale_factor + offset_x))
        data[TileKey.TILE_TOP] = torch.tensor((ys * scale_factor + offset_y))
        data[TileKey.TILE_RIGHT] = data[TileKey.TILE_LEFT] + self.tile_size * scale_factor
        data[TileKey.TILE_BOTTOM] = data[TileKey.TILE_TOP] + self.tile_size * scale_factor

    def set_tile_and_slide_ids(self, data: Dict, xs: np.ndarray, ys: np.ndarray) -> None:
        """Set the tile and slide id in the metadata."""
        h, w = self.tile_size, self.tile_size
        bag_size = data[SlideKey.IMAGE].meta[WSIPatchKeys.COUNT]
        data[TileKey.TILE_ID] = [get_tile_id(data[SlideKey.SLIDE_ID], Box(x=x, y=y, w=w, h=h)) for x, y in zip(xs, ys)]
        data[SlideKey.SLIDE_ID] = [data[SlideKey.SLIDE_ID]] * bag_size

    def convert_tiles_and_label_to_tensors(self, data: Dict) -> None:
        """Convert the tiles and label to tensors."""
        data[SlideKey.IMAGE] = data[SlideKey.IMAGE].as_tensor()
        data[SlideKey.LABEL] = torch.tensor(data[SlideKey.LABEL])

    def __call__(self, data: Mapping) -> Mapping:
        out_data = dict(data)
        ys, xs = self.extract_coordinates(out_data)
        self.set_coordinates(out_data, xs=xs, ys=ys)
        self.set_tile_and_slide_ids(out_data, xs=xs, ys=ys)
        self.convert_tiles_and_label_to_tensors(out_data)
        return out_data


class MetaTensorToTensord(MapTransform):
    """Converts a MetaTensor to a Tensor."""

    def __call__(self, data: Mapping) -> Mapping:
        out_data = dict(data)
        for key in self.key_iterator(out_data):
            assert isinstance(out_data[key], MetaTensor), f"Expected MetaTensor, got {type(out_data[key])}"
            out_data[key] = out_data[key].as_tensor()
        return out_data


class TimerWrapper(Transform):
    """Transform that measures the time it takes to execute the transform. Useful for debugging. This can be used by
    wrapping the transform in a TimerWrapperd transform.
    Example:
        transform = Compose([TimerWrapperd(LoadImaged(keys="image")), TimerWrapperd(MetaTensorToTensord(keys="image"))])
    """

    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, data: Mapping) -> Mapping:
        message = f"{self.transform.__class__.__name__}, Slide {data[SlideKey.SLIDE_ID]}"
        with elapsed_timer(message):
            out_data = self.transform(data)
        return out_data
