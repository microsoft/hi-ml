import datetime
import os
import torch
import param
from torch import Tensor
from typing import Any, Callable, Generator, List, Optional
from health_azure.utils import ENV_LOCAL_RANK
from health_cpath.models.transforms import ExtractCoordinatesd
from health_cpath.utils.naming import ModelKey, SlideKey
from health_ml.utils.bag_utils import multibag_collate
from monai.transforms.utility.dictionary import SplitDimd
from monai.transforms import GridPatchd, RandGridPatchd
from contextlib import contextmanager
from time import time
# from typing import Any, Callable, Dict, Generator, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
# from monai.config.type_definitions import KeysCollection
# from monai.config.type_definitions import NdarrayOrTensor
# from monai.transforms.transform import MapTransform, RandomizableTransform
# from monai.transforms.transform import Transform
# from monai.data.meta_tensor import MetaTensor
# import numpy as np
# from monai.utils.enums import GridPatchSort, PytorchPadMode, TransformBackends, WSIPatchKeys, NumpyPadMode
# from monai.utils.misc import ensure_tuple, ensure_tuple_rep, ensure_tuple_size
# from monai.utils.module import look_up_option
# from monai.transforms.utils import convert_pad_mode
# from monai.utils.type_conversion import convert_data_type
# from itertools import product, starmap


def image_collate(batch: List) -> Any:
    """
        Combine instances from a list of dicts into a single dict, by stacking them along first dim
        [{'image' : 3xHxW}, {'image' : 3xHxW}, {'image' : 3xHxW}...] - > {'image' : Nx3xHxW}
        followed by the default collate which will form a batch BxNx3xHxW.
        The list of dicts refers to the the list of tiles produced by the Rand/GridPatchd transform applied on a WSI.
    """
    # print_message_from_rank_pid(f"Collating {len(batch)} slides")
    for i, item in enumerate(batch):
        # The tiles have been splited into a list of dicts, each dict containing a single tile to be able to apply
        # tile wise transforms. We need to stack them back together.
        data = item[0]
        assert isinstance(data[SlideKey.IMAGE], Tensor), f"Expected torch.Tensor, got {type(data[SlideKey.IMAGE])}"
        data[SlideKey.IMAGE] = torch.stack([ix[SlideKey.IMAGE] for ix in item], dim=0)
        batch[i] = data
    return multibag_collate(batch)


def print_message_from_rank_pid(message: str = '') -> None:
    print(f"{datetime.datetime.now()}, Rank {os.getenv(ENV_LOCAL_RANK)}, PID {os.getpid()}, {message}")


@contextmanager
def elapsed_timer(message: str) -> Generator:
    start = time()
    yield
    elapsed = time() - start
    print_message_from_rank_pid(f"{message} took {elapsed:.2f} seconds")


# def get_valid_patch_size(image_size: Sequence[int], patch_size: Union[Sequence[int], int]) -> Tuple[int, ...]:
#     """
#     Given an image of dimensions `image_size`, return a patch size tuple taking the dimension from `patch_size` if this
#     is not 0/None. Otherwise, or if `patch_size` is shorter than `image_size`, the dimension from `image_size` is taken. This ensures
#     the returned patch size is within the bounds of `image_size`. If `patch_size` is a single number this is
#     interpreted as a
#     patch of the same dimensionality of `image_size` with that size in each dimension.
#     """
#     ndim = len(image_size)
#     patch_size_ = ensure_tuple_size(patch_size, ndim)

#     # ensure patch size dimensions are not larger than image dimension, if a dimension is None or 0 use whole dimension
#     return tuple(min(ms, ps or ms) for ms, ps in zip(image_size, patch_size_))


# def iter_patch_position(
#     image_size: Sequence[int],
#     patch_size: Union[Sequence[int], int],
#     start_pos: Sequence[int] = (),
#     overlap: Union[Sequence[float], float] = 0.0,
#     padded: bool = False,
# ):
#     """
#     Yield successive tuples of upper left corner of patches of size `patch_size` from an array of dimensions
#     `image_size`.
#     The iteration starts from position `start_pos` in the array, or starting at the origin if this isn't provided. Each
#     patch is chosen in a contiguous grid using a rwo-major ordering.

#     Args:
#         image_size: dimensions of array to iterate over
#         patch_size: size of patches to generate slices for, 0 or None selects whole dimension
#         start_pos: starting position in the array, default is 0 for each dimension
#         overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
#             If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
#         padded: if the image is padded so the patches can go beyond the borders. Defaults to False.

#     Yields:
#         Tuples of positions defining the upper left corner of each patch
#     """

#     # ensure patchSize and startPos are the right length
#     ndim = len(image_size)
#     patch_size_ = get_valid_patch_size(image_size, patch_size)
#     start_pos = ensure_tuple_size(start_pos, ndim)
#     overlap = ensure_tuple_rep(overlap, ndim)

#     # calculate steps, which depends on the amount of overlap
#     steps = tuple(round(p * (1.0 - o)) for p, o in zip(patch_size_, overlap))

#     # calculate the last starting location (depending on the padding)
#     end_pos = image_size if padded else tuple(s - round(p) + 1 for s, p in zip(image_size, patch_size_))

#     # collect the ranges to step over each dimension
#     ranges = starmap(range, zip(start_pos, end_pos, steps))

#     # choose patches by applying product to the ranges
#     return product(*ranges)


# def iter_patch_slices(
#     image_size: Sequence[int],
#     patch_size: Union[Sequence[int], int],
#     start_pos: Sequence[int] = (),
#     overlap: Union[Sequence[float], float] = 0.0,
#     padded: bool = True,
# ) -> Generator[Tuple[slice, ...], None, None]:
#     """
#     Yield successive tuples of slices defining patches of size `patch_size` from an array of dimensions `image_size`.
#     The iteration starts from position `start_pos` in the array, or starting at the origin if this isn't provided. Each
#     patch is chosen in a contiguous grid using a rwo-major ordering.

#     Args:
#         image_size: dimensions of array to iterate over
#         patch_size: size of patches to generate slices for, 0 or None selects whole dimension
#         start_pos: starting position in the array, default is 0 for each dimension
#         overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
#             If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
#         padded: if the image is padded so the patches can go beyond the borders. Defaults to False.

#     Yields:
#         Tuples of slice objects defining each patch
#     """

#     # ensure patch_size has the right length
#     patch_size_ = get_valid_patch_size(image_size, patch_size)

#     # create slices based on start position of each patch
#     for position in iter_patch_position(
#         image_size=image_size, patch_size=patch_size_, start_pos=start_pos, overlap=overlap, padded=padded
#     ):
#         yield tuple(slice(s, s + p) for s, p in zip(position, patch_size_))


# def iter_patch(
#     arr: np.ndarray,
#     patch_size: Union[Sequence[int], int] = 0,
#     start_pos: Sequence[int] = (),
#     overlap: Union[Sequence[float], float] = 0.0,
#     copy_back: bool = True,
#     mode: Optional[str] = NumpyPadMode.WRAP,
#     slide_id: str = '',
#     **pad_opts: Dict,
# ):
#     """
#     Yield successive patches from `arr` of size `patch_size`. The iteration can start from position `start_pos` in `arr`
#     but drawing from a padded array extended by the `patch_size` in each dimension (so these coordinates can be negative
#     to start in the padded region). If `copy_back` is True the values from each patch are written back to `arr`.

#     Args:
#         arr: array to iterate over
#         patch_size: size of patches to generate slices for, 0 or None selects whole dimension
#         start_pos: starting position in the array, default is 0 for each dimension
#         overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
#             If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
#         copy_back: if True data from the yielded patches is copied back to `arr` once the generator completes
#         mode: One of the listed string values in ``monai.utils.NumpyPadMode`` or ``monai.utils.PytorchPadMode``,
#             or a user supplied function. If None, no wrapping is performed. Defaults to ``"wrap"``.
#         pad_opts: padding options, see `numpy.pad`

#     Yields:
#         Patches of array data from `arr` which are views into a padded array which can be modified, if `copy_back` is
#         True these changes will be reflected in `arr` once the iteration completes.

#     Note:
#         coordinate format is:

#             [1st_dim_start, 1st_dim_end,
#              2nd_dim_start, 2nd_dim_end,
#              ...,
#              Nth_dim_start, Nth_dim_end]]

#     """

#     # ensure patchSize and startPos are the right length
#     patch_size_ = get_valid_patch_size(arr.shape, patch_size)
#     start_pos = ensure_tuple_size(start_pos, arr.ndim)

#     # set padded flag to false if pad mode is None
#     padded = bool(mode)
#     # pad image by maximum values needed to ensure patches are taken from inside an image
#     if padded:
#         arrpad = np.pad(arr, tuple((p, p) for p in patch_size_),
#                         look_up_option(mode, NumpyPadMode).value, **pad_opts)
#         # choose a start position in the padded image
#         start_pos_padded = tuple(s + p for s, p in zip(start_pos, patch_size_))

#         # choose a size to iterate over which is smaller than the actual padded image to prevent producing
#         # patches which are only in the padded regions
#         iter_size = tuple(s + p for s, p in zip(arr.shape, patch_size_))
#     else:
#         arrpad = arr
#         start_pos_padded = start_pos
#         iter_size = arr.shape

#     for slices in iter_patch_slices(iter_size, patch_size_, start_pos_padded, overlap, padded=padded):
#         # compensate original image padding
#         if padded:
#             coords_no_pad = tuple((coord.start - p, coord.stop - p) for coord, p in zip(slices, patch_size_))
#         else:
#             coords_no_pad = tuple((coord.start, coord.stop) for coord in slices)
#         yield arrpad[slices], np.asarray(coords_no_pad)  # data and coords (in numpy; works with torch loader)

#     # copy back data from the padded image if required
#     if copy_back:
#         slices = tuple(slice(p, p + s) for p, s in zip(patch_size_, arr.shape))
#         arr[...] = arrpad[slices]


# class GridPatch(Transform):
#     """
#     Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps.
#     It can sort the patches and return all or a subset of them.

#     Args:
#         patch_size: size of patches to generate slices for, 0 or None selects whole dimension
#         offset: offset of starting position in the array, default is 0 for each dimension.
#         num_patches: number of patches to return. Defaults to None, which returns all the available patches.
#             If the required patches are more than the available patches, padding will be applied.
#         overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
#             If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
#         sort_fn: when `num_patches` is provided, it determines if keep patches with highest values (`"max"`),
#             lowest values (`"min"`), or in their default order (`None`). Default to None.
#         threshold: a value to keep only the patches whose sum of intensities are less than the threshold.
#             Defaults to no filtering.
#         pad_mode: refer to NumpyPadMode and PytorchPadMode. If None, no padding will be applied. Defaults to
#         ``"constant"``.
#         pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.

#     Returns:
#         MetaTensor: A MetaTensor consisting of a batch of all the patches with associated metadata

#     """

#     backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

#     def __init__(
#         self,
#         patch_size: Sequence[int],
#         offset: Optional[Sequence[int]] = None,
#         num_patches: Optional[int] = None,
#         overlap: Union[Sequence[float], float] = 0.0,
#         sort_fn: Optional[str] = None,
#         threshold: Optional[float] = None,
#         pad_mode: str = PytorchPadMode.CONSTANT,
#         **pad_kwargs,
#     ):
#         self.patch_size = ensure_tuple(patch_size)
#         self.offset = ensure_tuple(offset) if offset else (0,) * len(self.patch_size)
#         self.pad_mode: Optional[NumpyPadMode] = convert_pad_mode(dst=np.zeros(1), mode=pad_mode) if pad_mode else None
#         self.pad_kwargs = pad_kwargs
#         self.overlap = overlap
#         self.num_patches = num_patches
#         self.sort_fn = sort_fn.lower() if sort_fn else None
#         self.threshold = threshold

#     def filter_threshold(self, image_np: np.ndarray, locations: np.ndarray):
#         """
#         Filter the patches and their locations according to a threshold
#         Args:
#             image_np: a numpy.ndarray representing a stack of patches
#             locations: a numpy.ndarray representing the stack of location of each patch
#         """
#         if self.threshold is not None:
#             n_dims = len(image_np.shape)
#             idx = np.argwhere(image_np.sum(axis=tuple(range(1, n_dims))) < self.threshold).reshape(-1)
#             image_np = image_np[idx]
#             locations = locations[idx]
#         return image_np, locations

#     def filter_count(self, image_np: np.ndarray, locations: np.ndarray):
#         """
#         Sort the patches based on the sum of their intensity, and just keep `self.num_patches` of them.
#         Args:
#             image_np: a numpy.ndarray representing a stack of patches
#             locations: a numpy.ndarray representing the stack of location of each patch
#         """
#         if self.sort_fn is None:
#             image_np = image_np[: self.num_patches]
#             locations = locations[: self.num_patches]
#         elif self.num_patches is not None:
#             n_dims = len(image_np.shape)
#             if self.sort_fn == GridPatchSort.MIN:
#                 idx = np.argsort(image_np.sum(axis=tuple(range(1, n_dims))))
#             elif self.sort_fn == GridPatchSort.MAX:
#                 idx = np.argsort(-image_np.sum(axis=tuple(range(1, n_dims))))
#             else:
#                 raise ValueError(f'`sort_fn` should be either "min", "max" or None! {self.sort_fn} provided!')
#             idx = idx[: self.num_patches]
#             image_np = image_np[idx]
#             locations = locations[idx]
#         return image_np, locations

#     def __call__(self, array: NdarrayOrTensor, slide_id: str = None):
#         with elapsed_timer(f"{slide_id} {array.shape} - create patches              "):
#             with elapsed_timer(f"      {slide_id} {array.shape} - create iter patch     "):
#                 # create the patch iterator which sweeps the image row-by-row
#                 array_np, *_ = convert_data_type(array, np.ndarray)
#                 patch_iterator = iter_patch(
#                     array_np,
#                     patch_size=(None,) + self.patch_size,  # expand to have the channel dim
#                     start_pos=(0,) + self.offset,  # expand to have the channel dim
#                     overlap=self.overlap,
#                     copy_back=False,
#                     mode=self.pad_mode,
#                     **self.pad_kwargs,
#                 )
#             with elapsed_timer(f"      {slide_id} {array.shape} - extract patches       "):
#                 patches = list(zip(*patch_iterator))
#             with elapsed_timer(f"      {slide_id} {len(patches[0])} - convert to numpy array"):
#                 patched_image = np.asarray(patches[0])
#             with elapsed_timer(f"      {slide_id} {patched_image.shape} - extract locations  "):
#                 locations = np.asarray(patches[1])[:, 1:, 0]  # only keep the starting location

#         with elapsed_timer(f"{slide_id} {array.shape} - filter patches              "):
#             # Filter patches
#             if self.num_patches:
#                 patched_image, locations = self.filter_count(patched_image, locations)
#             elif self.threshold:
#                 patched_image, locations = self.filter_threshold(patched_image, locations)

#         with elapsed_timer(f"{slide_id} {array.shape} - pad patches                 "):
#             # Pad the patch list to have the requested number of patches
#             if self.num_patches:
#                 padding = self.num_patches - len(patched_image)
#                 if padding > 0:
#                     patched_image = np.pad(
#                         patched_image,
#                         [[0, padding], [0, 0]] + [[0, 0]] * len(self.patch_size),
#                         constant_values=self.pad_kwargs.get("constant_values", 0),
#                     )
#                     locations = np.pad(locations, [[0, padding], [0, 0]], constant_values=0)

#         with elapsed_timer(f"{slide_id} {array.shape} - convert to MetaTensor       "):
#             # Convert to MetaTensor
#             metadata = array.meta if isinstance(array, MetaTensor) else MetaTensor.get_default_meta()
#             metadata[WSIPatchKeys.LOCATION] = locations.T
#             metadata[WSIPatchKeys.COUNT] = len(locations)
#             metadata["spatial_shape"] = np.tile(np.array(self.patch_size), (len(locations), 1)).T
#             output = MetaTensor(x=patched_image, meta=metadata)
#             output.is_batch = True

#         return output


# class GridPatchd(MapTransform):
#     """
#     Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps.
#     It can sort the patches and return all or a subset of them.

#     Args:
#         keys: keys of the corresponding items to be transformed.
#         patch_size: size of patches to generate slices for, 0 or None selects whole dimension
#         offset: starting position in the array, default is 0 for each dimension.
#             np.random.randint(0, patch_size, 2) creates random start between 0 and `patch_size` for a 2D image.
#         num_patches: number of patches to return. Defaults to None, which returns all the available patches.
#         overlap: amount of overlap between patches in each dimension. Default to 0.0.
#         sort_fn: when `num_patches` is provided, it determines if keep patches with highest values (`"max"`),
#             lowest values (`"min"`), or in their default order (`None`). Default to None.
#         threshold: a value to keep only the patches whose sum of intensities are less than the threshold.
#             Defaults to no filtering.
#         pad_mode: refer to NumpyPadMode and PytorchPadMode. If None, no padding will be applied. Defaults to
#         ``"constant"``.
#         allow_missing_keys: don't raise exception if key is missing.
#         pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.

#     Returns:
#         a list of dictionaries, each of which contains the all the original key/value with the values for `keys`
#             replaced by the patches. It also add the following new keys:

#             "patch_location": the starting location of the patch in the image,
#             "patch_size": size of the extracted patch
#             "num_patches": total number of patches in the image
#             "offset": the amount of offset for the patches in the image (starting position of upper left patch)
#     """

#     backend = GridPatch.backend

#     def __init__(
#         self,
#         keys: KeysCollection,
#         patch_size: Sequence[int],
#         offset: Optional[Sequence[int]] = None,
#         num_patches: Optional[int] = None,
#         overlap: float = 0.0,
#         sort_fn: Optional[str] = None,
#         threshold: Optional[float] = None,
#         pad_mode: str = PytorchPadMode.CONSTANT,
#         allow_missing_keys: bool = False,
#         **pad_kwargs,
#     ):
#         super().__init__(keys, allow_missing_keys)
#         self.patcher = GridPatch(
#             patch_size=patch_size,
#             offset=offset,
#             num_patches=num_patches,
#             overlap=overlap,
#             sort_fn=sort_fn,
#             threshold=threshold,
#             pad_mode=pad_mode,
#             **pad_kwargs,
#         )

#     def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
#         d = dict(data)
#         for key in self.key_iterator(d):
#             d[key] = self.patcher(d[key], data[SlideKey.SLIDE_ID])
#         return d


# class RandGridPatch(GridPatch, RandomizableTransform):
#     """
#     Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps,
#     and with random offset for the minimal corner of the image, (0,0) for 2D and (0,0,0) for 3D.
#     It can sort the patches and return all or a subset of them.

#     Args:
#         patch_size: size of patches to generate slices for, 0 or None selects whole dimension
#         min_offset: the minimum range of offset to be selected randomly. Defaults to 0.
#         max_offset: the maximum range of offset to be selected randomly.
#             Defaults to image size modulo patch size.
#         num_patches: number of patches to return. Defaults to None, which returns all the available patches.
#         overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
#             If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
#         sort_fn: when `num_patches` is provided, it determines if keep patches with highest values (`"max"`),
#             lowest values (`"min"`), or in their default order (`None`). Default to None.
#         threshold: a value to keep only the patches whose sum of intensities are less than the threshold.
#             Defaults to no filtering.
#         pad_mode: refer to NumpyPadMode and PytorchPadMode. If None, no padding will be applied. Defaults to ``"constant"``.
#         pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.

#     Returns:
#         MetaTensor: A MetaTensor consisting of a batch of all the patches with associated metadata

#     """

#     backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

#     def __init__(
#         self,
#         patch_size: Sequence[int],
#         min_offset: Optional[Union[Sequence[int], int]] = None,
#         max_offset: Optional[Union[Sequence[int], int]] = None,
#         num_patches: Optional[int] = None,
#         overlap: Union[Sequence[float], float] = 0.0,
#         sort_fn: Optional[str] = None,
#         threshold: Optional[float] = None,
#         pad_mode: str = PytorchPadMode.CONSTANT,
#         **pad_kwargs,
#     ):
#         super().__init__(
#             patch_size=patch_size,
#             offset=(),
#             num_patches=num_patches,
#             overlap=overlap,
#             sort_fn=sort_fn,
#             threshold=threshold,
#             pad_mode=pad_mode,
#             **pad_kwargs,
#         )
#         self.min_offset = min_offset
#         self.max_offset = max_offset

#     def randomize(self, array):
#         if self.min_offset is None:
#             min_offset = (0,) * len(self.patch_size)
#         else:
#             min_offset = ensure_tuple_rep(self.min_offset, len(self.patch_size))
#         if self.max_offset is None:
#             max_offset = tuple(s % p for s, p in zip(array.shape[1:], self.patch_size))
#         else:
#             max_offset = ensure_tuple_rep(self.max_offset, len(self.patch_size))

#         self.offset = tuple(self.R.randint(low=low, high=high + 1) for low, high in zip(min_offset, max_offset))

#     def __call__(self, array: NdarrayOrTensor, randomize: bool = True, slide_id: str = None):
#         if randomize:
#             start_time = time.time()
#             self.randomize(array)
#             print_message_from_rank_pid(f"Randomize time: {time.time() - start_time}")
#         return super().__call__(array, slide_id=slide_id)


# class RandGridPatchd(RandomizableTransform, MapTransform):
#     """
#     Extract all the patches sweeping the entire image in a row-major sliding-window manner with possible overlaps,
#     and with random offset for the minimal corner of the image, (0,0) for 2D and (0,0,0) for 3D.
#     It can sort the patches and return all or a subset of them.

#     Args:
#         keys: keys of the corresponding items to be transformed.
#         patch_size: size of patches to generate slices for, 0 or None selects whole dimension
#         min_offset: the minimum range of starting position to be selected randomly. Defaults to 0.
#         max_offset: the maximum range of starting position to be selected randomly.
#             Defaults to image size modulo patch size.
#         num_patches: number of patches to return. Defaults to None, which returns all the available patches.
#         overlap: the amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).
#             If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
#         sort_fn: when `num_patches` is provided, it determines if keep patches with highest values (`"max"`),
#             lowest values (`"min"`), or in their default order (`None`). Default to None.
#         threshold: a value to keep only the patches whose sum of intensities are less than the threshold.
#             Defaults to no filtering.
#         pad_mode: refer to NumpyPadMode and PytorchPadMode. If None, no padding will be applied. Defaults to
#         ``"constant"``.
#         allow_missing_keys: don't raise exception if key is missing.
#         pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.

#     Returns:
#         a list of dictionaries, each of which contains the all the original key/value with the values for `keys`
#             replaced by the patches. It also add the following new keys:

#             "patch_location": the starting location of the patch in the image,
#             "patch_size": size of the extracted patch
#             "num_patches": total number of patches in the image
#             "offset": the amount of offset for the patches in the image (starting position of the first patch)

#     """

#     backend = RandGridPatch.backend

#     def __init__(
#         self,
#         keys: KeysCollection,
#         patch_size: Sequence[int],
#         min_offset: Optional[Union[Sequence[int], int]] = None,
#         max_offset: Optional[Union[Sequence[int], int]] = None,
#         num_patches: Optional[int] = None,
#         overlap: float = 0.0,
#         sort_fn: Optional[str] = None,
#         threshold: Optional[float] = None,
#         pad_mode: str = PytorchPadMode.CONSTANT,
#         allow_missing_keys: bool = False,
#         **pad_kwargs,
#     ):
#         MapTransform.__init__(self, keys, allow_missing_keys)
#         self.patcher = RandGridPatch(
#             patch_size=patch_size,
#             min_offset=min_offset,
#             max_offset=max_offset,
#             num_patches=num_patches,
#             overlap=overlap,
#             sort_fn=sort_fn,
#             threshold=threshold,
#             pad_mode=pad_mode,
#             **pad_kwargs,
#         )

#     def set_random_state(
#         self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
#     ) -> "RandGridPatchd":
#         super().set_random_state(seed, state)
#         self.patcher.set_random_state(seed, state)
#         return self

#     def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
#         d = dict(data)
#         # All the keys share the same random noise
#         for key in self.key_iterator(d):
#             self.patcher.randomize(d[key])
#             break
#         for key in self.key_iterator(d):
#             d[key] = self.patcher(d[key], randomize=False, slide_id=data[SlideKey.SLIDE_ID])
#         return d


class TilingParams(param.Parameterized):
    """Parameters for Tiling On the Fly a WSI using RandGridPatchd and GridPatchd monai transforms"""

    tile_size: int = param.Integer(default=224, bounds=(1, None), doc="The size of the tile, Default: 224")
    tile_overlap: int = param.Number(
        default=0,
        bounds=(0.0, 1.0),
        doc="The amount of overlap of neighboring patches in each dimension (a value between 0.0 and 1.0).")
    tile_sort_fn: Optional[str] = param.String(
        default='min',
        doc="When bag_size is fixed, it determines whether to keep tiles with highest intensity values (`'max'`), "
            "lowest values (`'min'`) that assumes background is high values, or in their default order (`None`). ")
    tile_pad_mode: Optional[str] = param.String(
        default=None,
        doc="The mode of padding, refer to NumpyPadMode and PytorchPadMode. Defaults to None, for no padding.")
    intensity_threshold: float = param.Number(
        default=255.,
        doc="The intensity threshold to filter out tiles based on intensity values. Default to None.")
    background_val: int = param.Integer(
        default=255,
        doc="The intensity value of background. Default to 255.")
    rand_min_offset: int = param.Integer(
        default=0,
        bounds=(0, None),
        doc="The minimum range of sarting position to be selected randomly. This parameter is passed to RandGridPatchd."
            "the random version of RandGridPatchd used at training time. Default to 0.")
    rand_max_offset: int = param.Integer(
        default=None,
        bounds=(0, None),
        doc="The maximum range of sarting position to be selected randomly. This parameter is passed to RandGridPatchd."
            "the random version of RandGridPatchd used at training time. Default to None.")
    inf_offset: Optional[int] = param.Integer(
        default=None,
        doc="The offset to be used for inference sampling. This parameter is passed to GridPatchd. Default to None.")

    @property
    def scaled_threshold(self) -> float:
        """Returns the threshold to be used for filtering out tiles based on intensity values. We need to multiply
        the threshold by the tile size to account for the fact that the intensity is computed on the entire tile"""
        return 0.999 * 3 * self.intensity_threshold * self.tile_size * self.tile_size

    def get_tiling_transform(self, bag_size: int, stage: ModelKey,) -> Callable:
        if stage == ModelKey.TRAIN:
            return RandGridPatchd(
                keys=[SlideKey.IMAGE],
                patch_size=(self.tile_size, self.tile_size),
                min_offset=self.rand_min_offset,
                max_offset=self.rand_max_offset,
                num_patches=bag_size,
                overlap=self.tile_overlap,
                sort_fn=self.tile_sort_fn,
                threshold=self.scaled_threshold,
                pad_mode=self.tile_pad_mode,  # type: ignore
                constant_values=self.background_val,  # this arg is passed to np.pad or torch.pad
            )
        else:
            return GridPatchd(
                keys=[SlideKey.IMAGE],
                patch_size=(self.tile_size, self.tile_size),
                offset=self.inf_offset,  # type: ignore
                num_patches=bag_size,
                overlap=self.tile_overlap,
                sort_fn=self.tile_sort_fn,
                threshold=self.scaled_threshold,
                pad_mode=self.tile_pad_mode,  # type: ignore
                constant_values=self.background_val,  # this arg is passed to np.pad or torch.pad
            )

    def get_split_transform(self) -> Callable:
        """GridPatchd returns stacked tiles (bag_size, C, H, W), however we need to split them into separate
        tiles to be able to apply augmentations on each tile independently.
        """
        return SplitDimd(keys=SlideKey.IMAGE, dim=0, keepdim=False, list_output=True)

    def get_extract_coordinates_transform(self) -> Callable:
        """Extract the coordinates of the tiles returned as meta data by monai transforms to hi-ml-cpath format where
        the coordinates are represented as TileKey.TILE_LEFT, TileKey.TILE_TOP, TileKey.TILE_RIGHT, TileKey.TILE_BOTTOM.
        """
        return ExtractCoordinatesd(image_key=SlideKey.IMAGE, tile_size=self.tile_size)
