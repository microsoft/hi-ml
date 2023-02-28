#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Sequence, Union
import numpy as np
from _pytest.capture import SysCapture
import pytest
import torch
from monai.data.meta_tensor import MetaTensor
from monai.data.dataset import CacheDataset, Dataset, PersistentDataset
from monai.transforms import Compose
from monai.utils.enums import WSIPatchKeys
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from torchvision.transforms import RandomHorizontalFlip
from health_cpath.preprocessing.loading import ROIType
from health_cpath.utils.naming import SlideKey, TileKey
from health_ml.utils.bag_utils import BagDataset
from health_ml.utils.data_augmentations import HEDJitter

from health_cpath.datasets.default_paths import TCGA_CRCK_DATASET_DIR
from health_cpath.datasets.panda_tiles_dataset import PandaTilesDataset
from health_cpath.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset
from health_cpath.models.encoders import Resnet18
from health_cpath.models.transforms import (EncodeTilesBatchd, ExtractCoordinatesd, LoadTiled, TimerWrapper,
                                            LoadTilesBatchd, MetaTensorToTensord, Subsampled, transform_dict_adaptor,
                                            NormalizeBackgroundd)
from testhisto.utils.utils_testhisto import assert_dicts_equal


@pytest.mark.skipif(not os.path.isdir(TCGA_CRCK_DATASET_DIR),
                    reason="TCGA-CRCk tiles dataset is unavailable")
def test_load_tile() -> None:
    tiles_dataset = TcgaCrck_TilesDataset(TCGA_CRCK_DATASET_DIR)
    image_key = tiles_dataset.IMAGE_COLUMN
    load_transform = LoadTiled(image_key)
    index = 0

    # Test that the transform affects only the image entry in the sample
    input_sample = tiles_dataset[index]
    loaded_sample = load_transform(input_sample)
    assert_dicts_equal(loaded_sample, input_sample, exclude_keys=[image_key])

    # Test that the MONAI Dataset applies the same transform
    loaded_dataset = Dataset(tiles_dataset, transform=load_transform)  # type:ignore
    same_dataset_sample = loaded_dataset[index]
    assert_dicts_equal(same_dataset_sample, loaded_sample)

    # Test that loading another sample gives different results
    different_sample = loaded_dataset[index + 1]
    assert not torch.allclose(different_sample[image_key], loaded_sample[image_key])


@pytest.mark.skipif(not os.path.isdir(TCGA_CRCK_DATASET_DIR),
                    reason="TCGA-CRCk tiles dataset is unavailable")
def test_load_tiles_batch() -> None:
    tiles_dataset = TcgaCrck_TilesDataset(TCGA_CRCK_DATASET_DIR)
    image_key = tiles_dataset.IMAGE_COLUMN
    max_bag_size = 5
    bagged_dataset = BagDataset(tiles_dataset, bag_ids=tiles_dataset.slide_ids,  # type: ignore
                                max_bag_size=max_bag_size)
    load_batch_transform = LoadTilesBatchd(image_key)
    loaded_dataset = Dataset(tiles_dataset, transform=LoadTiled(image_key))  # type:ignore
    image_shape = loaded_dataset[0][image_key].shape
    index = 0

    # Test that the transform affects only the image entry in the batch,
    # and that the loaded images have the expected shape
    bagged_batch = bagged_dataset[index]
    manually_loaded_batch = load_batch_transform(bagged_batch)
    assert_dicts_equal(manually_loaded_batch, bagged_batch, exclude_keys=[image_key])
    assert manually_loaded_batch[image_key].shape == (max_bag_size, *image_shape)

    # Test that the MONAI Dataset applies the same transform
    loaded_bagged_dataset = Dataset(bagged_dataset, transform=load_batch_transform)  # type:ignore
    loaded_bagged_batch = loaded_bagged_dataset[index]
    assert_dicts_equal(loaded_bagged_batch, manually_loaded_batch)

    # Test that loading another batch gives different results
    different_batch = loaded_bagged_dataset[index + 1]
    assert not torch.allclose(different_batch[image_key], manually_loaded_batch[image_key])

    # Test that loading and bagging commute
    bagged_loaded_dataset = BagDataset(loaded_dataset,  # type: ignore
                                       bag_ids=tiles_dataset.slide_ids,
                                       max_bag_size=max_bag_size)
    bagged_loaded_batch = bagged_loaded_dataset[index]
    assert_dicts_equal(bagged_loaded_batch, loaded_bagged_batch)


@pytest.mark.parametrize("scale_intensity", [True, False])
def test_itensity_scaling_load_tiles_batch(scale_intensity: bool, mock_panda_tiles_root_dir: Path) -> None:
    tiles_dataset = PandaTilesDataset(mock_panda_tiles_root_dir)
    image_key = tiles_dataset.IMAGE_COLUMN
    max_bag_size = 4
    bagged_dataset = BagDataset(tiles_dataset, bag_ids=tiles_dataset.slide_ids,  # type: ignore
                                max_bag_size=max_bag_size)
    load_batch_transform = LoadTilesBatchd(image_key, scale_intensity=scale_intensity)
    index = 0

    # Test that the transform returns images in [0, 255] range
    bagged_batch = bagged_dataset[index]
    manually_loaded_batch = load_batch_transform(bagged_batch)

    pixels_dtype = torch.uint8 if not scale_intensity else torch.float32
    max_val = 255 if not scale_intensity else 1.

    for tile in manually_loaded_batch[image_key]:
        assert tile.dtype == pixels_dtype
        assert tile.max() <= max_val
        assert tile.min() >= 0
        if not scale_intensity:
            assert manually_loaded_batch[image_key][index].max() > 1
        assert tile.unique().shape[0] > 1


def _test_cache_and_persistent_datasets(tmp_path: Path,
                                        base_dataset: TorchDataset,
                                        transform: Union[Sequence[Callable], Callable],
                                        cache_subdir: str) -> None:
    default_dataset = Dataset(base_dataset, transform=transform)  # type: ignore
    cached_dataset = CacheDataset(base_dataset, transform=transform)  # type: ignore
    cache_dir = tmp_path / cache_subdir
    cache_dir.mkdir(exist_ok=True)
    persistent_dataset = PersistentDataset(base_dataset, transform=transform,  # type: ignore
                                           cache_dir=cache_dir)

    for default_sample, cached_sample, persistent_sample \
            in zip(default_dataset, cached_dataset, persistent_dataset):  # type: ignore
        assert_dicts_equal(cached_sample, default_sample)
        assert_dicts_equal(persistent_sample, default_sample)


@pytest.mark.skipif(not os.path.isdir(TCGA_CRCK_DATASET_DIR),
                    reason="TCGA-CRCk tiles dataset is unavailable")
def test_cached_loading(tmp_path: Path) -> None:
    tiles_dataset = TcgaCrck_TilesDataset(TCGA_CRCK_DATASET_DIR)
    image_key = tiles_dataset.IMAGE_COLUMN

    max_num_tiles = 100
    tiles_subset = Subset(tiles_dataset, range(max_num_tiles))
    _test_cache_and_persistent_datasets(tmp_path,
                                        tiles_subset,
                                        transform=LoadTiled(image_key),
                                        cache_subdir="TCGA-CRCk_tiles_cache")

    max_bag_size = 5
    max_num_bags = max_num_tiles // max_bag_size
    bagged_dataset = BagDataset(tiles_dataset, bag_ids=tiles_dataset.slide_ids,  # type: ignore
                                max_bag_size=max_bag_size)
    bagged_subset = Subset(bagged_dataset, range(max_num_bags))
    _test_cache_and_persistent_datasets(tmp_path,
                                        bagged_subset,
                                        transform=LoadTilesBatchd(image_key),
                                        cache_subdir="TCGA-CRCk_load_cache")


@pytest.mark.skipif(not os.path.isdir(TCGA_CRCK_DATASET_DIR),
                    reason="TCGA-CRCk tiles dataset is unavailable")
@pytest.mark.parametrize('use_gpu , chunk_size',
                         [(False, 0), (False, 2), (True, 0), (True, 2)]
                         )
def test_encode_tiles(tmp_path: Path, use_gpu: bool, chunk_size: int) -> None:
    tiles_dataset = TcgaCrck_TilesDataset(TCGA_CRCK_DATASET_DIR)
    image_key = tiles_dataset.IMAGE_COLUMN
    max_bag_size = 5
    bagged_dataset = BagDataset(tiles_dataset, bag_ids=tiles_dataset.slide_ids,  # type: ignore
                                max_bag_size=max_bag_size)

    encoder = Resnet18(tile_size=224, n_channels=3)
    if use_gpu:
        encoder.cuda()

    encode_transform = EncodeTilesBatchd(image_key, encoder, chunk_size=chunk_size)
    transform = Compose([LoadTilesBatchd(image_key), encode_transform])
    dataset = Dataset(bagged_dataset, transform=transform)  # type: ignore
    sample = dataset[0]
    assert sample[image_key].shape == (max_bag_size, encoder.num_encoding)
    # TODO: Ensure it works in DDP

    max_num_bags = 20
    bagged_subset = Subset(bagged_dataset, range(max_num_bags))
    _test_cache_and_persistent_datasets(tmp_path,
                                        bagged_subset,
                                        transform=transform,
                                        cache_subdir="TCGA-CRCk_embed_cache")


@pytest.mark.parametrize('include_non_indexable', [True, False])
@pytest.mark.parametrize('allow_missing_keys', [True, False])
def test_subsample(include_non_indexable: bool, allow_missing_keys: bool) -> None:
    batch_size = 5
    max_size = batch_size // 2
    data = {
        'array_1d': np.random.randn(batch_size),
        'array_2d': np.random.randn(batch_size, 4),
        'tensor_1d': torch.randn(batch_size),
        'tensor_2d': torch.randn(batch_size, 4),
        'list': torch.randn(batch_size).tolist(),
        'indices': list(range(batch_size)),
        'non-indexable': 42,
    }

    keys_to_subsample = list(data.keys())
    if not include_non_indexable:
        keys_to_subsample.remove('non-indexable')
    keys_to_subsample.append('missing-key')

    subsampling = Subsampled(keys_to_subsample, max_size=max_size,
                             allow_missing_keys=allow_missing_keys)
    subsampling.set_random_state(seed=0)
    if include_non_indexable:
        with pytest.raises(ValueError):
            sub_data = subsampling(data)
        return
    elif not allow_missing_keys:
        with pytest.raises(KeyError):
            sub_data = subsampling(data)
        return
    else:
        sub_data = subsampling(data)

    assert set(sub_data.keys()) == set(data.keys())

    # Check lenghts before and after subsampling
    for key in keys_to_subsample:
        if key not in data:
            continue  # Skip missing keys
        assert len(data[key]) == batch_size  # type: ignore
        assert len(sub_data[key]) == min(max_size, batch_size)  # type: ignore

    # Check contents of subsampled elements
    for key in ['tensor_1d', 'tensor_2d', 'array_1d', 'array_2d', 'list']:
        for idx, elem in zip(sub_data['indices'], sub_data[key]):
            assert np.array_equal(elem, data[key][idx])  # type: ignore

    # Check that the subsampled elements are not repeated
    for key in ['array_1d', 'array_2d', 'tensor_1d', 'tensor_2d']:
        assert sub_data[key].shape == np.unique(sub_data[key], axis=0).shape
    for key in ['list']:
        assert len(sub_data[key]) == len(set(sub_data[key]))

    # Check that subsampling is random, i.e. subsequent calls shouldn't give identical results
    sub_data2 = subsampling(data)
    for key in ['tensor_1d', 'tensor_2d', 'array_1d', 'array_2d', 'list']:
        assert not np.array_equal(sub_data[key], sub_data2[key])  # type: ignore


@pytest.mark.parametrize('max_size', [2, 5])
def test_shuffle(max_size: int) -> None:
    batch_size = 5
    data = {
        'array_1d': np.random.randn(batch_size),
        'array_2d': np.random.randn(batch_size, 4),
        'tensor_1d': torch.randn(batch_size),
        'tensor_2d': torch.randn(batch_size, 4),
        'list': torch.randn(batch_size).tolist(),
        'indices': list(range(batch_size)),
        'non-indexable': 42,
    }
    keys_to_subsample = list(data.keys())
    shuffling = Subsampled(keys_to_subsample, max_size=max_size, allow_missing_keys=True)
    shuffling.randomize(total_size=max_size)
    indices = shuffling._indices
    assert len(indices) <= len(data['indices'])         # type: ignore
    assert len(indices) == len(set(indices))


def test_transform_dict_adaptor() -> None:
    key = "key"
    transf1 = transform_dict_adaptor(RandomHorizontalFlip(p=0), key, key)
    transf2 = transform_dict_adaptor(RandomHorizontalFlip(p=1), key, key)
    transf3 = transform_dict_adaptor(HEDJitter(0), key, key)
    input_tensor = torch.arange(24).view(2, 3, 2, 2)
    input_dict = {'dummy': [], key: input_tensor}
    output_dict1 = transf1(input_dict)
    output_dict2 = transf2(input_dict)
    output_dict3 = transf3(input_dict)

    expected_output_dict2 = input_dict
    expected_output_dict2[key] = torch.flip(input_dict[key], [2])  # type: ignore

    assert output_dict1 == input_dict
    assert output_dict2 == expected_output_dict2
    assert output_dict3 == input_dict


def _get_sample(wsi_is_cropped: bool = False) -> Dict:
    torch.manual_seed(42)
    bag_size = 2
    h, w = 16, 16
    tiles = torch.randint(0, 254, (bag_size, 3, h, w))
    xs = torch.randint(0, w, (bag_size,))
    ys = torch.randint(0, h, (bag_size,))
    coords = torch.stack([ys, xs], dim=0)
    metadata = {WSIPatchKeys.LOCATION: coords, WSIPatchKeys.COUNT: bag_size}
    sample: Dict[str, Any] = {SlideKey.IMAGE: MetaTensor(tiles, meta=metadata),
                              SlideKey.LABEL: 0,
                              SlideKey.SLIDE_ID: "0"}
    if wsi_is_cropped:
        sample[SlideKey.ORIGIN] = (2, 3)
        sample[SlideKey.SCALE] = 4
    return sample


def test_extract_coordinates_from_non_metatensor() -> None:
    sample = _get_sample(wsi_is_cropped=False)
    sample[SlideKey.IMAGE] = sample[SlideKey.IMAGE].as_tensor()
    transform = ExtractCoordinatesd(tile_size=16, image_key=SlideKey.IMAGE)
    with pytest.raises(AssertionError, match="Expected MetaTensor"):
        _ = transform(sample)


@pytest.mark.parametrize('wsi_is_cropped', [True, False])
def test_extract_scale_factor(wsi_is_cropped: bool) -> None:
    sample = _get_sample(wsi_is_cropped=wsi_is_cropped)
    transform = ExtractCoordinatesd(tile_size=16, image_key=SlideKey.IMAGE)
    scale = transform.extract_scale_factor(sample)
    assert scale == (4 if wsi_is_cropped else 1)


@pytest.mark.parametrize('wsi_is_cropped', [True, False])
def test_extract_offset(wsi_is_cropped: bool) -> None:
    sample = _get_sample(wsi_is_cropped=wsi_is_cropped)
    transform = ExtractCoordinatesd(tile_size=16, image_key=SlideKey.IMAGE)
    offset = transform.extract_offset(sample)
    assert offset == ((2, 3) if wsi_is_cropped else (0, 0))


@pytest.mark.parametrize('roi_type', [r for r in ROIType])
def test_extract_coordinates_d_transform(roi_type: ROIType) -> None:
    tile_size = 16
    bag_size = 2
    wsi_is_cropped = (roi_type != ROIType.WHOLE)
    sample = _get_sample(wsi_is_cropped=wsi_is_cropped)

    transform = ExtractCoordinatesd(tile_size=tile_size, image_key=SlideKey.IMAGE)
    new_sample = transform(sample)

    offset_y, offset_x = (2, 3) if wsi_is_cropped else (0, 0)
    scale = 4 if wsi_is_cropped else 1

    # Check that the coordinates are correct
    ys, xs = sample[SlideKey.IMAGE].meta[WSIPatchKeys.LOCATION]
    assert torch.equal(new_sample[TileKey.TILE_LEFT], xs * scale + offset_x)
    assert torch.equal(new_sample[TileKey.TILE_TOP], ys * scale + offset_y)
    assert torch.equal(new_sample[TileKey.TILE_RIGHT], new_sample[TileKey.TILE_LEFT] + tile_size * scale)
    assert torch.equal(new_sample[TileKey.TILE_BOTTOM], new_sample[TileKey.TILE_TOP] + tile_size * scale)

    # Check that the image and label are tensors
    assert isinstance(new_sample[SlideKey.IMAGE], torch.Tensor)
    assert isinstance(new_sample[SlideKey.LABEL], torch.Tensor)

    # Check that the image and label are the same as the original
    assert torch.equal(new_sample[SlideKey.IMAGE], sample[SlideKey.IMAGE].as_tensor())
    assert new_sample[SlideKey.LABEL] == sample[SlideKey.LABEL]

    # Check that the tile_ids and slide_ids are set correctly
    assert TileKey.TILE_ID in new_sample
    assert SlideKey.SLIDE_ID in new_sample
    assert len(new_sample[TileKey.TILE_ID]) == bag_size
    assert len(new_sample[TileKey.SLIDE_ID]) == bag_size


def test_metatensor_to_tensor_d_transform() -> None:
    sample = _get_sample()
    transform = MetaTensorToTensord(keys=SlideKey.IMAGE)
    new_sample = transform(sample)
    assert isinstance(new_sample[SlideKey.IMAGE], torch.Tensor)
    with pytest.raises(AssertionError, match="Expected MetaTensor"):
        _ = transform(new_sample)


def test_timer_wrapper_transform(capsys: SysCapture) -> None:
    sample = {"a": 1, SlideKey.SLIDE_ID: "0"}

    class DummyTransform:
        def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
            time.sleep(0.1)
            return data

    transform = TimerWrapper(DummyTransform())
    out = transform(sample)
    assert out == sample
    message = capsys.readouterr().out  # type: ignore
    assert "Rank " in message
    assert "DummyTransform, Slide 0 took 0.10 seconds" in message


def _get_wsi_sample() -> Dict[str, Any]:
    torch.manual_seed(42)
    wsi = torch.randint(0, 254, (3, 4, 4))
    return {SlideKey.IMAGE: wsi}


def test_normalize_background_d_transform_invalid_percentile_background_keys() -> None:
    with pytest.raises(AssertionError, match=r"Either background_keys or q_percentile must be set."):
        _ = NormalizeBackgroundd(image_key=SlideKey.IMAGE, background_keys=None, q_percentile=None)


def test_normalize_background_d_transform_wrong_number_of_background_keys() -> None:
    with pytest.raises(AssertionError, match=r"Number of background keys must be 3"):
        _ = NormalizeBackgroundd(image_key=SlideKey.IMAGE, background_keys=["foo", "bar"])


def test_missing_metadata_field() -> None:
    sample = _get_wsi_sample()
    with pytest.raises(AssertionError, match=r"Background keys are expected to be in `SlideKey.METADATA`"):
        transform = NormalizeBackgroundd(image_key=SlideKey.IMAGE, background_keys=["foo", "bar", "baz"])
        _ = transform(sample)


def test_normalize_background_d_transform_missing_keys_in_data_dict() -> None:
    sample = _get_wsi_sample()
    sample[SlideKey.METADATA] = {"r": 1, "g": 2, "b": 3}
    with pytest.raises(AssertionError, match=r"Not all background keys present in data dictionary"):
        transform = NormalizeBackgroundd(image_key=SlideKey.IMAGE, background_keys=["foo", "bar", "baz"])
        _ = transform(sample)


def test_normalize_background_d_transform_no_background_keys() -> None:
    sample = _get_wsi_sample()
    transform = NormalizeBackgroundd(image_key=SlideKey.IMAGE, q_percentile=50)
    new_sample = transform(sample)
    assert new_sample[SlideKey.IMAGE].shape == sample[SlideKey.IMAGE].shape
    assert new_sample[SlideKey.IMAGE].dtype == sample[SlideKey.IMAGE].dtype
    assert new_sample[SlideKey.IMAGE].min() >= 0
    assert new_sample[SlideKey.IMAGE].max() <= 255
    assert torch.allclose(new_sample[SlideKey.IMAGE][0, 0], torch.tensor([255, 111, 246, 226], dtype=torch.uint8))
    assert torch.allclose(new_sample[SlideKey.IMAGE][1, 0], torch.tensor([64, 255, 255, 255], dtype=torch.uint8))
    assert torch.allclose(new_sample[SlideKey.IMAGE][2, 0], torch.tensor([215, 45, 255, 237], dtype=torch.uint8))


def test_normalize_background_d_transform_with_fixed_background_keys() -> None:
    sample = _get_wsi_sample()
    background_dict = {"r": 237, "g": 210, "b": 205}
    sample[SlideKey.METADATA] = background_dict
    transform = NormalizeBackgroundd(image_key=SlideKey.IMAGE, background_keys=list(background_dict.keys()))
    new_sample = transform(sample)
    assert new_sample[SlideKey.IMAGE].shape == sample[SlideKey.IMAGE].shape
    assert new_sample[SlideKey.IMAGE].dtype == sample[SlideKey.IMAGE].dtype
    assert new_sample[SlideKey.IMAGE].min() >= 0
    assert new_sample[SlideKey.IMAGE].max() <= 255
    assert torch.allclose(new_sample[SlideKey.IMAGE][0, 0], torch.tensor([182, 72, 159, 146], dtype=torch.uint8))
    assert torch.allclose(new_sample[SlideKey.IMAGE][1, 0], torch.tensor([37, 212, 255, 160], dtype=torch.uint8))
    assert torch.allclose(new_sample[SlideKey.IMAGE][2, 0], torch.tensor([135, 28, 170, 149], dtype=torch.uint8))
