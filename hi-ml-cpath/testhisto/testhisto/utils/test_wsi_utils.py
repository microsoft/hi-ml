from unittest.mock import patch
import torch
import pytest
import numpy as np

from health_cpath.utils.naming import ModelKey, SlideKey
from health_cpath.utils.wsi_utils import TilingParams, image_collate
from monai.data.meta_tensor import MetaTensor
from monai.transforms import RandGridPatchd, GridPatchd
from typing import Any, Dict, List, Union
from typing import Sequence
from torch.utils.data import Dataset


class MockTiledWSIDataset(Dataset):
    def __init__(self,
                 n_tiles: int,
                 n_slides: int,
                 n_classes: int,
                 tile_size: Sequence[int],
                 random_n_tiles: bool,
                 img_type: str = "np") -> None:

        self.n_tiles = n_tiles
        self.n_slides = n_slides
        self.tile_size = tile_size
        self.n_classes = n_classes
        self.random_n_tiles = random_n_tiles
        self.slide_ids = torch.arange(self.n_slides)
        self.img_type = img_type

    def __len__(self) -> int:
        return self.n_slides

    def __getitem__(self, index: int) -> List[Dict[SlideKey, Any]]:
        tile_count = np.random.randint(low=1, high=self.n_tiles) if self.random_n_tiles else self.n_tiles
        label = np.random.choice(self.n_classes)
        img: Union[np.ndarray, torch.Tensor]
        if self.img_type == "np":
            img = np.random.randint(0, 255, size=(tile_count, *self.tile_size))
        elif self.img_type == "torch":
            img = torch.randint(0, 255, size=(tile_count, *self.tile_size))
        elif self.img_type == "metatensor":
            img = MetaTensor(torch.randint(0, 255, size=(tile_count, *self.tile_size)))
        return [{SlideKey.SLIDE_ID: self.slide_ids[index],
                 SlideKey.IMAGE: img,
                 SlideKey.IMAGE_PATH: f"slide_{self.slide_ids[index]}.tiff",
                 SlideKey.LABEL: label
                 } for _ in range(tile_count)
                ]


@pytest.mark.parametrize("img_type", ["np", "torch", "metatensor"])
@pytest.mark.parametrize("random_n_tiles", [False, True])
def test_image_collate(random_n_tiles: bool, img_type: str) -> None:
    # random_n_tiles accounts for both train and inference settings where the number of tiles is fixed (during
    # training) and None during inference (validation and test)
    dataset = MockTiledWSIDataset(n_tiles=20,
                                  n_slides=10,
                                  n_classes=4,
                                  tile_size=(1, 4, 4),
                                  random_n_tiles=random_n_tiles,
                                  img_type=img_type)

    batch_size = 5
    samples_list = [dataset[idx] for idx in range(batch_size)]

    batch: dict = image_collate(samples_list)

    assert isinstance(batch, Dict)
    assert batch.keys() == samples_list[0].keys()  # type: ignore

    for key, value_list in batch.items():
        assert isinstance(value_list, List)
        assert len(value_list) == batch_size
        if key == SlideKey.IMAGE_PATH:
            assert all((value_list[idx] == samples_list[idx][key]) for idx in range(batch_size))
        else:
            assert all(torch.equal(value_list[idx], samples_list[idx][key]) for idx in range(batch_size))


@pytest.mark.parametrize("stage", [m for m in ModelKey])
def test_tiling_params(stage: ModelKey) -> None:
    params = TilingParams()
    expected_transform_type = RandGridPatchd if stage == ModelKey.TRAIN else GridPatchd
    transform = params.get_tiling_transform(stage=stage, bag_size=10)
    assert isinstance(transform, expected_transform_type)


def test_tiling_params_split_transform() -> None:
    params = TilingParams()
    with patch("health_cpath.utils.wsi_utils.SplitDimd") as mock_split_dim:
        _ = params.get_split_transform()
        mock_split_dim.assert_called_once()
        call_args = mock_split_dim.call_args_list[0][1]
        assert call_args == {'keys': SlideKey.IMAGE, 'dim': 0, 'keepdim': False, 'list_output': True}


def test_tiling_params_coordinates_transform() -> None:
    tile_size = 128
    params = TilingParams(tile_size=tile_size)
    with patch("health_cpath.utils.wsi_utils.ExtractCoordinatesd") as mock_extract_coordinates:
        _ = params.get_extract_coordinates_transform()
        mock_extract_coordinates.assert_called_once()
        call_args = mock_extract_coordinates.call_args_list[0][1]
        assert call_args == {'keys': SlideKey.IMAGE, 'tile_size': 128}
