#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
import shutil
from typing import Any

from pytorch_lightning import seed_everything
from histopathology.configs.classification.DeepSMILESlidesPandaBenchmark import DeepSMILESlidesPandaBenchmark
from histopathology.utils.naming import SlideKey
import sys
sys.path.append("/home/antonsc/hi-ml/hi-ml-histopathology/testhisto/")
import torch
from testhisto.mocks.base_data_generator import MockHistoDataType

from testhisto.mocks.slides_generator import MockPandaSlidesGenerator, TilesPositioningType


def test_panda_reproducibility(tmp_path: Path) -> None:
    """Check if subsequent enumerations of the Panda dataset produce identical sequences of tiles."""
    seed_everything(seed=123)
    tile_size = 28
    num_tiles = 4
    wsi_generator = MockPandaSlidesGenerator(
        dest_data_path=tmp_path,
        mock_type=MockHistoDataType.FAKE,
        n_tiles=num_tiles,
        n_slides=10,
        n_channels=3,
        n_levels=3,
        tile_size=tile_size,
        background_val=255,
        tiles_pos_type=TilesPositioningType.RANDOM
    )
    wsi_generator.generate_mock_histo_data()

    container = DeepSMILESlidesPandaBenchmark()
    container.tile_size = tile_size
    container.max_bag_size = num_tiles
    container.local_datasets = [tmp_path]

    def get_items_from_new_loader(num_items: int) -> Any:

        data_module = container.get_data_module()
        data_module.dataloader_kwargs = {**data_module.dataloader_kwargs,
                                         "multiprocessing_context": None,
                                         "num_workers": 0}
        loader = data_module.train_dataloader()
        iterator = iter(loader)
        if num_items == 1:
            return next(iterator)
        else:
            return [next(iterator) for _ in range(num_items)]

    item1 = get_items_from_new_loader(1)
    item2 = get_items_from_new_loader(1)
    assert item1[SlideKey.SLIDE_ID] == item2[SlideKey.SLIDE_ID], "Order of slides must match"
    assert len(item1[SlideKey.IMAGE]) == len(item2[SlideKey.IMAGE]), "Length of images must match"
    for i in range(len(item1[SlideKey.IMAGE])):
        assert torch.allclose(item1[SlideKey.IMAGE][i], item2[SlideKey.IMAGE][i]), "Images don't match"



if __name__ == '__main__':
    tmp_path = Path("/tmp/mine/")
    if tmp_path.is_dir():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    test_panda_reproducibility(tmp_path)
