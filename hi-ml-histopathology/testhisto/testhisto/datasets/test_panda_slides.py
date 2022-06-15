#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from histopathology.configs.classification.DeepSMILEPanda import SlidesPandaImageNetMIL

from testhisto.mocks.slides_generator import MockPandaSlidesGenerator


def test_panda_reproducibility(tmp_path: Path) -> None:
    """Check if subsequent enumerations of the Panda dataset produce identical sequences of tiles."""
    data_generator = MockPandaSlidesGenerator(dest_data_path=tmp_path)
    data_generator.generate_mock_histo_data()

    container = SlidesPandaImageNetMIL()
    container.local_datasets = [tmp_path]
    data_module1 = container.get_data_module()
    train_loader1 = data_module1.train_dataloader()
    items1 = [next(train_loader1) for _ in range(5)]
    data_module2 = container.get_data_module()
    train_loader2 = data_module2.train_dataloader()
    items2 = [next(train_loader2) for _ in range(5)]
    assert items1 == items2
