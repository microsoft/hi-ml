from pathlib import Path

from health_ml.lightning_container import LightningContainer
from health_ml.model_trainer import (create_lightning_trainer, write_args_file)
from health_ml.utils.common_utils import ARGS_TXT


def test_write_args_file(tmp_path: Path):
    config = {"Container":
                  {"_min_l_rate": 0.0,
                   "_model_name": "HelloContainer",
                   "adam_betas": "(0.9, 0.999)",
                   "azure_datasets": "[]"}
              }
    expected_args_path = tmp_path / ARGS_TXT
    write_args_file(config, tmp_path)

    actual_args = expected_args_path.read_text()
    assert actual_args == str(config)


def test_create_lightning_trainer():
    container = LightningContainer()
    trainer, storing_logger = create_lightning_trainer(container)
