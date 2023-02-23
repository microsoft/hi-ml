#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from torch import nn
from health_cpath.models.encoders import SSLEncoder
from health_cpath.scripts.generate_checkpoint_url import get_checkpoint_url_from_aml_run
from health_cpath.utils.deepmil_utils import EncoderParams
from health_ml.utils.checkpoint_utils import CheckpointParser, LAST_CHECKPOINT_FILE_NAME, MODEL_WEIGHTS_DIR_NAME
from health_ml.utils.common_utils import DEFAULT_AML_CHECKPOINT_DIR
from testhiml.utils.fixed_paths_for_tests import full_test_data_path
from testhisto.models.test_encoders import TEST_SSL_RUN_ID
from testhiml.utils_testhiml import DEFAULT_WORKSPACE


LAST_CHECKPOINT = f"{DEFAULT_AML_CHECKPOINT_DIR}/{LAST_CHECKPOINT_FILE_NAME}"


def test_validate_encoder_params() -> None:
    with pytest.raises(ValueError, match=r"SSLEncoder requires an ssl_checkpoint"):
        encoder = EncoderParams(encoder_type=SSLEncoder.__name__)
        encoder.validate()


def test_load_ssl_checkpoint_from_local_file(tmp_path: Path) -> None:
    checkpoint_filename = "hello_world_checkpoint.ckpt"
    local_checkpoint_path = full_test_data_path(suffix=checkpoint_filename)
    encoder_params = EncoderParams(
        encoder_type=SSLEncoder.__name__, ssl_checkpoint=CheckpointParser(str(local_checkpoint_path))
    )
    assert encoder_params.ssl_checkpoint.is_local_file
    ssl_checkpoint_path = encoder_params.ssl_checkpoint.get_or_download_checkpoint(tmp_path)
    assert ssl_checkpoint_path.exists()
    assert ssl_checkpoint_path == local_checkpoint_path
    with patch("health_cpath.models.encoders.SSLEncoder._get_encoder") as mock_get_encoder:
        mock_get_encoder.return_value = (MagicMock(), MagicMock())
        encoder = encoder_params.get_encoder(tmp_path)
        assert isinstance(encoder, SSLEncoder)


def test_load_ssl_checkpoint_from_url(tmp_path: Path) -> None:
    blob_url = get_checkpoint_url_from_aml_run(
        run_id=TEST_SSL_RUN_ID,
        checkpoint_filename=LAST_CHECKPOINT_FILE_NAME,
        expiry_days=1,
        aml_workspace=DEFAULT_WORKSPACE.workspace)
    encoder_params = EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=CheckpointParser(blob_url))
    assert encoder_params.ssl_checkpoint.is_url
    ssl_checkpoint_path = encoder_params.ssl_checkpoint.get_or_download_checkpoint(tmp_path)
    assert ssl_checkpoint_path.exists()
    assert ssl_checkpoint_path == tmp_path / MODEL_WEIGHTS_DIR_NAME / LAST_CHECKPOINT_FILE_NAME
    encoder = encoder_params.get_encoder(tmp_path)
    assert isinstance(encoder, SSLEncoder)


def test_load_ssl_checkpoint_from_run_id(tmp_path: Path) -> None:
    encoder_params = EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=CheckpointParser(TEST_SSL_RUN_ID))
    assert encoder_params.ssl_checkpoint.is_aml_run_id
    ssl_checkpoint_path = encoder_params.ssl_checkpoint.get_or_download_checkpoint(tmp_path)
    assert ssl_checkpoint_path.exists()
    assert ssl_checkpoint_path == tmp_path / TEST_SSL_RUN_ID / LAST_CHECKPOINT
    encoder = encoder_params.get_encoder(tmp_path)
    assert isinstance(encoder, SSLEncoder)


def test_projection_dim() -> None:
    num_encoding = 50
    encoder_params = EncoderParams(projection_dim=0)
    projection_layer = encoder_params.get_projection_layer(num_encoding=num_encoding)
    assert isinstance(projection_layer, nn.Identity)
    projection_dim = 20
    encoder_params = EncoderParams(projection_dim=projection_dim)
    projection_layer = encoder_params.get_projection_layer(num_encoding=num_encoding)
    assert isinstance(projection_layer, nn.Sequential)
    assert isinstance(projection_layer[0], nn.Linear)
    assert isinstance(projection_layer[1], nn.ReLU)
    assert projection_layer[0].in_features == num_encoding
    assert projection_layer[0].out_features == projection_dim
