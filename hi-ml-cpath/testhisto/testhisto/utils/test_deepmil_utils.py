#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from unittest.mock import MagicMock, patch

from health_cpath.models.encoders import SSLEncoder
from health_cpath.scripts.generate_ssl_checkpoint_url import get_ssl_checkpoint_url
from health_cpath.utils.deepmil_utils import SSL_CHECKPOINT_DIRNAME, EncoderParams
from health_ml.utils.checkpoint_utils import LAST_CHECKPOINT_FILE_NAME
from health_ml.utils.common_utils import DEFAULT_AML_CHECKPOINT_DIR
from testhiml.utils.fixed_paths_for_tests import full_test_data_path
from testhisto.models.test_encoders import TEST_SSL_RUN_ID
from testhiml.utils_testhiml import DEFAULT_WORKSPACE


LAST_CHECKPOINT = f"{DEFAULT_AML_CHECKPOINT_DIR}/{LAST_CHECKPOINT_FILE_NAME}"


def test_load_ssl_checkpoint_from_local_file(tmp_path: Path) -> None:
    checkpoint_filename = "hello_world_checkpoint.ckpt"
    local_checkpoint_path = full_test_data_path(suffix=checkpoint_filename)
    encoder_params = EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=str(local_checkpoint_path))

    ssl_checkpoint_path = encoder_params.ssl_checkpoint.get_path(tmp_path)
    assert ssl_checkpoint_path.exists()
    assert ssl_checkpoint_path == local_checkpoint_path
    with patch("health_cpath.models.encoders.SSLEncoder._get_encoder") as mock_get_encoder:
        mock_get_encoder.return_value = (MagicMock(), MagicMock())
        encoder = encoder_params.get_encoder(tmp_path)
        assert isinstance(encoder, SSLEncoder)


def test_load_ssl_checkpoint_from_url(tmp_path: Path) -> None:
    blob_url = get_ssl_checkpoint_url(
        run_id=TEST_SSL_RUN_ID,
        checkpoint_filename=LAST_CHECKPOINT_FILE_NAME,
        expiry_hours=1,
        aml_workspace=DEFAULT_WORKSPACE.workspace)
    encoder_params = EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=blob_url)
    ssl_checkpoint_path = encoder_params.ssl_checkpoint.get_path(tmp_path)
    assert ssl_checkpoint_path.exists()
    assert ssl_checkpoint_path == tmp_path / SSL_CHECKPOINT_DIRNAME / LAST_CHECKPOINT_FILE_NAME
    encoder = encoder_params.get_encoder(tmp_path)
    assert isinstance(encoder, SSLEncoder)


def test_load_ssl_checkpoint_from_run_id(tmp_path: Path) -> None:
    encoder_params = EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=TEST_SSL_RUN_ID)
    with patch("health_ml.utils.checkpoint_utils.get_workspace") as mock_get_workspace:
        mock_get_workspace.return_value = DEFAULT_WORKSPACE.workspace
        ssl_checkpoint_path = encoder_params.ssl_checkpoint.get_path(tmp_path)
        assert ssl_checkpoint_path.exists()
        assert ssl_checkpoint_path == tmp_path / TEST_SSL_RUN_ID / LAST_CHECKPOINT
        encoder = encoder_params.get_encoder(tmp_path)
        assert isinstance(encoder, SSLEncoder)
