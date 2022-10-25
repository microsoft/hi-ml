from pathlib import Path

import pytest
from health_cpath.models.encoders import SSLEncoder
from health_cpath.utils.deepmil_utils import EncoderParams
from testhiml.utils.fixed_paths_for_tests import full_test_data_path, mock_run_id


def _test_invalid_ssl_checkpoint_encoder_params(ssl_checkpoint: str) -> None:
    with pytest.raises(ValueError, match=r"Invalid ssl_checkpoint:"):
        EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=ssl_checkpoint).validate()


def test_validate_ssl_checkpoint_encoder_params() -> None:

    _test_invalid_ssl_checkpoint_encoder_params(ssl_checkpoint="dummy/local/path/model.ckpt")
    _test_invalid_ssl_checkpoint_encoder_params(ssl_checkpoint="INV@lid%RUN*id")
    _test_invalid_ssl_checkpoint_encoder_params(ssl_checkpoint="http/dummy_url-com")

    # The following should be okay
    local_path = full_test_data_path(suffix="hello_world_checkpoint.ckpt")
    EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=str(local_path)).validate()
    run_id = mock_run_id(id=0)
    EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=run_id, ).validate()


def test_load_ssl_checkpoints_from_local_file(tmp_path: Path) -> None:
    checkpoint_filename = "hello_world_checkpoint.ckpt"
    local_checkpoint_path = full_test_data_path(suffix=checkpoint_filename)
    encoder_params = EncoderParams(ssl_checkpoint=local_checkpoint_path,
                                   encoder_type=SSLEncoder.__name__)
    ssl_checkpoint_path = encoder_params.get_ssl_checkpoint_path(tmp_path)
    assert ssl_checkpoint_path.exists()
    assert ssl_checkpoint_path == tmp_path / checkpoint_filename
    encoder = encoder_params.get_encoder(tmp_path)
    assert encoder is not None
