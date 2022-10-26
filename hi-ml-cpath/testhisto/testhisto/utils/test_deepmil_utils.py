import pytest

from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


from health_cpath.models.encoders import SSLEncoder
from health_cpath.utils.deepmil_utils import SSL_CHECKPOINT_DIRNAME, EncoderParams
from health_ml.utils.checkpoint_utils import LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
from health_ml.utils.common_utils import DEFAULT_AML_CHECKPOINT_DIR
from testhiml.utils.fixed_paths_for_tests import full_test_data_path, mock_run_id
from testhisto.models.test_encoders import TEST_SSL_RUN_ID
from testhiml.utils_testhiml import DEFAULT_WORKSPACE


LAST_CHECKPOINT = f"{DEFAULT_AML_CHECKPOINT_DIR}/{LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX}"


def get_ssl_checkpoint_url(run_id: str) -> str:
    datastore = DEFAULT_WORKSPACE.workspace.get_default_datastore()
    account_name = datastore.account_name
    container_name = 'azureml'
    blob_name = f'ExperimentRun/dcid.{run_id}/{LAST_CHECKPOINT}'

    sas_token = generate_blob_sas(account_name=datastore.account_name,
                                  container_name=container_name,
                                  blob_name=blob_name,
                                  account_key=datastore.account_key,
                                  permission=BlobSasPermissions(read=True),
                                  expiry=datetime.utcnow() + timedelta(hours=1))

    return f'https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}'


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
    EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=run_id).validate()


def test_load_ssl_checkpoint_from_local_file(tmp_path: Path) -> None:
    checkpoint_filename = "hello_world_checkpoint.ckpt"
    local_checkpoint_path = full_test_data_path(suffix=checkpoint_filename)
    encoder_params = EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=str(local_checkpoint_path))

    ssl_checkpoint_path = encoder_params.get_ssl_checkpoint_path(tmp_path)
    assert ssl_checkpoint_path.exists()
    assert ssl_checkpoint_path == local_checkpoint_path
    with patch("health_cpath.models.encoders.SSLEncoder._get_encoder") as mock_get_encoder:
        mock_get_encoder.return_value = (MagicMock(), MagicMock())
        encoder = encoder_params.get_encoder(tmp_path)
        assert isinstance(encoder, SSLEncoder)


def test_load_ssl_checkpoint_from_url(tmp_path: Path) -> None:
    blob_url = get_ssl_checkpoint_url(TEST_SSL_RUN_ID)
    encoder_params = EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=blob_url)
    ssl_checkpoint_path = encoder_params.get_ssl_checkpoint_path(tmp_path)
    assert ssl_checkpoint_path.exists()
    assert ssl_checkpoint_path == tmp_path / SSL_CHECKPOINT_DIRNAME / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    encoder = encoder_params.get_encoder(tmp_path)
    assert isinstance(encoder, SSLEncoder)


def test_load_ssl_checkpoint_from_run_id(tmp_path: Path) -> None:
    encoder_params = EncoderParams(encoder_type=SSLEncoder.__name__, ssl_checkpoint=TEST_SSL_RUN_ID)
    ssl_checkpoint_path = encoder_params.get_ssl_checkpoint_path(tmp_path)
    assert ssl_checkpoint_path.exists()
    assert ssl_checkpoint_path == tmp_path / TEST_SSL_RUN_ID / LAST_CHECKPOINT
    encoder = encoder_params.get_encoder(tmp_path)
    assert isinstance(encoder, SSLEncoder)
