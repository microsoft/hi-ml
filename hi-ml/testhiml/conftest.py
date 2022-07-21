import logging
import pytest
import sys
from pathlib import Path


root = Path(__file__).parent.parent.parent
paths_to_add = [
    Path("hi-ml-azure") / "src",
    Path("hi-ml-azure") / "testazure",
    Path("hi-ml") / "src",
]
for folder in paths_to_add:
    full_folder = str(root / folder)
    if full_folder not in sys.path:
        print(f"Adding to sys.path for running hi-ml: {full_folder}")
        sys.path.insert(0, full_folder)

# Matplotlib is very talkative in DEBUG mode
logging.getLogger('matplotlib').setLevel(logging.INFO)

from health_azure.utils import create_aml_run_object  # noqa: E402
from testazure.utils_testazure import DEFAULT_WORKSPACE  # noqa: E402
from testhiml.utils.fixed_paths_for_tests import full_test_data_path  # noqa: E402


@pytest.fixture(scope="session")
def mock_run_id() -> str:
    """Create a mock aml run that contains a checkpoint for hello_world container.

    :return: The run id of the created run that contains the checkpoint.
    """

    experiment_name = "himl-tests"
    run_to_download_from = create_aml_run_object(experiment_name=experiment_name, workspace=DEFAULT_WORKSPACE.workspace)
    full_file_path = full_test_data_path(suffix="hello_world_checkpoint.ckpt")
    run_to_download_from.upload_file("outputs/checkpoints/last.ckpt", str(full_file_path))
    run_to_download_from.upload_file("outputs/checkpoints/best_val.ckpt", str(full_file_path))
    run_to_download_from.complete()
    return run_to_download_from.id
