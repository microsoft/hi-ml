import logging
import sys
from pathlib import Path
from typing import Tuple

import pytest
import param
from _pytest.main import ExitCode

# Add hi-ml packages to sys.path so that AML can find them if we are using the runner directly from the git repo
himl_root = Path(__file__).resolve().parent.parent
folders_to_add = [himl_root / "hi-ml" / "src", himl_root / "hi-ml-azure" / "src"]
for folder in folders_to_add:
    folder_str = str(folder)
    if folder.is_dir() and folder_str not in sys.path:
        sys.path.insert(0, str(folder))

from health_azure import submit_to_azure_if_needed  # noqa: E402
from health_azure.utils import WORKSPACE_CONFIG_JSON, check_config_json, create_argparser, parse_arguments  # noqa: E402
from health_ml.utils.common_utils import DEFAULT_AML_UPLOAD_DIR  # noqa: E402
from health_ml.utils.fixed_paths import repository_root_directory  # noqa: E402

PYTEST_RESULTS_FILE = "pytest_results.xml"


class RunPytestConfig(param.Parameterized):
    mark: str = param.String(default="", doc="The value to pass to pytest for the -m (mark) argument.")
    folder: str = param.String(default="", doc="The folder of tests that should be run.")
    cluster: str = param.String(default="", doc="The name of the AzureML compute cluster where the script should run.")
    conda_env: str = param.String(default="", doc="The path to the Conda environment file that should be used.")
    experiment: str = param.String(default="", doc="The name of the AzureML experiment where the run should live.")


def run_pytest(folder_to_test: str, pytest_mark: str) -> None:
    """
    Runs pytest on a given folder, restricting to the tests that have the given PyTest mark.
    If pytest finds no tests, or any of the tests fail, this function raises a ValueError. When run inside
    AzureML, this will make the job fail.

    :param pytest_mark: The PyTest mark to use for filtering out the tests to run.
    :param folder_to_test: The folder with tests that should be run.
    """
    results_file = Path(DEFAULT_AML_UPLOAD_DIR) / PYTEST_RESULTS_FILE
    pytest_args = [folder_to_test, f"--junitxml={str(results_file)}"]

    if pytest_mark:
        pytest_args += ["-m", pytest_mark]
    logging.info(f"Starting pytest with these args: {pytest_args}")
    status_code = pytest.main(pytest_args)
    if status_code == ExitCode.NO_TESTS_COLLECTED:
        raise ValueError(f"PyTest did not find any tests to run, when restricting with this mark: {pytest_mark}")
    if status_code != ExitCode.OK:
        raise ValueError(f"PyTest failed with exit code: {status_code}")


def logging_to_stdout() -> None:
    """Redirects output from the logging module to stdout."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    root.addHandler(handler)


if __name__ == "__main__":
    logging_to_stdout()

    config = RunPytestConfig()

    parser = create_argparser(config)
    parser_results = parse_arguments(parser, fail_on_unknown_args=True)
    config = RunPytestConfig(**parser_results.args)
    submit_to_azureml = config.cluster != ""
    if submit_to_azureml:
        # For runs on the github agents: Create a workspace config file from environment variables.
        # For local runs, this will fall back to a config.json file in the current folder or at repository root
        root_config_json = himl_root / WORKSPACE_CONFIG_JSON
        with check_config_json(path=Path.cwd(), shared_config_json=root_config_json):
            submit_to_azure_if_needed(
                compute_cluster_name=config.cluster,
                submit_to_azureml=submit_to_azureml,
                wait_for_completion=True,
                snapshot_root_directory=repository_root_directory(),
                conda_environment_file=config.conda_env,
                experiment_name=config.experiment,
            )
    run_pytest(folder_to_test=config.folder, pytest_mark=config.mark)
