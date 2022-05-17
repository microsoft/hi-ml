#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import logging
import os
import param
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
from azureml.core import Workspace, Run

# Add hi-ml packages to sys.path so that AML can find them if we are using the runner directly from the git repo
himl_root = Path(__file__).resolve().parent.parent.parent.parent
folders_to_add = [himl_root / "hi-ml" / "src",
                  himl_root / "hi-ml-azure" / "src",
                  himl_root / "hi-ml-histopathology" / "src"]
for folder in folders_to_add:
    if folder.is_dir():
        sys.path.insert(0, str(folder))

from health_azure import AzureRunInfo, submit_to_azure_if_needed  # noqa: E402
from health_azure.datasets import create_dataset_configs  # noqa: E402
from health_azure.logging import logging_to_stdout   # noqa: E402
from health_azure.paths import is_himl_used_from_git_repo  # noqa: E402
from health_azure.utils import (get_workspace, is_local_rank_zero, merge_conda_files,  # noqa: E402
                                set_environment_variables_for_multi_node, create_argparser, parse_arguments,
                                ParserResult, apply_overrides)

from health_ml.experiment_config import ExperimentConfig  # noqa: E402
from health_ml.lightning_container import LightningContainer  # noqa: E402
from health_ml.run_ml import MLRunner  # noqa: E402
from health_ml.utils import fixed_paths  # noqa: E402
from health_ml.utils.common_utils import (check_conda_environments, get_all_environment_files,  # noqa: E402
                                          get_all_pip_requirements_files,
                                          is_linux)
from health_ml.utils.config_loader import ModelConfigLoader  # noqa: E402


# We change the current working directory before starting the actual training. However, this throws off starting
# the child training threads because sys.argv[0] is a relative path when running in AzureML. Turn that into an absolute
# path.
runner_path = Path(sys.argv[0])
sys.argv[0] = str(runner_path.resolve())

DEFAULT_DOCKER_BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04"


def initialize_rpdb() -> None:
    """
    On Linux only, import and initialize rpdb, to enable remote debugging if necessary.
    """
    # rpdb signal trapping does not work on Windows, as there is no SIGTRAP:
    if not is_linux():
        return
    import rpdb
    rpdb_port = 4444
    rpdb.handle_trap(port=rpdb_port)
    # For some reason, os.getpid() does not return the ID of what appears to be the currently running process.
    logging.info("rpdb is handling traps. To debug: identify the main runner.py process, then as root: "
                 f"kill -TRAP <process_id>; nc 127.0.0.1 {rpdb_port}")


def package_setup_and_hacks() -> None:
    """
    Set up the Python packages where needed. In particular, reduce the logging level for some of the used
    libraries, which are particularly talkative in DEBUG mode. Usually when running in DEBUG mode, we want
    diagnostics about the model building itself, but not for the underlying libraries.
    It also adds workarounds for known issues in some packages.
    """
    # Numba code generation is extremely talkative in DEBUG mode, disable that.
    logging.getLogger('numba').setLevel(logging.WARNING)
    # Matplotlib is also very talkative in DEBUG mode, filling half of the log file in a PR build.
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    # Urllib3 prints out connection information for each call to write metrics, etc
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('msrest').setLevel(logging.INFO)
    # AzureML prints too many details about logging metrics
    logging.getLogger('azureml').setLevel(logging.INFO)
    # Jupyter notebook report generation
    logging.getLogger('papermill').setLevel(logging.INFO)
    logging.getLogger('nbconvert').setLevel(logging.INFO)
    # This is working around a spurious error message thrown by MKL, see
    # https://github.com/pytorch/pytorch/issues/37377
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    # Workaround for issues with matplotlib on some X servers, see
    # https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x
    # -server-loc
    matplotlib.use('Agg')


def create_runner_parser() -> argparse.ArgumentParser:
    """
    Creates a commandline parser, that understands all necessary arguments for training a model

    :return: An instance of ArgumentParser with args from ExperimentConfig added
    """
    config = ExperimentConfig()
    parser = create_argparser(config)
    return parser


class Runner:
    """
    This class contains the high-level logic to start a training run: choose a model configuration by name,
    submit to AzureML if needed, or otherwise start the actual training and test loop.

    :param project_root: The root folder that contains all of the source code that should be executed.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.experiment_config: ExperimentConfig = ExperimentConfig()
        self.lightning_container: LightningContainer = None  # type: ignore
        # This field stores the MLRunner object that has been created in the most recent call to the run() method.
        self.ml_runner: Optional[MLRunner] = None

    def parse_and_load_model(self) -> ParserResult:
        """
        Parses the command line arguments, and creates configuration objects for the model itself, and for the
        Azure-related parameters. Sets self.experiment_config to its proper values. Returns the
        parser output from parsing the model commandline arguments.

        :return: ParserResult object containing args, overrides and settings
        """
        parser1 = create_runner_parser()
        parser1_result = parse_arguments(parser1, args=sys.argv[1:])
        experiment_config = ExperimentConfig(**parser1_result.args)

        self.experiment_config = experiment_config
        if not experiment_config.model:
            raise ValueError("Parameter 'model' needs to be set to specify which model to run.")
        model_config_loader: ModelConfigLoader = ModelConfigLoader()
        # Create the model as per the "model" commandline option. This is a LightningContainer.
        container = model_config_loader.create_model_config_from_name(model_name=experiment_config.model)

        # parse overrides and apply
        assert isinstance(container, param.Parameterized)
        parser2 = create_argparser(container)
        # For each parser, feed in the unknown settings from the previous parser. All commandline args should
        # be consumed by name, hence fail if there is something that is still unknown.
        parser2_result = parse_arguments(parser2, args=parser1_result.unknown, fail_on_unknown_args=True)
        # Apply the overrides and validate. Overrides can come from either YAML settings or the commandline.
        apply_overrides(container, parser2_result.overrides)  # type: ignore
        container.validate()

        self.lightning_container = container

        return parser2_result

    def validate(self) -> None:
        """
        Runs sanity checks on the whole experiment.
        """
        if not self.experiment_config.azureml:
            if self.lightning_container.hyperdrive:
                logging.info("You have turned on HyperDrive for parameter tuning. This can "
                             "only be run in AzureML. We switched on submitting to AzureML.")
                self.experiment_config.azureml = True
            if self.lightning_container.is_crossvalidation_enabled:
                logging.info("You have turned on cross-validation. This can "
                             "only be run in AzureML. We switched on submitting to AzureML.")
                self.experiment_config.azureml = True
            if self.experiment_config.cluster:
                logging.info("You have provided a compute cluster name, hence we switched on submitting to AzureML.")
                self.experiment_config.azureml = True

    def additional_run_tags(self, script_params: List[str]) -> Dict[str, str]:
        """
        Gets the set of tags that will be added to the AzureML run as metadata.

        :param script_params: The commandline arguments used to invoke the present script.
        """
        return {
            "commandline_args": " ".join(script_params),
            "tag": self.experiment_config.tag
        }

    def run(self) -> Tuple[LightningContainer, AzureRunInfo]:
        """
        The main entry point for training and testing models from the commandline. This chooses a model to train
        via a commandline argument, runs training or testing, and writes all required info to disk and logs.

        :return: a tuple of the LightningContainer object and an AzureRunInfo containing all information about
            the present run (whether running in AzureML or not)
        """
        # Usually, when we set logging to DEBUG, we want diagnostics about the model
        # build itself, but not the tons of debug information that AzureML submissions create.
        logging_to_stdout(logging.INFO if is_local_rank_zero() else "ERROR")
        initialize_rpdb()
        self.parse_and_load_model()
        self.validate()
        azure_run_info = self.submit_to_azureml_if_needed()
        self.run_in_situ(azure_run_info)
        return self.lightning_container, azure_run_info

    def submit_to_azureml_if_needed(self) -> AzureRunInfo:
        """
        Submit a job to AzureML, returning the resulting Run object, or exiting if we were asked to wait for
        completion and the Run did not succeed.

        :return: an AzureRunInfo object containing all of the details of the present run. If AzureML is not
            specified, the attribute 'run' will None, but the object still contains helpful information
            about datasets etc
        """

        def after_submission_hook(azure_run: Run) -> None:
            """
            A function that will be called right after job submission.
            """
            # Set the default display name to what was provided as the "tag". This will affect single runs
            # and Hyperdrive parent runs
            if self.experiment_config.tag:
                azure_run.display_name = self.experiment_config.tag

        root_folder = self.project_root
        entry_script = Path(sys.argv[0]).resolve()
        script_params = sys.argv[1:]

        # TODO: Update environment variables
        environment_variables: Dict[str, Any] = {}

        # Get default datastore from the provided workspace. Authentication can take a few seconds, hence only do
        # that if we are really submitting to AzureML.
        workspace: Optional[Workspace] = None
        if self.experiment_config.azureml:
            try:
                workspace = get_workspace()
            except ValueError:
                raise ValueError("Unable to submit the script to AzureML because no workspace configuration file "
                                 "(config.json) was found.")
        default_datastore = workspace.get_default_datastore().name if workspace is not None else ""

        local_datasets = self.lightning_container.local_datasets
        all_local_datasets = [Path(p) for p in local_datasets] if len(local_datasets) > 0 else []
        # When running in AzureML, respect the commandline flag for mounting. Outside of AML, we always mount
        # datasets to be quicker.
        use_mounting = self.experiment_config.mount_in_azureml if self.experiment_config.azureml else True
        input_datasets = \
            create_dataset_configs(all_azure_dataset_ids=self.lightning_container.azure_datasets,
                                   all_dataset_mountpoints=self.lightning_container.dataset_mountpoints,
                                   all_local_datasets=all_local_datasets,  # type: ignore
                                   datastore=default_datastore,
                                   use_mounting=use_mounting)
        if self.lightning_container.is_crossvalidation_enabled and not self.experiment_config.azureml:
            raise ValueError("Cross-validation is only supported when submitting the job to AzureML.")
        hyperdrive_config = self.lightning_container.get_hyperdrive_config()
        temp_conda: Optional[Path] = None
        try:
            if self.experiment_config.azureml:
                conda_files = get_all_environment_files(root_folder,
                                                        additional_files=self.lightning_container.additional_env_files)
                check_conda_environments(conda_files)
                # This adds all pip packages required by hi-ml and hi-ml-azure in case the code is used directly from
                # source (submodule) rather than installed as a package.
                pip_requirements_files = get_all_pip_requirements_files()

                # Merge the project-specific dependencies with the packages and write unified definition to temp file.
                if len(conda_files) > 1 or len(pip_requirements_files) > 0:
                    temp_conda = root_folder / f"temp_environment-{uuid.uuid4().hex[:8]}.yml"
                    merge_conda_files(conda_files, temp_conda, pip_files=pip_requirements_files)

                if not self.experiment_config.cluster:
                    raise ValueError("You need to specify a cluster name via '--cluster NAME' to submit "
                                     "the script to run in AzureML")
                azure_run_info = submit_to_azure_if_needed(
                    entry_script=entry_script,
                    snapshot_root_directory=root_folder,
                    script_params=script_params,
                    conda_environment_file=temp_conda or conda_files[0],
                    aml_workspace=workspace,
                    compute_cluster_name=self.experiment_config.cluster,
                    environment_variables=environment_variables,
                    default_datastore=default_datastore,
                    experiment_name=self.lightning_container.model_name,  # create_experiment_name(),
                    input_datasets=input_datasets,  # type: ignore
                    num_nodes=self.experiment_config.num_nodes,
                    wait_for_completion=self.experiment_config.wait_for_completion,
                    ignored_folders=[],
                    submit_to_azureml=self.experiment_config.azureml,
                    docker_base_image=DEFAULT_DOCKER_BASE_IMAGE,
                    docker_shm_size=self.experiment_config.docker_shm_size,
                    hyperdrive_config=hyperdrive_config,
                    create_output_folders=False,
                    after_submission=after_submission_hook,
                    tags=self.additional_run_tags(script_params)
                )
                if self.experiment_config.tag and azure_run_info.run:
                    if self.lightning_container.is_crossvalidation_enabled:
                        # This code is only reached inside Azure. Set display name again - this will now affect
                        # Hypdrive child runs (for other jobs, this has already been done after submission)
                        cv_index = self.lightning_container.crossval_index
                        full_display_name = f"{self.experiment_config.tag} {cv_index}"
                        azure_run_info.run.display_name = full_display_name

            else:
                azure_run_info = submit_to_azure_if_needed(
                    input_datasets=input_datasets,  # type: ignore
                    submit_to_azureml=False)
        finally:
            if temp_conda and temp_conda.is_file():
                temp_conda.unlink()
        # submit_to_azure_if_needed calls sys.exit after submitting to AzureML. We only reach this when running
        # the script locally or in AzureML.
        return azure_run_info

    def run_in_situ(self, azure_run_info: AzureRunInfo) -> None:
        """
        Actually run the AzureML job; this method will typically run on an Azure VM.

        :param azure_run_info: Contains all information about the present run in AzureML, in particular where the
        datasets are mounted.
        """
        # Only set the logging level now. Usually, when we set logging to DEBUG, we want diagnostics about the model
        # build itself, but not the tons of debug information that AzureML submissions create.
        # Suppress the logging from all processes but the one for GPU 0 on each node, to make log files more readable
        logging_to_stdout("INFO" if is_local_rank_zero() else "ERROR")
        package_setup_and_hacks()

        # Set environment variables for multi-node training if needed. This function will terminate early
        # if it detects that it is not in a multi-node environment.
        set_environment_variables_for_multi_node()
        self.ml_runner = MLRunner(
            experiment_config=self.experiment_config,
            container=self.lightning_container,
            project_root=self.project_root)
        self.ml_runner.setup(azure_run_info)
        self.ml_runner.run()


def run(project_root: Path) -> Tuple[LightningContainer, AzureRunInfo]:
    """
    The main entry point for training and testing models from the commandline. This chooses a model to train
    via a commandline argument, runs training or testing, and writes all required info to disk and logs.

    :param project_root: The root folder that contains all of the source code that should be executed.
    :return: If submitting to AzureML, returns the model configuration that was used for training,
    including commandline overrides applied (if any). For details on the arguments, see the constructor of Runner.
    """
    print(f"project root: {project_root}")
    runner = Runner(project_root)
    return runner.run()


def main() -> None:
    run(project_root=fixed_paths.repository_root_directory() if is_himl_used_from_git_repo() else Path.cwd())


if __name__ == '__main__':
    main()
