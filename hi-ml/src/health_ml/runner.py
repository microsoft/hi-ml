#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
from datetime import date
import getpass
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
from azureml.core import Run

from health_ml.utils import fixed_paths
from health_azure import AzureRunInfo, submit_to_azure_if_needed
from health_azure.datasets import create_dataset_configs
from health_azure.utils import (GenericConfig, PathOrString, get_workspace, is_local_rank_zero, merge_conda_files,
                                set_environment_variables_for_multi_node,)

# from health_ml.deep_learning_config import DeepLearningConfig
from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
from health_ml.run_ml import MLRunner

from health_ml.utils.common_utils import (CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY,
                                          get_all_environment_files, is_linux, logging_to_stdout)
from health_ml.utils.config_loader import ModelConfigLoader
from health_ml.utils.generic_parsing import ParserResult, parse_args_and_add_yaml_variables, parse_arguments

# We change the current working directory before starting the actual training. However, this throws off starting
# the child training threads because sys.argv[0] is a relative path when running in AzureML. Turn that into an absolute
# path.
runner_path = Path(sys.argv[0])
if not runner_path.is_absolute():
    sys.argv[0] = str(runner_path.absolute())


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


def create_runner_parser(model_config_class: Any = None) -> argparse.ArgumentParser:
    """
    Creates a commandline parser, that understands all necessary arguments for running a script in Azure,
    plus all arguments for the given class. The class must be a subclass of GenericConfig.

    :param model_config_class: A class that contains the model-specific parameters.
    :return: An instance of ArgumentParser.
    """
    parser = ExperimentConfig.create_argparser()
    ModelConfigLoader.add_args(parser)
    if model_config_class is not None:
        if not issubclass(model_config_class, GenericConfig):
            raise ValueError(f"The given class must be a subclass of GenericConfig, but got: {model_config_class}")
        model_config_class.add_args(parser)

    return parser


def additional_run_tags(commandline_args: str) -> Dict[str, str]:
    """
    Gets the set of tags from the commandline arguments that will be added to the AzureML run as metadata

    :param commandline_args: A string that holds all commandline arguments that were used for the present run.
    """
    return {
        "commandline_args": commandline_args,
        CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY: "-1",
    }


def create_experiment_name() -> str:
    """
    Gets the name of the AzureML experiment. This is taken from the commandline, or from the git branch.

    :param azure_config: The object containing all Azure-related settings.
    :return: The name to use for the AzureML experiment.
    """
    # branch = get_git_information().branch
    # If no branch information is found anywhere, create an experiment name that is the user alias and a timestamp
    # at monthly granularity, so that not too many runs accumulate in that experiment.
    return getpass.getuser() + f"_local_branch_{date.today().strftime('%Y%m')}"


class Runner:
    """
    This class contains the high-level logic to start a training run: choose a model configuration by name,
    submit to AzureML if needed, or otherwise start the actual training and test loop.

    :param project_root: The root folder that contains all of the source code that should be executed.
    :param yaml_config_file: The path to the YAML file that contains values to supply into sys.argv.
    The function is called with the model configuration and the path to the downloaded and merged metrics files.
    :param model_deployment_hook: an optional function for deploying a model in an application-specific way.
    If present, it should take a model config, and an AzureML Model as arguments, and return an optional
    Path and a further object of any type.
    :param command_line_args: command-line arguments to use; if None, use sys.argv.
    """

    def __init__(self,
                 project_root: Path,
                 yaml_config_file: Path):
        self.project_root = project_root
        self.yaml_config_file = yaml_config_file
        # self.post_cross_validation_hook = post_cross_validation_hook
        # self.model_deployment_hook = model_deployment_hook
        # model_config and experiment_config are placeholders for now, and are set properly when command line args are
        # parsed.
        self.model_config: Optional[GenericConfig] = None
        self.experiment_config: ExperimentConfig = ExperimentConfig()
        self.lightning_container: LightningContainer = None  # type: ignore
        # This field stores the MLRunner object that has been created in the most recent call to the run() method.
        self.ml_runner: Optional[MLRunner] = None

    def parse_and_load_model(self) -> ParserResult:
        """
        Parses the command line arguments, and creates configuration objects for the model itself, and for the
        Azure-related parameters. Sets self.experiment_config and self.model_config to their proper values. Returns the
        parser output from parsing the model commandline arguments.
        If no "model" argument is provided on the commandline, self.model_config will be set to None, and the return
        value is None.

        :return: ParserResult object containing args, overrides and settings
        """
        # Create a parser that will understand only the args we need for an AzureConfig
        parser1 = create_runner_parser()
        parser_result = parse_args_and_add_yaml_variables(parser1,

                                                          fail_on_unknown_args=False)
        experiment_config = ExperimentConfig(**parser_result.args)
        self.experiment_config = experiment_config
        # self.azure_config = azure_config
        self.model_config = None
        if not experiment_config.model:
            raise ValueError("Parameter 'model' needs to be set to specify which model to run.")
        model_config_loader: ModelConfigLoader = ModelConfigLoader(**parser_result.args)
        # Create the model as per the "model" commandline option. This can return either a built-in config
        # of type DeepLearningConfig, or a LightningContainer.
        config_or_container = model_config_loader.create_model_config_from_name(model_name=experiment_config.model)

        def parse_overrides_and_apply(c: object, previous_parser_result: ParserResult) -> ParserResult:
            assert isinstance(c, GenericConfig)
            parser = type(c).create_argparser()
            # For each parser, feed in the unknown settings from the previous parser. All commandline args should
            # be consumed by name, hence fail if there is something that is still unknown.
            parser_result = parse_arguments(parser,
                                            args=previous_parser_result.unknown,
                                            fail_on_unknown_args=True)
            # Apply the overrides and validate. Overrides can come from either YAML settings or the commandline.
            # c.apply_overrides(parser_result.known_settings_from_yaml)
            c.apply_overrides(parser_result.overrides)
            c.validate()
            return parser_result

        # Now create a parser that understands overrides at model/container level.
        parser_result = parse_overrides_and_apply(config_or_container, parser_result)

        if isinstance(config_or_container, LightningContainer):
            self.lightning_container = config_or_container
        else:
            raise ValueError(f"Don't know how to handle a loaded configuration of type {type(config_or_container)}")

        return parser_result

    def run(self) -> Tuple[Optional[GenericConfig], AzureRunInfo]:
        """
        The main entry point for training and testing models from the commandline. This chooses a model to train
        via a commandline argument, runs training or testing, and writes all required info to disk and logs.

        :return: If submitting to AzureML, returns the model configuration that was used for training,
        including commandline overrides applied (if any).
        """
        # Usually, when we set logging to DEBUG, we want diagnostics about the model
        # build itself, but not the tons of debug information that AzureML submissions create.
        logging_to_stdout(logging.INFO if is_local_rank_zero() else "ERROR")
        initialize_rpdb()
        self.parse_and_load_model()
        azure_run_info = self.submit_to_azureml_if_needed()
        self.run_in_situ(azure_run_info)
        if self.model_config is None:
            return self.lightning_container, azure_run_info
        return self.model_config, azure_run_info

    def submit_to_azureml_if_needed(self) -> AzureRunInfo:
        """
        Submit a job to AzureML, returning the resulting Run object, or exiting if we were asked to wait for
        completion and the Run did not succeed.
        """
        root_folder = self.project_root
        entry_script = Path(sys.argv[0]).resolve()
        script_params = sys.argv[1:]
        conda_dependencies_files = get_all_environment_files(self.project_root)

        # TODO: Update environment variables
        environment_variables: Dict[str, Any] = {}
        # For large jobs, upload of results can time out because of large checkpoint files. Default is 600

        # get default datastore from provided workspace
        workspace = get_workspace()
        default_datastore = workspace.get_default_datastore().name

        # Reduce the size of the snapshot by adding unused folders to amlignore. The Test* subfolders are only needed
        # when running pytest.
        ignored_folders: List[PathOrString] = []

        all_local_datasets = self.lightning_container.all_local_dataset_paths()
        input_datasets = \
            create_dataset_configs(all_azure_dataset_ids=self.lightning_container.all_azure_dataset_ids(),
                                   all_dataset_mountpoints=self.lightning_container.all_dataset_mountpoints(),
                                   all_local_datasets=all_local_datasets,  # type: ignore
                                   datastore=default_datastore)

        # Create a temporary file for the merged conda file, that will be removed after submission of the job.
        temp_conda: Optional[Path] = None
        try:
            if len(conda_dependencies_files) > 1:
                temp_conda = root_folder / f"temp_environment-{uuid.uuid4().hex[:8]}.yml"
                # Merge the project-specific dependencies with the packages.
                # In case of version conflicts, the package version in the outer project is given priority.
                merge_conda_files(conda_dependencies_files, temp_conda)

            # Calls like `self.azure_config.get_workspace()` will fail if we have no AzureML credentials set up, and so
            # we should only attempt them if we intend to elevate this to AzureML
            if self.experiment_config.azureml:
                if not self.experiment_config.cluster:
                    raise ValueError("self.azure_config.cluster not set, but we need a compute_cluster_name to submit"
                                     "the script to run in AzureML")
                azure_run_info = submit_to_azure_if_needed(
                    entry_script=entry_script,
                    snapshot_root_directory=root_folder,
                    script_params=script_params,
                    conda_environment_file=temp_conda or conda_dependencies_files[0],
                    aml_workspace=workspace,
                    compute_cluster_name=self.experiment_config.cluster,
                    environment_variables=environment_variables,
                    default_datastore=default_datastore,
                    experiment_name=create_experiment_name(),
                    input_datasets=input_datasets,  # type: ignore
                    num_nodes=self.experiment_config.num_nodes,
                    wait_for_completion=False,
                    ignored_folders=ignored_folders,
                    submit_to_azureml=self.experiment_config.azureml,
                    docker_base_image=DEFAULT_DOCKER_BASE_IMAGE,
                    tags=additional_run_tags(
                        commandline_args=" ".join(script_params))
                )
                # Set the default display name to what was provided as the "tag"
                # if self.azure_config.tag:
                #     azure_run_info.run.display_name = self.azure_config.tag
            else:
                azure_run_info = submit_to_azure_if_needed(
                    input_datasets=input_datasets,  # type: ignore
                    submit_to_azureml=False)
        finally:
            if temp_conda:
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
        # logging_to_stdout(self.azure_config.log_level if is_local_rank_zero() else "ERROR")
        package_setup_and_hacks()
        # if is_global_rank_zero():
        #     self.print_git_tags()

        # Set environment variables for multi-node training if needed. This function will terminate early
        # if it detects that it is not in a multi-node environment.
        set_environment_variables_for_multi_node()
        self.ml_runner = self.create_ml_runner()
        self.ml_runner.setup(azure_run_info)
        self.ml_runner.run()

    def create_ml_runner(self) -> MLRunner:
        """
        Create and return an ML runner using the attributes of this Runner object.
        """
        return MLRunner(
            # model_config=self.model_config,
            experiment_config=self.experiment_config,
            container=self.lightning_container,
            project_root=self.project_root)


def run(project_root: Path,
        yaml_config_file: Path,
        # model_deployment_hook: Optional[ModelDeploymentHookSignature] = None
        ) -> \
        Tuple[Optional[GenericConfig], AzureRunInfo]:
    """
    The main entry point for training and testing models from the commandline. This chooses a model to train
    via a commandline argument, runs training or testing, and writes all required info to disk and logs.

    :return: If submitting to AzureML, returns the model configuration that was used for training,
    including commandline overrides applied (if any). For details on the arguments, see the constructor of Runner.
    """
    runner = Runner(project_root, yaml_config_file)
    return runner.run()


def main() -> None:
    run(project_root=fixed_paths.repository_root_directory(),
        yaml_config_file=fixed_paths.SETTINGS_YAML_FILE,
        )


if __name__ == '__main__':
    main()
