#! /usr/bin/env python

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import logging
import param
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from azureml.core import Workspace, Run

# Add hi-ml packages to sys.path so that AML can find them if we are using the runner directly from the git repo
himl_root = Path(__file__).resolve().parent.parent.parent.parent
folders_to_add = [himl_root / "hi-ml" / "src",
                  himl_root / "hi-ml-azure" / "src",
                  himl_root / "hi-ml-cpath" / "src"]
for folder in folders_to_add:
    if folder.is_dir():
        sys.path.insert(0, str(folder))

from health_azure import AzureRunInfo, submit_to_azure_if_needed  # noqa: E402
from health_azure.amulet import prepare_amulet_job, is_amulet_job  # noqa: E402
from health_azure.datasets import create_dataset_configs  # noqa: E402
from health_azure.himl import DEFAULT_DOCKER_BASE_IMAGE  # noqa: E402
from health_azure.logging import logging_to_stdout   # noqa: E402
from health_azure.paths import is_himl_used_from_git_repo  # noqa: E402
from health_azure.utils import (get_workspace, get_ml_client, is_local_rank_zero,  # noqa: E402
                                is_running_in_azure_ml, set_environment_variables_for_multi_node,
                                create_argparser, parse_arguments, ParserResult, apply_overrides,
                                filter_v2_input_output_args, is_global_rank_zero)

from health_ml.experiment_config import DEBUG_DDP_ENV_VAR, ExperimentConfig  # noqa: E402
from health_ml.lightning_container import LightningContainer  # noqa: E402
from health_ml.run_ml import MLRunner  # noqa: E402
from health_ml.utils import fixed_paths  # noqa: E402
from health_ml.utils.common_utils import (check_conda_environment,  # noqa: E402
                                          choose_conda_env_file, is_linux)
from health_ml.utils.config_loader import ModelConfigLoader  # noqa: E402
from health_ml.utils import health_ml_package_setup  # noqa: E402


# We change the current working directory before starting the actual training. However, this throws off starting
# the child training threads because sys.argv[0] is a relative path when running in AzureML. Turn that into an absolute
# path.
runner_path = Path(sys.argv[0])
sys.argv[0] = str(runner_path.resolve())


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
        # Filter out any args for passing inputs and outputs to scripts with AML SDK v2
        args = sys.argv[1:]
        filtered_args = filter_v2_input_output_args(args)

        parser1 = create_runner_parser()
        parser1_result = parse_arguments(parser1, args=filtered_args)
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
        if not self.experiment_config.cluster:
            if self.lightning_container.hyperdrive:
                raise ValueError("HyperDrive for hyperparameters tuning is only supported when submitting the job to "
                                 "AzureML. You need to specify a compute cluster with the argument --cluster.")
            if self.lightning_container.is_crossvalidation_enabled and not is_amulet_job():
                raise ValueError("Cross-validation is only supported when submitting the job to AzureML."
                                 "You need to specify a compute cluster with the argument --cluster.")

    def additional_run_tags(self, script_params: List[str]) -> Dict[str, str]:
        """
        Gets the set of tags that will be added to the AzureML run as metadata.

        :param script_params: The commandline arguments used to invoke the present script.
        """
        return {
            "commandline_args": " ".join(script_params),
            "tag": self.lightning_container.tag,
            **self.lightning_container.get_additional_aml_run_tags()
        }

    def additional_environment_variables(self) -> Dict[str, str]:
        return {
            DEBUG_DDP_ENV_VAR: self.experiment_config.debug_ddp.value,
            **self.lightning_container.get_additional_environment_variables()
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

        def after_submission_hook(azure_run: Run, *args: Any) -> None:
            """
            A function that will be called right after job submission. The function has a second unused argument
            to support both the required signatures for AzureML SDK v1 and v2.
            """
            # Set the default display name to what was provided as the "tag". This will affect single runs
            # and Hyperdrive parent runs
            if self.lightning_container.tag:
                azure_run.display_name = self.lightning_container.tag

        root_folder = self.project_root
        entry_script = Path(sys.argv[0]).resolve()
        script_params = sys.argv[1:]

        environment_variables = self.additional_environment_variables()

        # Get default datastore from the provided workspace. Authentication can take a few seconds, hence only do
        # that if we are really submitting to AzureML.
        workspace: Optional[Workspace] = None
        if self.experiment_config.cluster:
            try:
                workspace = get_workspace(workspace_config_path=self.experiment_config.workspace_config_path)
            except ValueError:
                raise ValueError("Unable to submit the script to AzureML because no workspace configuration file "
                                 "(config.json) was found.")

        if self.lightning_container.datastore:
            datastore = self.lightning_container.datastore
        elif workspace:
            datastore = workspace.get_default_datastore().name
        else:
            datastore = ""

        local_datasets = self.lightning_container.local_datasets
        all_local_datasets = [Path(p) for p in local_datasets] if len(local_datasets) > 0 else []
        # When running in AzureML, respect the commandline flag for mounting. Outside of AML, we always mount
        # datasets to be quicker.
        use_mounting = self.experiment_config.mount_in_azureml if self.experiment_config.cluster else True
        input_datasets = \
            create_dataset_configs(all_azure_dataset_ids=self.lightning_container.azure_datasets,
                                   all_dataset_mountpoints=self.lightning_container.dataset_mountpoints,
                                   all_local_datasets=all_local_datasets,  # type: ignore
                                   datastore=datastore,
                                   use_mounting=use_mounting)

        if self.experiment_config.cluster and not is_running_in_azure_ml():
            if self.experiment_config.strictly_aml_v1:
                hyperdrive_config = self.lightning_container.get_hyperdrive_config()
                hyperparam_args = None
            else:
                hyperparam_args = self.lightning_container.get_hyperparam_args()
                hyperdrive_config = None
            ml_client = get_ml_client() if not self.experiment_config.strictly_aml_v1 else None

            env_file = choose_conda_env_file(env_file=self.experiment_config.conda_env)
            logging.info(f"Using this Conda environment definition: {env_file}")
            check_conda_environment(env_file)

            azure_run_info = submit_to_azure_if_needed(
                entry_script=entry_script,
                snapshot_root_directory=root_folder,
                script_params=script_params,
                conda_environment_file=env_file,
                aml_workspace=workspace,
                ml_client=ml_client,
                compute_cluster_name=self.experiment_config.cluster,
                environment_variables=environment_variables,
                default_datastore=datastore,
                experiment_name=self.lightning_container.effective_experiment_name,
                input_datasets=input_datasets,  # type: ignore
                num_nodes=self.experiment_config.num_nodes,
                wait_for_completion=self.experiment_config.wait_for_completion,
                max_run_duration=self.experiment_config.max_run_duration,
                ignored_folders=[],
                submit_to_azureml=bool(self.experiment_config.cluster),
                docker_base_image=DEFAULT_DOCKER_BASE_IMAGE,
                docker_shm_size=self.experiment_config.docker_shm_size,
                hyperdrive_config=hyperdrive_config,
                hyperparam_args=hyperparam_args,
                after_submission=after_submission_hook,
                tags=self.additional_run_tags(script_params),
                strictly_aml_v1=self.experiment_config.strictly_aml_v1,
            )
        else:
            azure_run_info = submit_to_azure_if_needed(
                input_datasets=input_datasets,  # type: ignore
                submit_to_azureml=False,
                environment_variables=environment_variables,
                strictly_aml_v1=self.experiment_config.strictly_aml_v1,
                default_datastore=datastore,
            )
        if azure_run_info.run:
            # This code is only reached inside Azure. Set display name again - this will now affect
            # Hypdrive child runs (for other jobs, this has already been done after submission)
            suffix = None
            if self.lightning_container.is_crossvalidation_enabled:
                suffix = f"crossval {self.lightning_container.crossval_index}"
            elif self.lightning_container.different_seeds > 0:
                suffix = f"seed {self.lightning_container.random_seed}"
            if suffix:
                current_name = self.lightning_container.tag or azure_run_info.run.display_name
                azure_run_info.run.display_name = f"{current_name} {suffix}"
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
        health_ml_package_setup()
        prepare_amulet_job()

        # Add tags and arguments to Amulet runs
        if is_amulet_job():
            assert azure_run_info.run is not None
            azure_run_info.run.set_tags(self.additional_run_tags(sys.argv[1:]))

        # Set environment variables for multi-node training if needed. This function will terminate early
        # if it detects that it is not in a multi-node environment.
        if self.experiment_config.num_nodes > 1:
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
    if is_global_rank_zero():
        print(f"project root: {project_root}")
    runner = Runner(project_root)
    return runner.run()


def main() -> None:
    run(project_root=fixed_paths.repository_root_directory() if is_himl_used_from_git_repo() else Path.cwd())


if __name__ == '__main__':
    main()
