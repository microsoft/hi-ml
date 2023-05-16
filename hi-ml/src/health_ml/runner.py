#! /usr/bin/env python

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import contextlib
from datetime import datetime
import logging
import os
import traceback
import param
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add hi-ml packages to sys.path so that AML can find them if we are using the runner directly from the git repo
himl_root = Path(__file__).resolve().parent.parent.parent.parent
folders_to_add = [himl_root / "hi-ml" / "src", himl_root / "hi-ml-azure" / "src", himl_root / "hi-ml-cpath" / "src"]
for folder in folders_to_add:
    if folder.is_dir():
        sys.path.insert(0, str(folder))

from health_azure import AzureRunInfo, submit_to_azure_if_needed  # noqa: E402
from health_azure.argparsing import create_argparser, parse_arguments, ParserResult, apply_overrides  # noqa: E402
from health_azure.amulet import prepare_amulet_job, is_amulet_job  # noqa: E402
from health_azure.datasets import create_dataset_configs  # noqa: E402
from health_azure.himl import DEFAULT_DOCKER_BASE_IMAGE, OUTPUT_FOLDER  # noqa: E402
from health_azure.logging import logging_to_stdout  # noqa: E402
from health_azure.paths import is_himl_used_from_git_repo  # noqa: E402
from health_azure.utils import (  # noqa: E402
    ENV_LOCAL_RANK,
    ENV_NODE_RANK,
    is_local_rank_zero,
    is_running_in_azure_ml,
    set_environment_variables_for_multi_node,
    filter_v2_input_output_args,
    is_global_rank_zero,
)

from health_ml.eval_runner import EvalRunner  # noqa: E402
from health_ml.experiment_config import DEBUG_DDP_ENV_VAR, ExperimentConfig, RunnerMode  # noqa: E402
from health_ml.lightning_container import LightningContainer  # noqa: E402
from health_ml.runner_base import RunnerBase  # noqa: E402
from health_ml.training_runner import TrainingRunner  # noqa: E402
from health_ml.utils import fixed_paths  # noqa: E402
from health_ml.utils.logging import ConsoleAndFileOutput  # noqa: E402
from health_ml.utils.common_utils import check_conda_environment, choose_conda_env_file, is_linux  # noqa: E402
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
    logging.info(
        "rpdb is handling traps. To debug: identify the main runner.py process, then as root: "
        f"kill -TRAP <process_id>; nc 127.0.0.1 {rpdb_port}"
    )


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
        # This field stores the TrainingRunner object that has been created in the most recent call to the run() method.
        self.ml_runner: Optional[RunnerBase] = None

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

        from health_azure.logging import logging_stdout_handler  # noqa: E402

        if logging_stdout_handler is not None and experiment_config.log_level is not None:
            print(f"Setting custom logging level to {experiment_config.log_level}")
            logging_stdout_handler.setLevel(experiment_config.log_level.value)
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
        # Apply the commandline overrides and validate. Any parameters specified on the commandline should have
        # higher priority than what is done in the model variant
        assert isinstance(container, LightningContainer)
        container.set_model_variant(experiment_config.model_variant)
        apply_overrides(container, parser2_result.overrides)  # type: ignore
        container.validate()

        self.lightning_container = container

        return parser2_result

    def validate(self) -> None:
        """
        Runs sanity checks on the whole experiment.
        """
        if not self.experiment_config.submit_to_azure_ml:
            if self.lightning_container.hyperdrive:
                raise ValueError(
                    "HyperDrive for hyperparameters tuning is only supported when submitting the job to "
                    "AzureML. You need to specify a compute cluster with the argument --cluster."
                )
            if self.lightning_container.is_crossvalidation_parent_run and not is_amulet_job():
                raise ValueError(
                    "Cross-validation is only supported when submitting the job to AzureML."
                    "You need to specify a compute cluster with the argument --cluster."
                )

    def additional_run_tags(self, script_params: List[str]) -> Dict[str, str]:
        """
        Gets the set of tags that will be added to the AzureML run as metadata.

        :param script_params: The commandline arguments used to invoke the present script.
        """
        return {
            "commandline_args": " ".join(script_params),
            "tag": self.lightning_container.tag,
            **self.lightning_container.get_additional_aml_run_tags(),
        }

    def additional_environment_variables(self) -> Dict[str, str]:
        return {
            DEBUG_DDP_ENV_VAR: self.experiment_config.debug_ddp.value,
            **self.lightning_container.get_additional_environment_variables(),
        }

    def run(self) -> Tuple[LightningContainer, AzureRunInfo]:
        """
        The main entry point for training and testing models from the commandline. This chooses a model to train
        via a commandline argument, runs training or testing, and writes all required info to disk and logs.

        :return: a tuple of the LightningContainer object and an AzureRunInfo containing all information about
            the present run (whether running in AzureML or not)
        """
        # Suppress the logging from all processes but the one for GPU 0 on each node, to make log files more readable
        log_level = logging.INFO if is_local_rank_zero() else logging.ERROR
        logging_to_stdout(log_level)
        # When running in Azure, also output logging to a file. This can help in particular when jobs
        # get preempted, but we don't get access to the logs from the previous incarnation of the job
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
        root_folder = self.project_root
        entry_script = Path(sys.argv[0]).resolve()
        script_params = sys.argv[1:]

        environment_variables = self.additional_environment_variables()
        local_datasets = self.lightning_container.local_datasets
        all_local_datasets = [Path(p) for p in local_datasets] if len(local_datasets) > 0 else []
        # When running in AzureML, respect the commandline flag for mounting. Outside of AML, we always mount
        # datasets to be quicker.
        use_mounting = self.experiment_config.mount_in_azureml if self.experiment_config.submit_to_azure_ml else True
        input_datasets = create_dataset_configs(
            all_azure_dataset_ids=self.lightning_container.azure_datasets,
            all_dataset_mountpoints=self.lightning_container.dataset_mountpoints,
            all_local_datasets=all_local_datasets,  # type: ignore
            datastore=self.lightning_container.datastore,
            use_mounting=use_mounting,
        )

        if self.experiment_config.submit_to_azure_ml and not is_running_in_azure_ml():
            if self.experiment_config.strictly_aml_v1:
                hyperdrive_config = self.lightning_container.get_hyperdrive_config()
                hyperparam_args = None
            else:
                hyperparam_args = self.lightning_container.get_hyperparam_args()
                hyperdrive_config = None

            env_file = choose_conda_env_file(env_file=self.experiment_config.conda_env)
            logging.info(f"Using this Conda environment definition: {env_file}")
            check_conda_environment(env_file)

            azure_run_info = submit_to_azure_if_needed(
                entry_script=entry_script,
                snapshot_root_directory=root_folder,
                script_params=script_params,
                conda_environment_file=env_file,
                compute_cluster_name=self.experiment_config.cluster,
                environment_variables=environment_variables,
                experiment_name=self.lightning_container.effective_experiment_name,
                input_datasets=input_datasets,  # type: ignore
                num_nodes=self.experiment_config.num_nodes,
                wait_for_completion=self.experiment_config.wait_for_completion,
                max_run_duration=self.experiment_config.max_run_duration,
                ignored_folders=[],
                submit_to_azureml=self.experiment_config.submit_to_azure_ml,
                docker_base_image=DEFAULT_DOCKER_BASE_IMAGE,
                docker_shm_size=self.experiment_config.docker_shm_size,
                hyperdrive_config=hyperdrive_config,
                hyperparam_args=hyperparam_args,
                display_name=self.lightning_container.tag,
                tags=self.additional_run_tags(script_params),
                strictly_aml_v1=self.experiment_config.strictly_aml_v1,
                identity_based_auth=self.experiment_config.identity_based_auth,
            )
        else:
            azure_run_info = submit_to_azure_if_needed(
                input_datasets=input_datasets,  # type: ignore
                submit_to_azureml=False,
                environment_variables=environment_variables,
                strictly_aml_v1=self.experiment_config.strictly_aml_v1,
            )
        if azure_run_info.run:
            # This code is only reached inside Azure. Set display name again - this will now affect
            # Hypdrive child runs (for other jobs, this has already been done after submission)
            suffix = None
            if self.lightning_container.is_crossvalidation_child_run:
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
        health_ml_package_setup()
        prepare_amulet_job()

        # Add tags and arguments to Amulet runs
        if is_amulet_job():
            assert azure_run_info.run is not None
            azure_run_info.run.set_tags(self.additional_run_tags(sys.argv[1:]))

        if self.experiment_config.mode == RunnerMode.TRAIN:
            # Set environment variables for multi-node training if needed. This function will terminate early
            # if it detects that it is not in a multi-node environment.
            if self.experiment_config.num_nodes > 1:
                set_environment_variables_for_multi_node()
            self.ml_runner = TrainingRunner(
                experiment_config=self.experiment_config,
                container=self.lightning_container,
                project_root=self.project_root,
            )
        elif self.experiment_config.mode == RunnerMode.EVAL_FULL:
            self.ml_runner = EvalRunner(
                experiment_config=self.experiment_config,
                container=self.lightning_container,
                project_root=self.project_root,
            )
        else:
            raise ValueError(f"Unknown mode {self.experiment_config.mode}")
        self.ml_runner.validate()
        self.ml_runner.run_and_cleanup(azure_run_info)


def run(project_root: Path) -> Tuple[LightningContainer, AzureRunInfo]:
    """
    The main entry point for training and testing models from the commandline. This chooses a model to train
    via a commandline argument, runs training or testing, and writes all required info to disk and logs.

    :param project_root: The root folder that contains all of the source code that should be executed.
    :return: If submitting to AzureML, returns the model configuration that was used for training,
        including commandline overrides applied (if any). For details on the arguments, see the constructor of Runner.
    """
    if is_global_rank_zero():
        print(f"Project root: {project_root}")
    return Runner(project_root).run()


def create_logging_filename() -> Path:
    """Creates a file name for console logs, based on the current time and the DDP rank.
    The filename is timestamped to seconds level, so that we also get a full history of all
    low-priority preemptions, because each restart will create a logfile afresh.

    :return: A full path to a file for console logs.
    """
    rank = os.getenv(ENV_LOCAL_RANK, "0")
    node = os.getenv(ENV_NODE_RANK, "0")
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")
    cwd = Path.cwd()
    logging_filename = cwd
    # DDP subprocesses may already be running in the "outputs" folder. Add the "outputs" hence only if necessary.
    if cwd.name != OUTPUT_FOLDER:
        logging_filename = logging_filename / Path(OUTPUT_FOLDER)
    logging_filename = logging_filename / "console_logs" / f"logging_{timestamp}_node{node}_rank{rank}.txt"
    logging_filename.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rank {rank}: Redirecting all console logs to {logging_filename}")
    return logging_filename


def run_with_logging(project_root: Path) -> Tuple[LightningContainer, AzureRunInfo]:
    """
    Start the main main entry point for training and testing models from the commandline.
    When running in Azure, this method also redirects the stdout stream, so that all console output is visible both on
    the console and stored in a file. The filename is timestamped and contains the DDP rank of the current process.

    :param project_root: The root folder that contains all of the source code that should be executed.
    :return: If submitting to AzureML, returns the model configuration that was used for training,
        including commandline overrides applied (if any). For details on the arguments, see the constructor of Runner.
    """
    if is_running_in_azure_ml():
        logging_filename = create_logging_filename()
        with logging_filename.open("w") as logging_file:
            console_and_file = ConsoleAndFileOutput(logging_file)
            with contextlib.redirect_stdout(console_and_file):
                try:
                    return run(project_root)
                except:  # noqa
                    # Exceptions would only be printed to the console at the very top level, and would not be visible
                    # in the log file. Hence, write here specifically.
                    traceback.print_exc(file=logging_file)
                    raise
                finally:
                    logging_file.flush()
    return run(project_root)


def main() -> None:
    project_root = fixed_paths.repository_root_directory() if is_himl_used_from_git_repo() else Path.cwd()
    run_with_logging(project_root)


if __name__ == '__main__':
    main()
