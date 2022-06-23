#!/usr/bin/env python3
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import param
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from requests import Session
from subprocess import PIPE, Popen
from threading import Event
from typing import Any, Optional

from azureml._run_impl.run_watcher import RunWatcher
from azureml.tensorboard import Tensorboard

from health_azure import utils as azure_util
from health_azure.himl import get_workspace


ROOT_DIR = Path.cwd()
OUTPUT_DIR = ROOT_DIR / "outputs"
TENSORBOARD_DIR = ROOT_DIR / "tensorboard_logs"


class HimlTensorboardConfig(azure_util.AmlRunScriptConfig):
    log_dir: Path = param.ClassSelector(class_=Path, default=Path("outputs"), instantiate=False,
                                        doc="Path to directory in which Tensorboard  files"
                                            "(summarywriter and TB logs) are stored")
    port: int = param.Integer(default=6006, doc="The port to run Tensorboard on")


class WrappedTensorboard(Tensorboard):

    def __init__(self, remote_root: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.remote_root = remote_root

    def start(self) -> Optional[str]:
        """
        Start the Tensorboard instance, and begin processing logs.

        :return: The URL for accessing the Tensorboard instance.
        """
        self._tb_proc: Optional[Popen]
        if self._tb_proc is not None:
            return None

        self._executor = ThreadPoolExecutor()
        self._event = Event()
        self._session = Session()

        # Make a run watcher for each run we are monitoring
        self._run_watchers = []
        local_log_dirs = []
        run_local_root: str = ""
        for run in self._runs:
            run_local_root = os.path.join(self._local_root, run.id)
            local_log_dirs.append(f"{run.id}:{run_local_root}")
            run_watcher = RunWatcher(
                run,
                local_root=run_local_root,
                remote_root=self.remote_root,
                executor=self._executor,
                event=self._event,
                session=self._session)
            self._run_watchers.append(run_watcher)

        for w in self._run_watchers:
            self._executor.submit(w.refresh_requeue)  # type: ignore

        # We use sys.executable here to ensure that we can import modules from the same environment
        # as the current process.
        # (using just "python" results in the global environment, which might not have a Tensorboard module)
        # sometimes, sys.executable might not give us what we want (i.e. in a notebook), and then we just have to hope
        # that "python" will give us something useful
        python_binary = sys.executable or "python"
        python_command = [
            python_binary, "-m", "tensorboard.main",
            "--port", str(self._port)
        ]
        if len(local_log_dirs) > 1:
            # logdir_spec is not recommended but it is the only working way to display multiple dirs
            logdir_str = ','.join(local_log_dirs)
            python_command.append("--logdir_spec")
            logging.info("Loading tensorboard files for > 1 run. You may notice reduced functionality as noted "
                         "here: https://github.com/tensorflow/tensorboard#logdir--logdir_spec-legacy-mode ")
        else:
            logdir_str = run_local_root
            python_command.append("--logdir")

        python_command.append(logdir_str)

        self._tb_proc = Popen(
            python_command,
            stderr=PIPE, stdout=PIPE, universal_newlines=True)
        if os.name == "nt":
            self._win32_kill_subprocess_on_exit(self._tb_proc)

        url = self._wait_for_url()
        # in notebooks, this shows as a clickable link (whereas the returned value is not parsed in output)
        logging.info(f"Tensorboard running at: {url}")

        return url


def main() -> None:  # pragma: no cover
    tb_config = HimlTensorboardConfig()
    tb_config = azure_util.parse_args_and_update_config(tb_config, sys.argv[1:])

    config_path = tb_config.config_file

    if not config_path:
        logging.info("You have not provided a config path. Therefore we will try to find one in your "
                     "current directory, and its parents")

    workspace = get_workspace(aml_workspace=None, workspace_config_path=config_path)

    runs = azure_util._get_runs_from_script_config(tb_config, workspace)

    print(f"Runs:\n{runs}")
    if len(runs) == 0:
        raise ValueError("No runs were found")

    local_logs_dir = ROOT_DIR / tb_config.log_dir
    logging.info(f"Creating directory {local_logs_dir} to store TensorBoard logs in")
    local_logs_dir.mkdir(exist_ok=True, parents=True)

    remote_logs_dir = local_logs_dir.relative_to(ROOT_DIR)

    ts = WrappedTensorboard(remote_root=str(remote_logs_dir) + '/',
                            runs=runs,
                            local_root=str(local_logs_dir),
                            port='6006')

    ts.start()
    print("=============================================================================\n\n")
    input("Press Enter to close TensorBoard...")
    ts.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
