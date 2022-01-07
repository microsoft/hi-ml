#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeVar

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


from health_azure.utils import (ENV_GLOBAL_RANK, ENV_LOCAL_RANK, ENV_NODE_RANK, RUN_CONTEXT, is_global_rank_zero,
                                is_local_rank_zero, is_offline_run_context)

from health_ml.lightning_container import LightningContainer
from health_ml.utils import AzureMLLogger, AzureMLProgressBar, BatchTimeCallback
from health_ml.utils.common_utils import ARGS_TXT, change_working_directory
from health_ml.utils.lightning_loggers import StoringLogger

TEMP_PREFIX = "temp/"

T = TypeVar('T')


def upload_output_file_as_temp(file_path: Path, outputs_folder: Path) -> None:
    """
    Uploads a file to the AzureML run. It will get a name that is composed of a "temp/" prefix, plus the path
    of the file relative to the outputs folder that is used for training.

    :param file_path: The path of the file to upload.
    :param outputs_folder: The root folder that contains all training outputs.
    """
    upload_name = TEMP_PREFIX + str(file_path.relative_to(outputs_folder))
    RUN_CONTEXT.upload_file(upload_name, path_or_stream=str(file_path))


def write_args_file(config: Any, outputs_folder: Path) -> None:
    """
    Writes the given config to disk in plain text in the default output folder.
    """
    output = str(config)
    outputs_folder.mkdir(exist_ok=True, parents=True)
    dst = outputs_folder / ARGS_TXT
    dst.write_text(output)
    logging.info(output)


def create_lightning_trainer(container: LightningContainer,
                             resume_from_checkpoint: Optional[Path] = None,
                             num_nodes: int = 1,
                             **kwargs: Dict[str, Any]) -> \
        Tuple[Trainer, StoringLogger]:
    """
    Creates a Pytorch Lightning Trainer object for the given model configuration. It creates checkpoint handlers
    and loggers. That includes a diagnostic logger for use in unit tests, that is also returned as the second
    return value.

    :param container: The container with model and data.
    :param resume_from_checkpoint: If provided, training resumes from this checkpoint point.
    :param num_nodes: The number of nodes to use in distributed training.
    :param kwargs: Any additional keyowrd arguments will be passed to the constructor of Trainer.
    :return: A tuple [Trainer object, diagnostic logger]
    """
    logging.debug(f"resume_from_checkpoint: {resume_from_checkpoint}")
    num_gpus = container.num_gpus_per_node()
    effective_num_gpus = num_gpus * num_nodes
    # Accelerator should be "ddp" when running large models in AzureML (when using DDP_spawn, we get out of GPU memory).
    if effective_num_gpus > 1:
        accelerator: Optional[str] = "ddp"
        # Initialize the DDP plugin. The default for pl_find_unused_parameters is False. If True, the plugin prints out
        # lengthy warnings about the performance impact of find_unused_parameters.
        plugins = [DDPPlugin(num_nodes=num_nodes, sync_batchnorm=True,
                             find_unused_parameters=container.pl_find_unused_parameters)]
    else:
        accelerator = None
        plugins = []
    logging.info(f"Using {num_gpus} GPUs per node with accelerator '{accelerator}'")
    tensorboard_logger = TensorBoardLogger(save_dir=str(container.logs_folder), name="Lightning", version="")
    loggers = [tensorboard_logger, AzureMLLogger(False)]
    storing_logger = StoringLogger()
    loggers.append(storing_logger)
    # Use 32bit precision when running on CPU. Otherwise, make it depend on use_mixed_precision flag.
    precision = 32 if num_gpus == 0 else 16 if container.use_mixed_precision else 32
    # The next two flags control the settings in torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
    # https://pytorch.org/docs/stable/notes/randomness.html
    # For the classification models, we observed only a small performance deterioration (increase in 10sec on total
    # training time of 22min) when switching to deterministic.
    if container.pl_deterministic:
        deterministic = True
        benchmark = False
    else:
        deterministic = False
        benchmark = True

    callbacks = []
    if container.monitor_loading:
        callbacks.append(BatchTimeCallback())
    if num_gpus > 0 and container.monitor_gpu:
        logging.info("Adding monitoring for GPU utilization")
        callbacks.append(GPUStatsMonitor(intra_step_time=True, inter_step_time=True))  # type: ignore
    # Add the additional callbacks that were specified in get_trainer_arguments for LightningContainers
    if "callbacks" in kwargs:
        more_callbacks = kwargs.pop("callbacks")
        if isinstance(more_callbacks, list):
            callbacks.extend(more_callbacks)  # type: ignore
        else:
            callbacks.append(more_callbacks)  # type: ignore
    is_azureml_run = not is_offline_run_context(RUN_CONTEXT)
    progress_bar_refresh_rate = container.pl_progress_bar_refresh_rate
    if is_azureml_run:
        if progress_bar_refresh_rate is None:
            progress_bar_refresh_rate = 50
            logging.info(f"The progress bar refresh rate is not set. Using a default of {progress_bar_refresh_rate}. "
                         f"To change, modify the pl_progress_bar_refresh_rate field of the container.")
        callbacks.append(AzureMLProgressBar(refresh_rate=progress_bar_refresh_rate,  # type: ignore
                                            write_to_logging_info=True,
                                            print_timestamp=False))
    trainer = Trainer(default_root_dir=str(container.outputs_folder),
                      deterministic=deterministic,
                      benchmark=benchmark,
                      accelerator=accelerator,
                      plugins=plugins,
                      max_epochs=container.num_epochs,
                      # Both these arguments can be integers or floats. If integers, it is the number of batches.
                      # If float, it's the fraction of batches. We default to 1.0 (processing all batches).
                      limit_train_batches=container.pl_limit_train_batches or 1.0,
                      limit_val_batches=container.pl_limit_val_batches or 1.0,
                      num_sanity_val_steps=container.pl_num_sanity_val_steps,
                      callbacks=callbacks,
                      logger=loggers,
                      progress_bar_refresh_rate=progress_bar_refresh_rate,
                      num_nodes=num_nodes,
                      gpus=num_gpus,
                      precision=precision,
                      sync_batchnorm=True,
                      terminate_on_nan=container.detect_anomaly,
                      profiler=container.pl_profiler,
                      resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
                      **kwargs)
    return trainer, storing_logger


def model_train(container: LightningContainer,
                num_nodes: int = 1) -> Tuple[Trainer, StoringLogger]:
    """
    The main training loop. It creates the Pytorch model based on the configuration options passed in,
    creates a Pytorch Lightning trainer, and trains the model.
    If a checkpoint was specified, then it loads the checkpoint before resuming training.

    :param num_nodes: The number of nodes to use in distributed training.
    :param container: A container object that holds the training data in PyTorch Lightning format
    and the model to train.
    :return: A tuple of [Trainer, StoringLogger]. Trainer is the Lightning Trainer object that was used for fitting
    the model. The StoringLogger object is returned when training a built-in model, this is None when
    fitting other models.
    """
    lightning_model = container.model

    # resource_monitor: Optional[ResourceMonitor] = None
    # Execute some bookkeeping tasks only once if running distributed:
    if is_global_rank_zero():
        # logging.info(f"Model checkpoints are saved at {container.checkpoint_folder}")
        write_args_file(container,
                        outputs_folder=container.outputs_folder)
        # if container.monitoring_interval_seconds > 0:
        #     resource_monitor = start_resource_monitor(container)

    # Run all of the container-related operations consistently with changed outputs folder, even ones that
    # should not rely on the current working directory, like get_data_module.
    with change_working_directory(container.outputs_folder):
        data_module = container.get_data_module()
        if is_global_rank_zero():
            container.before_training_on_global_rank_zero()
        if is_local_rank_zero():
            container.before_training_on_local_rank_zero()
        container.before_training_on_all_ranks()

    # Create the trainer object. Backup the environment variables before doing that, in case we need to run a second
    # training in the unit tests.d
    old_environ = dict(os.environ)
    # Set random seeds just before training. For segmentation models, we have
    # something that changes the random seed in the before_training_on_rank_zero hook.
    seed_everything(container.get_effective_random_seed())
    trainer, storing_logger = create_lightning_trainer(container,
                                                       num_nodes=num_nodes,
                                                       **container.get_trainer_arguments())
    rank_info = ", ".join(f"{env}: {os.getenv(env)}"
                          for env in [ENV_GLOBAL_RANK, ENV_LOCAL_RANK, ENV_NODE_RANK])
    logging.info(f"Environment variables: {rank_info}. trainer.global_rank: {trainer.global_rank}")

    logging.info("Starting training")
    # When training models that are not built-in models, we have no guarantee that they write
    # files to the right folder. Best guess is to change the current working directory to where files should go.
    with change_working_directory(container.outputs_folder):
        trainer.fit(lightning_model, datamodule=data_module)
        trainer.logger.close()  # type: ignore
    world_size = getattr(trainer, "world_size", 0)
    is_azureml_run = not is_offline_run_context(RUN_CONTEXT)
    # Per-subject model outputs for regression models are written per rank, and need to be aggregated here.
    # Each thread per rank will come here, and upload its files to the run outputs. Rank 0 will later download them.
    if is_azureml_run and world_size > 1:
        upload_output_file_as_temp(lightning_model.train_subject_outputs_logger.csv_path,  # type: ignore
                                   container.outputs_folder)
        upload_output_file_as_temp(lightning_model.val_subject_outputs_logger.csv_path,  # type: ignore
                                   container.outputs_folder)
    # DDP will start multiple instances of the runner, one for each GPU. Those should terminate here after training.
    # We can now use the global_rank of the Lightining model, rather than environment variables, because DDP has set
    # all necessary properties.
    if lightning_model.global_rank != 0:
        logging.info(f"Terminating training thread with rank {lightning_model.global_rank}.")
        sys.exit()

    logging.info("Choosing the best checkpoint and removing redundant files.")
    # get_best_checkpoint_path(container.checkpoint_folder)
    # Lightning modifies a ton of environment variables. If we first run training and then the test suite,
    # those environment variables will mislead the training runs in the test suite, and make them crash.
    # Hence, restore the original environment after training.
    os.environ.clear()
    os.environ.update(old_environ)

    logging.info("Finished training")

    return trainer, storing_logger
