# Commandline tools

## Run TensorBoard

From the command line, run the command

```himl-tb```

specifying one of 
`[--experiment] [--latest_run_file] [--run_recovery_ids] [--run_ids]` 

This will start a TensorBoard session, by default running on port 6006. To use an alternative port, specify this with `--port`.

If `--experiment` is provided, the most recent Run from this experiment will be visualised.
If `--latest_run_file` is provided, the script will expect to find a RunId in this file.
Alternatively you can specify the Runs to visualise via  `--run_recovery_ids` or `--run_ids`.

By default, this tool expects that your TensorBoard logs live in a folder named 'logs' and will create a similarly named folder in your root directory. If your TensorBoard logs are stored elsewhere, you can specify this with the `--log_dir` argument.

If you choose to specify `--experiment`, you can also specify `--num_runs` to view and/or `--tags` to filter by.

If your AML config path is not ROOT_DIR/config.json, you must also specify `--config_file`.

To see an example of how to create TensorBoard logs using PyTorch on AML, see the 
[AML submitting script](examples/9/aml_sample.rst) which submits the following [pytorch sample script](examples/9/pytorch_sample.rst). Note that to run this, you'll need to create an environment with pytorch and tensorboard as dependencies, as a minimum. See an [example conda environemnt](examples/9/tensorboard_env.rst). This will create an experiment named 'tensorboard_test' on your Workspace, with a single run. Go to outputs + logs -> outputs to see the tensorboard events file.
## Download files from AML Runs

From the command line, run the command 

```himl-download```

specifying one of 
`[--experiment] [--latest_run_file] [--run_recovery_ids] [--run_ids]` 

If `--experiment` is provided, the most recent Run from this experiment will be downloaded.
If `--latest_run_file` is provided, the script will expect to find a RunId in this file.
Alternatively you can specify the Run to download via  `--run_recovery_ids` or `--run_ids`.

The files associated with your Run will be downloaded to the location specified with `--output_dir` (by default ROOT_DIR/outputs)

If you choose to specify `--experiment`, you can also specify `--tags` to filter by.

If your AML config path is not `ROOT_DIR/config.json`, you must also specify `--config_file`.
