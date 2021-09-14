# Commandline tools

## Run TensorBoard

From the command line, run the command

```himl-tb```

specifying one of 
`[--experiment_name] [--latest_run_path] [--run_recovery_ids]` 

This will start a TensorBoard session, by default running on port 6006. To use an alternative port, specify this with `--port`.

If `--experiment_name` is provided, the most recent Run from this experiment will be visualised.
If `--latest_run_path` is provided, the script will expect to find a RunId in this file.
Alternatively you can specify the Runs to visualise via  `--run_recovery_ids` or `--run_ids`.
You can specify the location where TensorBoard logs will be stored, using the `--run_logs_dir` argument.

If you choose to specify `--experiment_name`, you can also specify `--num_runs` to view and/or `--tags` to filter by.

If your AML config path is not ROOT_DIR/config.json, you must also specify `--config_path`.

## Download files from AML Runs

From the command line, run the command 

```himl-download```

specifying one of 
`[--experiment_name] [--latest_run_path] [--run_recovery_ids] [--run_ids]` 

If `--experiment_name` is provided, the most recent Run from this experiment will be downloaded.
If `--latest_run_path` is provided, the script will expect to find a RunId in this file.
Alternatively you can specify the Runs to download via  `--run_recovery_ids` or `--run_ids`.

The files associated with your Run(s) will be downloaded to the location specified with `--output_dir` (by default ROOT_DIR/outputs)

If you choose to specify `--experiment_name`, you can also specify `--num_runs` to view and/or `--tags` to filter by.

If your AML config path is not `ROOT_DIR/config.json`, you must also specify `--config_path`.

