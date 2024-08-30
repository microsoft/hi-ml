# HI-ML-Azure

This folder contains the source code for PyPI package `hi-ml-azure`.

## Testing an AzureML setup

To test if your AzureML setup is correct, follow these steps to setup up Python on your local machine:

- Change the working directory to `<repo_root>/hi-ml-azure`
- Run `make conda` to install MiniConda
- Run `make env_hello_world` to build a simple Python environment with the necessary packages
- Run `conda activate hello_world` to activate the environment

Then follow these steps to test the AzureML setup:

- Download the `config.json` file from your AzureML workspace and place it in folder `<repo_root>/hi-ml-azure`
  There is a `Download config.json` button once you expand the dropdown menu on the top-right of your AzureML workspace.
  This is not in the core Azure portal, but only visible once you open `AzureML Studio` from the portal.
  The file `config.json` should look like this:

  ```json
  {
    "subscription_id": "your-subscription-id",
    "resource_group": "your-resource-group",
    "workspace_name": "your-workspace-name"
  }
  ```

- To run the test script, you must have created a compute cluster in your AzureML workspace.
  You can do this by clicking on `Compute` in the left-hand menu, selecting the "Compute clusters" tab, and
  then `+ New` to create a new compute cluster. To run the test script, it is sufficient to use a cheap CPU-only VM
  type, like `STANDARD_DS3_V2`. Give the cluster a name, and use the same name in the script below.
- Log into Azure by running `az login` in the terminal.
- Start the test script via `python hello_world.py --cluster <your_compute_cluster_name>`.
  If successful, this will print out "Successfully queued run..." at the end, and a "Run URL" that points to your job.
- Open the "Run URL" that was printed on the console in the browser to monitor the run.

## Testing access to OpenAI from an AzureML job

Requirements:

- Your compute cluster has a managed identity assigned.
- You have an OpenAI deployment URL and model name.
- The managed identity has "Cognitive Services OpenAI User" access to the OpenAI deployment.

Run the following script to test access to OpenAI from an AzureML job:

```python
python hello_world.py --cluster <your_compute_cluster_name> --openai_url <URL> --openai_model <Model>
```

If successful, this will print out the response from OpenAI.

## Testing access to datasets

Requirements:

- You have a datastore registered in your AzureML workspace
- You have a dataset registered in your AzureML workspace. For example, upload an empty file to a folder `hello_world`
  in your storage account, and register this folder as a dataset called `hello_world` in your AzureML workspace.
- You have a folder in your storage account for an output dataset. For that, upload an empty file to a folder
  `hello_world_output` and register this folder as a dataset `hello_world_output` in your AzureML workspace.

Then run the following script to test access to datasets:

```python
python hello_world.py --cluster <your_compute_cluster_name> --input_dataset hello_world --output_dataset hello_world_output
```
