# Submitting jobs to Singularity using Amulet

PLEASE NOTE: Amulet is only intended for those with access to internal Microsoft compute resources. To access Amulet you
must have an identity associated with the Microsoft tenant. This means you must be a Microsoft employee or approved
external user.

The documentation below describes how to submit `hi-ml` jobs via Amulet. In addition, we also provide a
[minimal sample script](amulet_example.md) that shows the essential parts of a trainer script and
how it interacts with Amulet.

## Install Amulet

This package is not included in the hi-ml environment definition, since these instructions only apply to users
associated with the Microsoft tenant. See the [instructions for installing](https://amulet-docs.azurewebsites.net/main/setup.html#install-commands).

## Create an Azure ML Storage Account

As stated in the [Amulet docs](https://amulet-docs.azurewebsites.net/main/setup.html#azure-storage-account), a Storage
Account is required for storing information about your experiments, outputs of jobs etc. The Storage Account must be of
type Storage V2. See the docs for steps on setting up.

## Add your Singularity Workspace

```bash
amlt workspace add WORKSPACE_NAME --subscription SUBSCRIPTION_ID --resource-group RESOURCE_GROUP
amlt workspace set-default VC_NAME WORKSPACE_NAME
```

## Create or checkout a project

As stated in the docs, an [Amulet project](https://amulet-docs.azurewebsites.net/main/basics/00_create_project.html)
"usually corresponds to a single research endeavor, e.g. a publication". Projects contain experiments which contain
jobs. To create a project:

```bash
amlt project create <your-project-name> <storage-account-name>
```

To manage existing projects, use:

```bash
amlt project {list|checkout|remove}
```

## Create a configuration file

A configuration (yaml) file is required to specify your job. For example, to run the HelloWorld model via the hi-ml runner:

```yaml
description: Hello World on Singularity

environment:
  image: azureml/openmpi3.1.2-cuda10.2-cudnn7-ubuntu18.04:latest
  conda_yaml_file: $CONFIG_DIR/hi-ml/supercomputer_environment.yml

code:
  # local directory of the code. this will be uploaded to the server.
  # $CONFIG_DIR is expanded to the directory of this config file
  local_dir: $CONFIG_DIR

# list of jobs to run
jobs:
- name: HelloWorld
  sku: G1
  command:
  - python hi-ml/src/health_ml/runner.py --model=health_cpath.TilesPandaImageNetMIL --tune_encoder --batch_size=2
```

Note that SKU here refers to the number of GPUs/CPUs to reserve, and its memory. In this case we have specified 1 GPU.
For other options, see [the docs](https://amulet-docs.azurewebsites.net/main/config_file.html#jobs).

You can specify multiple jobs here. There are a variety of arguments for controlling factors such as `sla_tier`, whether
the job is `preemptible`, the job `priority` and more. For full details see [the docs](https://amulet-docs.azurewebsites.net/main/config_file.html#jobs)

## Submit the job to Singularity

```bash
amlt run <path-to-config> <experiment-name> -t <target-cluster>
```

## To run a specific job from a config

If you have multiple jobs specified in your config file, it is possible to submit just one of them as follows:

```bash
amlt run <path-to-config> <experiment-name> :<job_name> -t <target-cluster>
```

## Running a distributed job

There are multiple ways to distribute a job with Amulet. The recommended way is to add the following section to your config file

```yaml
env_defaults:
  NODES: 1
  GPUS: 8
  MEM: 32
```

Then you should update your job definition as follows:

```yaml
jobs:
- name: <job name>
  sku: ${NODES}x${MEM}G${GPUS}
  command:
  - python <script> <args>
  process_count_per_node: ${GPUS}
```

Additional settings and other methods for distributing can be found
[here](https://amulet-docs.azurewebsites.net/main/advanced/51_distributed.html).

For training jobs using PyTorch Lightning, the field `process_count_per_node` can be set to 0
or omitted altogether. This indicates to Amulet that the user is responsible for spawning the
additional processes. This is the case for PyTorch Lightning, which will later spawn 1 process
per GPU.

## View your job

Once your job is running, you can view it in the Azure ML UI. Alternatively, you can check on the status using the
`amlt` CLI as follows:

```bash
amlt status <exp-name> :<job-name-1>
```

Similarly, to get the STDOUT from your job, run

```bash
amlt logs <exp-name> :<job-name-1>
```

To view all of your runs in one place, run:

```bash
amlt browse
```

This will launch a Flask app which allows you to view the runs within your project

## Download the /logs outputs of your job

Amulet provides a simple way to download the logs from your job:

```bash
amlt results <exp-name> :<job-name-1>
```
