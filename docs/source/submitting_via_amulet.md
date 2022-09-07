# Submitting jobs to Singularity using Amulet
PLEASE NOTE: Amulet is only intended for those with access to internal Microsoft compute resources. To access Amulet you
must have an identity associated with the Microsoft tenant. This means you must be a Microsoft employee or approved
external user.

## Install Amulet
This package is not included in the hi-ml environment definition, since these instructions only apply to users
associated with the Microsoft tenant
```bash
$ pip install -U amlt --extra-index-url https://msrpypi.azurewebsites.net/stable/7e404de797f4e1eeca406c1739b00867
```

## Create an Azure ML Storage Account
As stated in the [Amulet docs](https://amulet-docs.azurewebsites.net/main/setup.html#azure-storage-account), a Storage
Account is required for storing information about your experiments, outputs of jobs etc. The Storage Account must be of
type Storage V2. See the docs for steps on setting up.

## Add your Singularity Workspace
```bash
$ amlt workspace add WORKSPACE_NAME --subscription SUBSCRIPTION_ID --resource-group RESOURCE_GROUP
$ amlt workspace set-default VC_NAME WORKSPACE_NAME
```

## (Optional) Create a project
As stated in the docs, an [Amulet project](https://amulet-docs.azurewebsites.net/main/basics/00_create_project.html)
"usually corresponds to a single research endeavor, e.g. a publication". Projects contain experiments which contain
jobs. To create a project:
```bash
$ amlt project create <your-project-name> <storage-account-name>
```
To manage existing projects, use:
```bash
$ amlt project {list|checkout|remove}
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
  - python hi-ml/src/health_ml/runner.py --model=health_cpath.TilesPandaImageNetMIL --is_finetune --batch_size=2
```

## Submit the job to Singularity
```bash
$ amlt run <path-to-config> <experiment-name> -t <target-cluster>
```

## To run a specific job from a config
If you have multiple jobs specified in your config file, it is possible to submit just one of them as follows:
```bash
$ amlt run <path-to-config> <experiment-name> :<job_name> -t <target-cluster>
```
