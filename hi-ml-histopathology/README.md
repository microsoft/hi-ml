# Histopathology Models and Workflows

## Getting started

For working on the histopathology folder, please create a separate Conda environment.

```shell
cd hi-ml-histopathology
make env
```

In addition, please download an AzureML workspace configuration file for the workspace that you wish to use:

* In the browser, navigate to the workspace in question
* Click on the drop-down menu on upper right of the page, to the left of your account picture.
* Select "Download config file".
* Save that file into the `hi-ml-histopathology` folder of the git repo.

After that, all Python runs that you start inside the `hi-ml-histopathology` folder will automatically use this config file.

## Running histopathology models

To test your setup, please execute in the repository root:

```shell
python hi-ml/src/health_ml/runner.py --model histopathology.DeepSMILECrck --cluster=training-nd24
```
