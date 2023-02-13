# Instructions to setup and run HistoQC on H&E slides datasets

## Install Openslide

```shell
sudo apt-get install openslide-tools
sudo apt-get install python3-openslide
```

## Clone HistoQC

```shell
git clone https://github.com/choosehappy/HistoQC
```

Then edit `HistoQC/environment.devenv.yml` and remove line 17

Create a Conda environment called `histoqc` by running

```shell
conda env create --file environment.devenv.yml
conda activate histoqc
```

Then install the full repo as a Python package in editable mode:

```shell
pip install -e .
```

## Install Blobfuse

Get blobfuse to access slides datasets:

```shell
sudo apt-get install blobfuse
```

Create a file `blobfuse.cfg` with

```text
accountName <your_storage_account>
accontKey <redacted>
containerName datasets
```

```shell
blobfuse /<your_storage_account> \
            --tmp-path=/tmp/blobfusetmp \
            --config-file=blobfuse.cfg \
            -o attr_timeout=240 \
            -o entry_timeout=240 \
            -o negative_timeout=120 \
            -o allow_other  # important if you want other users to see that mount too
```

## Test run HistoQC

Run a test run as follows (on only a small set of images to start with).

```shell
python -m histoqc /<your_storage_account>/<slides_dataset_folder>/00*/**.svs -n 8 -c <path_to_config_file> -o <target_folder>
```

Argument `slides_dataset_folder` refes to the folder where the slides are stored. Argument `-n` specifies the number of parallel processes. Argument `path_to config_file` refers to the `histoqc` config file used for pre-processing the slides; for H&E slides, `config_v2.1_modified.ini` can be used, this is  a modified version of the original `histoqc` config `config_v2.1.ini`. The argument `-o` specifies the output folder where `histoqc` results will be stored (`results.tsv`, `error.log`, and intermediate result folders).
