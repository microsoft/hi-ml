# Instructions to setup and run HistoQC on H&E slides datasets

## Clone HistoQC

```shell
git clone https://github.com/choosehappy/HistoQC
```

Then edit `HistoQC/environment.devenv.yml` and remove line 17 completely,
which contains a string like this `... set HISTOQC_DEVEL = os.environ.get('HISTOQC_DEVEL', False)`.

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

The following steps are useful when the datasets are located on the cloud. The section can be skipped when the datasets are available locally on the machine.

Get blobfuse to access slides datasets:

```shell
sudo apt-get install blobfuse
```

Create a file `blobfuse.cfg` with

```text
accountName <your_storage_account>
accountKey <redacted>
containerName <container_name>
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

Run a test run as follows (on only a small set of images to start with). The example below works with `.svs` slides, can be changed for other slide formats.

```shell
python -m histoqc /<your_storage_account>/<slides_dataset_folder>/00*/**.svs -n 8 -c <path_to_config_file> -o <target_folder>
```

Argument `slides_dataset_folder` refers to the folder where the slides are stored. Argument `-n` specifies the number of parallel processes. The argument `-o` specifies the output folder where `histoqc` results will be stored (`results.tsv`, `error.log`, and intermediate result folders). The argument `path_to_config_file` refers to the `histoqc` config file used for pre-processing the slides. The original configs in `histoqc/configs` can be used, such as `config_v2.1.ini` for H&E slides and `config_ihc.ini` for IHC slides.**

**For best results with H&E images, we commented out lines 4 and 16 in `config_v2.1.ini` as shown below (pen marking and blur detection modules respectively).

```shell
; ClassificationModule.byExampleWithFeatures:pen_markings
; BlurDetectionModule.identifyBlurryRegions
```
