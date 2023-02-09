## Install openslide

```shell
sudo apt-get install openslide-tools
sudo apt-get install python3-openslide
```

## Clone histoQC:

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

Get blobfuse to access datasets:

```shell
sudo apt-get install blobfuse
```

Create a file `blobfuse.cfg` with

```text
accountName innereye4data1
accontKey <redacted>
containerName datasets
```

```
blobfuse ~/innereye4data1 \
            --tmp-path=/tmp/blobfusetmp \
            --config-file=blobfuse.cfg \
            -o attr_timeout=240 \
            -o entry_timeout=240 \
            -o negative_timeout=120 \
            -o allow_other  # important if you want other users to see that mount too
```

Run as follows (on only a small set of images to start with)

```shell
python -m histoqc innereye4data1/TCGA-PRAD_20220712/00*/**.svs --nprocesses 8
```
