# Creating Montages from Whole Slide Images

For working with large amounts of histology data, it is often useful to create montages of the data.
Montages are a collection of images that are stitched together to form a single image.
Montages are useful for visualizing large amounts of data at once, and can be used to create a single image that can be used for analysis.
The `hi-ml-cpath` toolbox contains scripts that help with the creation of montages from whole slide images (WSIs).

Creating montages can be very time-consuming. It can hence be helpful to run the process in the cloud. The montage
creation code provided here can be run in AzureML very easily.

## Types of Data for Montage Creation

1. Montages can be created from a folder of imagees, by specifying the name of the folder and a glob pattern, like
   `**/foo_*.tiff`.
1. Montages can be created by first reading a file called `dataset.csv` located in a folder. `dataset.csv` is
   effectively a Pandas DataFrame, with each row corresponding to a single image.

When working with a `dataset.csv` file, the following columns are handled:

| Column name | Contents | Required? |
|---|---|---|
| `image` | The path of the image that should be loaded | Required |
| `slide_id` | A unique identifier for the slide | Required |
| `label` | An additional string that will be placed on the montage, This could be `0`, `1`, `tumour`, ... | Optional |
| `mask` | The path of an additional image that will rendered next to the image given in `image` | Optional |

## Setup

- Check out the `hi-ml` repository via `git clone https://github.com/microsoft/hi-ml`
- Run the following commands:

```shell
cd hi-ml-cpath
make env
conda activate HimlHisto
make pip_local
```

All the commands listed below assume that
- you have activated the Conda environment
- your current working directory is `<repo root>/hi-ml-cpath`

## Creating Montages From a Folder With Files

The following command will create a montage from all files in the folder `/data` that match the glob pattern
`**/*.tiff`.

```shell
python src/health_cpath/scripts/create_montage.py --dataset /data --image_glob_pattern '**/*.tiff' --level 2 --width 1000 --output_path montage1
```

This will create a montage from all TIFF files in folder `/data`. Each TIFF file is read as a multi-level image, and
level 2 is read for creating the montage.

The `--width` argument determines the width in pixel of the output image. The height of the output image is

This will output two images, `montage1/montage.jpg` and `montage1/montage.png`.

Here's an example how this could look like for a folder with 6 images, `0.tiff` through `5.tiff`:

![image](images/montage_from_folder.png)

## Creating Montages From a `dataset.csv` File

If the montage creation script is only pointed to a folder, without providing a glob pattern,
it assumes that a file `dataset.csv` is present. A montage will be created from only the images
listed in `dataset.csv`. In addition, an optional `label` column will be added to the text that is
overlayed onto the images itself.

```shell
python src/health_cpath/scripts/create_montage.py --dataset /data --level 2 --width 1000 --output_path montage1
```
