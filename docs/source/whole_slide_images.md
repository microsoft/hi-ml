# Whole Slide Images

Computational Pathology works with image files that can be very large in size, up to many GB. These files may be too large to load entirely into memory at once, or at least too large to act as training data. Instead they may be split into multiple tiles of a much smaller size, e.g. 224x224 pixels before being used for training. There are two popular libraries used for handling this type of image:

* [OpenSlide](https://openslide.org/)
* [cuCIM](https://github.com/rapidsai/cucim)

but they both come with trade offs and complications.

In development there is also [tifffile](https://github.com/cgohlke/tifffile/), but this is untested.

## OpenSlide

There is a Python interface for OpenSlide at [openslide-python](https://pypi.org/project/openslide-python/), but this first requires the installation of the OpenSlide library itself. This can be done on Ubuntu with:

```bash
apt-get install openslide-tools
```

On Windows follow the instructions [here](https://openslide.org/docs/windows/) and make sure that the install directory is added to the system path.

Once the shared library/dlls are installed, install the Python interface with:

```bash
pip install openslide-python
```

## cuCIM

cuCIM is much easier to install, it can be done entirely with the Python package: [cucim](https://pypi.org/project/cucim/). However, there are the following caveats:

* It requires a GPU, with NVIDIA driver 450.36+
* It requires CUDA 11.0+
* It supports only a subset of tiff image files.

The suitable AzureML base Docker images are therefore the ones containing `cuda11`, and the compute instance must contain a GPU.

## Performance

An exploratory set of scripts for comparing loading images with OpenSlide or cuCIM, and performing tiling using both libraries can be found at [`slide_image_loading`](https://github.com/microsoft/hi-ml/tree/main/hi-ml-cpath/other/slide_image_loading).

### Loading and saving at lowest resolution

Four test tiff files are used:

* a 44.5 MB file with level dimensions: ((27648, 29440), (6912, 7360), (1728, 1840))
* a 19.9 MB file with level dimensions: ((5888, 25344), (1472, 6336), (368, 1584))
* a 5.5 MB file with level dimensions: ((27648, 29440), (6912, 7360), (1728, 1840)), but acting as a mask
* a 2.1 MB file with level dimensions: ((5888, 25344), (1472, 6336), (368, 1584)), but acting as a mask

For OpenSlide the following code:

```python
    with OpenSlide(str(input_file)) as img:
        count = img.level_count
        dimensions = img.level_dimensions

        print(f"level_count: {count}")
        print(f"dimensions: {dimensions}")

        for k, v in img.properties.items():
            print(k, v)

        region = img.read_region(location=(0, 0),
                                 level=count-1,
                                 size=dimensions[count-1])
        region.save(output_file)
```

took an average of 29ms to open the file, 88ms to read the region, and 243ms to save the region as a png.

For cuCIM the following code:

```python
    img = cucim.CuImage(str(input_file))

    count = img.resolutions['level_count']
    dimensions = img.resolutions['level_dimensions']

    print(f"level_count: {count}")
    print(f"level_dimensions: {dimensions}")

    print(img.metadata)

    region = img.read_region(location=(0, 0),
                             size=dimensions[count-1],
                             level=count-1)
    np_img_arr = np.asarray(region)
    img2 = Image.fromarray(np_img_arr)
    img2.save(output_file)
```

took an average of 369ms to open the file, 7ms to read the region and 197ms to save the region as a png, but note that it failed to handle the mask images.

### Loading and saving as tiles at the medium resolution

Test code created tiles of size 224x224 pilfes, loaded the mask images, and used occupancy levels to decide which tiles to create and save from level 1 - the middle resolution. This was profiled against both images, as above.

For cuCIM the total time was 4.7s, 2.48s to retain the tiles as a Numpy stack but not save them as pngs. cuCIM has the option of cacheing images, but is actually made performance slightly worse, possibly because the natural tile sizes in the original tiffs were larger than the tile sizes.

For OpenSlide the comparable total times were 5.7s, and 3.26s.
