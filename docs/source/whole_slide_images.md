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
