# Loading Images as Torch Tensors

There are many libraries available that can load png images. Simple examples were made using most of them and time of execution was compared. The goal was to load a png file, either RGB or greyscale, into a torch.Tensor.

The tensor specification was:

* shape [3, Height, Width] (for RGB images, in RGB order) or [1, Height, Width] (for greyscale images);
* dtype float32;
* scaled to between 0.0 and 1.0.

## matplotlib

Two methods using [matplotlib](https://matplotlib.org/) were compared. The first manipulates the numpy array from the image read before creating the torch tensor, the second uses torchvision to do the transformation.

```python
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import torch
import torchvision.transforms.functional as TF

def read_image_matplotlib(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with matplotlib and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
    # numpy_array is a numpy.array of shape: (H, W), (H, W, 3), or (H, W, 4)
    # where H = height, W = width
    numpy_array = mpimg.imread(input_filename)
    if len(numpy_array.shape) == 2:
        # if loaded a greyscale image, then it is of shape (H, W) so add in an extra axis
        numpy_array = np.expand_dims(numpy_array, 2)
    # transpose to shape (C, H, W)
    numpy_array = np.transpose(numpy_array, (2, 0, 1))
    torch_tensor = torch.from_numpy(numpy_array)
    return torch_tensor


def read_image_matplotlib2(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with matplotlib and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imread.html
    # numpy_array is a numpy.array of shape: (H, W), (H, W, 3), or (H, W, 4)
    # where H = height, W = width
    numpy_array = mpimg.imread(input_filename)
    torch_tensor = TF.to_tensor(numpy_array)
    return torch_tensor
```

## OpenCV

Two methods using the Python interface to [OpenCV](https://opencv.org/) were compared. The first manipulates the numpy array from the image read before creating the torch tensor, the second uses torchvision to do the transformation. Note that OpenCV loads images in BGR format, so they need to be transformed.

```python
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

def read_image_opencv(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with OpenCV and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    # https://docs.opencv.org/4.5.3/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
    # numpy_array is a numpy.ndarray, in BGR format.
    numpy_array = cv2.imread(str(input_filename))
    numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
    is_greyscale = False not in \
        ((numpy_array[:, :, 0] == numpy_array[:, :, 1]) == (numpy_array[:, :, 1] == numpy_array[:, :, 2]))
    if is_greyscale:
        numpy_array = numpy_array[:, :, 0]
    if len(numpy_array.shape) == 2:
        # if loaded a greyscale image, then it is of shape (H, W) so add in an extra axis
        numpy_array = np.expand_dims(numpy_array, 2)
    numpy_array = np.float32(numpy_array) / 255.0
    # transpose to shape (C, H, W)
    numpy_array = np.transpose(numpy_array, (2, 0, 1))
    torch_tensor = torch.from_numpy(numpy_array)
    return torch_tensor


def read_image_opencv2(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with OpenCV and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    # https://docs.opencv.org/4.5.3/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
    # numpy_array is a numpy.ndarray, in BGR format.
    numpy_array = cv2.imread(str(input_filename))
    numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
    is_greyscale = False not in \
        ((numpy_array[:, :, 0] == numpy_array[:, :, 1]) == (numpy_array[:, :, 1] == numpy_array[:, :, 2]))
    if is_greyscale:
        numpy_array = numpy_array[:, :, 0]
    torch_tensor = TF.to_tensor(numpy_array)
    return torch_tensor
```

## Pillow

[Pillow](https://python-pillow.org/) is one of the easiest libraries to use because torchvision has a function to convert directly from Pillow images.

```python
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms.functional as TF

def read_image_pillow(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with pillow and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    pil_image = Image.open(input_filename)
    torch_tensor = TF.to_tensor(pil_image)
    return torch_tensor
```

## SciPy

[SciPy](https://scipy.org/) is also easy to use because it loads images into a numpy array of the expected shape so that it can easily be transformed into a torch tensor.

```python
from pathlib import Path

import imageio
import torch
import torchvision.transforms.functional as TF

def read_image_scipy(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with scipy and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    numpy_array = imageio.imread(input_filename)
    torch_tensor = TF.to_tensor(numpy_array)
    return torch_tensor
```

## SimpleITK

[SimpleITK](https://simpleitk.org/) requires a two step process to load an image and extract the data as a numpy array, but it is then in the correct format.

```python
from pathlib import Path

import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF

def read_image_sitk(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with SimpleITK and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    itk_image = sitk.ReadImage(str(input_filename))
    numpy_array = sitk.GetArrayFromImage(itk_image)
    torch_tensor = TF.to_tensor(numpy_array)
    return torch_tensor
```

## scikit-image

[scikit-image](https://scikit-image.org/) is also very simple to use, since it loads the image as a numpy array in the correct format.

```python
from pathlib import Path

from skimage import io
import torch
import torchvision.transforms.functional as TF

def read_image_skimage(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with scikit-image and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    numpy_array = io.imread(input_filename)
    torch_tensor = TF.to_tensor(numpy_array)
    return torch_tensor
```

## numpy

For comparison, the png image data was saved in the numpy native data format and then reloaded.

```python
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

def read_image_numpy(input_filename: Path) -> torch.Tensor:
    """
    Read an Numpy file with Torch and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    numpy_array = np.load(input_filename)
    torch_tensor = torch.from_numpy(numpy_array)
    return torch_tensor
```

## torch

Again, for comparison, the png image data was saved in the torch tensor native data format and then reloaded.

```python
from pathlib import Path

import torch

def read_image_torch2(input_filename: Path) -> torch.Tensor:
    """
    Read a Torch file with Torch and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    torch_tensor = torch.load(input_filename)
    return torch_tensor
```

## Results

All the above methods were ran against 122 small test images, repeated 10 times. So in total there were 1220 calls to each of the functions.

### RGB Images

For 61 RGB images of size 224 x 224 pixels and 61 of size 180 x 224 pixels, repeated 10 times, there are the following timings:

| Function               | Total time (s) |
|------------------------|----------------|
| read_image_matplotlib  | **9.81336**    |
| read_image_matplotlib2 | 9.96016        |
| read_image_opencv      | 12.4301        |
| read_image_opencv2     | 12.6227        |
| read_image_pillow      | 16.2288        |
| read_image_scipy       | 17.9958        |
| read_image_sitk        | 63.6669        |
| read_image_skimage     | 18.273         |
| read_image_numpy       | 7.29741        |
| read_image_torch2      | **7.07304**    |

### Greyscale Images

Similarly, with greyscale versions of the RGB images:

| Function               | Total time (s) |
|------------------------|----------------|
| read_image_matplotlib  | 8.32523        |
| read_image_matplotlib2 | **8.26399**    |
| read_image_opencv      | 11.6838        |
| read_image_opencv2     | 11.7935        |
| read_image_pillow      | 15.7406        |
| read_image_scipy       | 17.9061        |
| read_image_sitk        | 71.8732        |
| read_image_skimage     | 18.0698        |
| read_image_numpy       | 7.94197        |
| read_image_torch2      | **7.73153**    |

The recommendation therefore is to use matplotlib `mpimg.imread` to load the image and `TF.to_tensor` to transform the numpy array to a torch tensor. This is almost as fast as loading the data directly in a native numpy or torch format.

## Loading Images as Numpy Arrays

Alternatively, a numpy array may be required with an equivalent form to PIL:

* shape [Height, Width, 3] (for RGB images), in RGB order or [Height, Width] (for greyscale images);
* dtype float;
* range between 0.0 and 255.0.

### Pillow

If the image is known to be a png then a shortcut can be taken, which is quicker:

```python
from pathlib import Path

import numpy as np
from PIL import PngImagePlugin
from PIL import Image


def read_image_pillow2(input_filename: Path) -> np.array:  # type: ignore
    """
    Read an image file with pillow and return a numpy array.

    :param input_filename: Source image file path.
    :return: numpy array of shape (H, W), (H, W, 3).
    """
    with Image.open(input_filename) as pil_png:
        return np.asarray(pil_png, np.float)


def read_image_pillow3(input_filename: Path) -> np.array:  # type: ignore
    """
    Read an image file with pillow and return a numpy array.

    :param input_filename: Source image file path.
    :return: numpy array of shape (H, W), (H, W, 3).
    """
    with PngImagePlugin.PngImageFile(input_filename) as pil_png:
        return np.asarray(pil_png, np.float)
```

### SciPy

Similarly, using SciPy:

```python
from pathlib import Path

import imageio
import numpy as np


def read_image_scipy2(input_filename: Path) -> np.array:  # type: ignore
    """
    Read an image file with scipy and return a numpy array.

    :param input_filename: Source image file path.
    :return: numpy array of shape (H, W), (H, W, 3).
    """
    numpy_array = imageio.imread(input_filename).astype(np.float)
    return numpy_array
```

## Results

The three above methods were tested against the same images as above.

### RGB Images

For 61 RGB images of size 224 x 224 pixels and 61 of size 180 x 224 pixels, repeated 10 times, there are the following timings:

| Function               | Total time (s) |
|------------------------|----------------|
| read_image_pillow2     | 44.8641        |
| read_image_pillow3     | 18.1665        |
| read_image_scipy2      | 51.8801        |

### Greyscale Images

Similarly, with greyscale versions of the RGB images:

| Function               | Total time (s) |
|------------------------|----------------|
| read_image_pillow2     | 38.3468       |
| read_image_pillow3     | 14.664        |
| read_image_scipy2      | 39.6123       |
