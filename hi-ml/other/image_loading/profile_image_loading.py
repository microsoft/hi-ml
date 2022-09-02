#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from azureml.data.file_dataset import FileDataset
import cv2
import imageio
import matplotlib.image as mpimg
import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from line_profiler import LineProfiler
from PIL import Image
from PIL import PngImagePlugin
from skimage import io
from torchvision.io.image import read_image

from health_azure import get_workspace
from health_azure.datasets import get_or_create_dataset


def crop_size(width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Given an image size, return a test box, as a tuple: (left, top, right, bottom).

    :param width: Image width.
    :param height: Image height.
    :return: Test image crop box.
    """
    left = width / 10
    top = 0
    right = 9 * width / 10
    bottom = height
    return (round(left), top, round(right), bottom)


def convert_pillow(pil_image: Image.Image, greyscale: bool, crop: bool) -> Image.Image:
    """
    Using Pillow optionally crop a vertical section out of an image, then optionally convert it to greyscale.

    :param pil_image: Source image.
    :param greyscale: Optionally convert to greyscale.
    :param crop: Optionally crop.
    :return: Optionally cropped, optionally greyscale image.
    """
    if crop:
        width, height = pil_image.size
        box = crop_size(width, height)
        pil_image = pil_image.crop(box)
    if greyscale:
        pil_image = pil_image.convert("L")
    return pil_image


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


def read_image_pillow(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with pillow and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    pil_image = Image.open(input_filename)
    torch_tensor = TF.to_tensor(pil_image)
    return torch_tensor


def read_image_pillow2(input_filename: Path) -> np.array:  # type: ignore
    """
    Read an image file with pillow and return a numpy array.

    :param input_filename: Source image file path.
    :return: numpy array of shape (H, W), (H, W, 3).
    """
    with Image.open(input_filename) as pil_png:
        return np.asarray(pil_png, np.float)  # type: ignore


def read_image_pillow3(input_filename: Path) -> np.array:  # type: ignore
    """
    Read an image file with pillow and return a numpy array.

    :param input_filename: Source image file path.
    :return: numpy array of shape (H, W), (H, W, 3).
    """
    with PngImagePlugin.PngImageFile(input_filename) as pil_png:
        return np.asarray(pil_png, np.float)  # type: ignore


def read_image_scipy(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with scipy and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    numpy_array = imageio.imread(input_filename)
    torch_tensor = TF.to_tensor(numpy_array)
    return torch_tensor


def read_image_scipy2(input_filename: Path) -> np.array:  # type: ignore
    """
    Read an image file with scipy and return a numpy array.

    :param input_filename: Source image file path.
    :return: numpy array of shape (H, W), (H, W, 3).
    """
    numpy_array = imageio.imread(input_filename).astype(np.float)  # type: ignore
    return numpy_array


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


def read_image_skimage(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with scikit-image and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    numpy_array = io.imread(input_filename)
    torch_tensor = TF.to_tensor(numpy_array)
    return torch_tensor


def read_image_torch(input_filename: Path) -> torch.Tensor:
    """
    Read an image file with Torch and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    torch_tensor = read_image(str(input_filename))
    return torch_tensor


def read_image_torch2(input_filename: Path) -> torch.Tensor:
    """
    Read a Torch file with Torch and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    torch_tensor = torch.load(input_filename)
    return torch_tensor


def write_image_torch2(tensor: torch.Tensor, output_filename: Path) -> None:
    """
    Save a torch.Tensor as native torch.Tensor.

    :param tensor: Tensor to save.
    :param output_filename: Target filename.
    :return: None.
    """
    torch.save(tensor, output_filename)


def read_image_numpy(input_filename: Path) -> torch.Tensor:
    """
    Read an Numpy file with Torch and return a torch.Tensor.

    :param input_filename: Source image file path.
    :return: torch.Tensor of shape (C, H, W).
    """
    numpy_array = np.load(input_filename)
    torch_tensor = torch.from_numpy(numpy_array)
    return torch_tensor


def write_image_numpy(tensor: torch.Tensor, output_filename: Path) -> None:
    """
    Save a torch.Tensor as native Numpy array.

    :param tensor: Tensor to save.
    :param output_filename: Target filename.
    :return: None.
    """
    numpy_array = tensor.numpy()
    np.save(output_filename, numpy_array)


def check_loaded_image(type: str, image_file: Path, tensor: torch.Tensor) -> None:
    """
    Check that an image loaded as a Tensor has the expected forat, size, and value range.

    :param type: Label for printing progress.
    :param image_file: Path to reference png.
    :param tensor: Loaded torch.Tensor.
    :return: None.
    """
    im = Image.open(image_file)
    reference_tensor = TF.to_tensor(im)
    source_greyscale = im.mode == 'L'
    channels = 1 if source_greyscale else 3
    width, height = im.size
    print(f"Testing file: {image_file}, type: {type}, format: {im.format}, size: {im.size}, mode: {im.mode}")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (channels, height, width)
    assert torch.max(tensor) <= 1.0
    assert torch.min(tensor) >= 0.0
    assert torch.equal(tensor, reference_tensor)


def check_loaded_image2(type: str, image_file: Path, im2: np.ndarray) -> None:
    """
    Check that an image loaded as a numpy array has the expected forat, size, and value range.

    :param type: Label for printing progress.
    :param image_file: Path to reference png.
    :param im2: Loaded numpy array.
    :return: None.
    """
    im = Image.open(image_file)
    source_greyscale = im.mode == 'L'
    width, height = im.size
    print(f"Testing file: {image_file}, type: {type}, format: {im.format}, size: {im.size}, mode: {im.mode}")
    assert isinstance(im2, np.ndarray)
    assert im2.dtype == np.float  # type: ignore
    if source_greyscale:
        assert im2.shape == (height, width)
    else:
        assert im2.shape == (height, width, 3)
    assert np.max(im2) <= 255.0
    assert np.min(im2) >= 0.0
    im_data = np.asarray(im, np.float)  # type: ignore
    assert np.array_equal(im_data, im2)


def mount_and_convert_source_files(
        dataset: FileDataset,
        output_folder: Path,
        source_options: List[Tuple[str, bool, bool]],
        bin_libs: List[Tuple[str, str, Callable[[torch.Tensor, Path], None], Callable[[Path], torch.Tensor]]]) -> None:
    """
    Mount the dataset, and loop through all the png files creating cropped/greyscale pngs from them.
    Also create torch.Tensor and numpy array versions.

    :param dataset: File dataset containing pngs to mount.
    :param output_folder: Root folder to build variants in.
    :param source_options: List of subfolder names and greyscale/crop options.
    :param bin_libs: List of subfolder names, file suffix, and write/read functions for tensor/array options.
    :return: None.
    """
    with dataset.mount("/tmp/datasets/panda_tiles_small") as mount_context:
        input_folder = Path(mount_context.mount_point)

        for option, greyscale, crop in source_options:
            output_folder_name = output_folder / "png" / option
            output_folder_name.mkdir(parents=True, exist_ok=True)

            for image_file in input_folder.glob("*.png"):
                im = Image.open(image_file)
                im2 = convert_pillow(im, greyscale, crop)
                im2.save(output_folder_name / image_file.name)
                tensor = TF.to_tensor(im2.copy())

                for folder, suffix, write_op, _ in bin_libs:
                    target_folder = output_folder / folder / option
                    target_folder.mkdir(parents=True, exist_ok=True)
                    target_file = target_folder / image_file.with_suffix(suffix).name
                    write_op(tensor, target_file)

                print(f"Converted file: {image_file}, format: {im.format} -> {im2.format}, "
                      f"size: {im.size} -> {im2.size}, mode: {im.mode} -> {im2.mode}")


def run_profiling(
        repeats: int,
        output_folder: Path,
        source_options: List[str],
        png_libs: List[Tuple[str, Callable[[Path], torch.Tensor]]],
        png2_libs: List[Tuple[str, Callable[[Path], np.array]]],  # type: ignore
        bin_libs: List[Tuple[str, str, Callable[[torch.Tensor, Path], None], Callable[[Path], torch.Tensor]]]) -> None:
    """
    Loop through multiple repeats of each source type, loading the image file and processing it with each
    library.

    :param repeats: Number of times to process each source option.
    :param output_folder: Root folder to build variants in.
    :param source_options: List of subfolder names and greyscale/crop options.
    :param png_libs: List of image processing libraries.
    :param bin_libs: List of subfolder names, file suffix, and write/read functions for tensor/array options.
    :return: None.
    """
    for repeat in range(0, repeats):
        for source_option in source_options:
            print("~~~~~~~~~~~~~")
            print(f"repeat: {repeat}, source_option: {source_option}")
            print("~~~~~~~~~~~~~")
            source_folder = output_folder / "png" / source_option
            for image_file in source_folder.glob("*.png"):
                for lib, op in png_libs:
                    tensor = op(image_file)
                    check_loaded_image(lib, image_file, tensor)

                for lib, op in png2_libs:
                    nd = op(image_file)
                    check_loaded_image2(lib, image_file, nd)

                for folder, suffix, _, op in bin_libs:
                    target_folder = output_folder / folder / source_option
                    native_file = target_folder / image_file.with_suffix(suffix).name
                    tensor = op(native_file)
                    check_loaded_image(folder, image_file, tensor)


def wrap_run_profiling(
        repeats: int,
        output_folder: Path,
        png_libs: List[Tuple[str, Callable[[Path], torch.Tensor]]],
        png2_libs: List[Tuple[str, Callable[[Path], np.array]]],  # type: ignore
        bin_libs: List[Tuple[str, str, Callable[[torch.Tensor, Path], None], Callable[[Path], torch.Tensor]]],
        profile_name: str,
        profile_source_options: List[str]) -> None:
    """
    Setup lineProfiler and call run_profiling.

    :param repeats: Number of times to process each source option.
    :param output_folder: Root folder to build variants in.
    :param png_libs: List of image processing libraries.
    :param bin_libs: List of subfolder names, file suffix, and write/read functions for tensor/array options.
    :param profile_name: Name to use for saving profile results file.
    :param profile_source_options: List of source folders to test.
    :return: None.
    """
    def curry_run_profiling() -> None:
        """
        Create a new parameterless function by applying all the options, for ease of profiling.

        :return: None.
        """
        run_profiling(repeats,
                      output_folder,
                      profile_source_options,
                      png_libs,
                      png2_libs,
                      bin_libs)

    """
    Create a LineProfiler and time calls to convert_image, writing results to a text file.
    """
    lp = LineProfiler()
    lp.add_function(read_image_matplotlib)
    lp.add_function(read_image_matplotlib2)
    lp.add_function(read_image_opencv)
    lp.add_function(read_image_opencv2)
    lp.add_function(read_image_pillow)
    lp.add_function(read_image_pillow2)
    lp.add_function(read_image_pillow3)
    lp.add_function(read_image_scipy)
    lp.add_function(read_image_scipy2)
    lp.add_function(read_image_sitk)
    lp.add_function(read_image_skimage)
    lp.add_function(read_image_torch)
    lp.add_function(read_image_torch2)
    lp.add_function(read_image_numpy)
    lp_wrapper = lp(curry_run_profiling)
    lp_wrapper()
    with open(f"outputs/profile_{profile_name}.txt", "w", encoding="utf-8") as f:
        lp.print_stats(f)


def main() -> None:
    """
    Mount a dataset called 'panda_tiles_small', assumed to contain image files, with file extension png.
    Load each png file, convert to greyscale, and save to a separate folder.

    :return: None.
    """
    source_options: List[Tuple[str, bool, bool]] = [
        ("load", False, False),
        ("greyscale", True, False),
        ("crop", False, True),
        ("crop_greyscale", True, True),
    ]

    png_libs: List[Tuple[str, Callable[[Path], torch.Tensor]]] = [
        ("matplotlib", read_image_matplotlib),
        ("matplotlib2", read_image_matplotlib2),
        ("opencv", read_image_opencv),
        ("opencv2", read_image_opencv2),
        ("pillow", read_image_pillow),
        ("scipy", read_image_scipy),
        ("sikt", read_image_sitk),
        ("skimage", read_image_skimage),
        # ("torch", read_image_torch),
    ]

    png2_libs: List[Tuple[str, Callable[[Path], np.array]]] = [  # type: ignore
        ("pillow2", read_image_pillow2),
        ("pillow3", read_image_pillow3),
        ("scipy2", read_image_scipy2),
    ]

    bin_libs: List[Tuple[str, str, Callable[[torch.Tensor, Path], None], Callable[[Path], torch.Tensor]]] = [
        ("pt", ".pt", write_image_torch2, read_image_torch2),
        ("npy", ".npy", write_image_numpy, read_image_numpy),
    ]

    workspace = get_workspace(aml_workspace=None, workspace_config_path=None)

    dataset = get_or_create_dataset(workspace=workspace,
                                    datastore_name='himldatasets',
                                    dataset_name='panda_tiles_small')

    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True)

    mount_and_convert_source_files(dataset, output_folder, source_options, bin_libs)

    profile_sets: Dict[str, List[str]] = {
        "rgb": [source_options[0][0], source_options[2][0]],
        "grey": [source_options[1][0], source_options[3][0]]
    }

    for profile_name, profile_source_options in profile_sets.items():
        wrap_run_profiling(10,
                           output_folder,
                           png_libs,
                           png2_libs,
                           bin_libs,
                           profile_name,
                           profile_source_options)


if __name__ == '__main__':
    main()
