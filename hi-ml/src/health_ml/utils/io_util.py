#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, Iterable, List, Optional, Tuple, Type, TypeVar

import SimpleITK as sitk
import numpy as np
import pydicom as dicom
import torch
from numpy.lib.npyio import NpzFile

from health_azure.utils import PathOrString
from health_ml.common import common_util
from health_ml.common.type_annotations import TupleFloat3
from health_ml.utils.image_util import ImageHeader, get_unit_image_header

TensorOrNumpyArray = TypeVar('TensorOrNumpyArray', torch.Tensor, np.ndarray)


class PhotometricInterpretation(Enum):
    MONOCHROME1 = "MONOCHROME1"
    MONOCHROME2 = "MONOCHROME2"


class DicomTags(Enum):
    # DICOM General Study Module Attributes
    # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.2.html#table_C.7-3
    StudyInstanceUID = "0020|000D"
    StudyID = "0020|0010"
    # DICOM General Series Module Attributes
    # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.3.html#table_C.7-5a
    Modality = "0008|0060"
    SeriesInstanceUID = "0020|000E"
    PatientPosition = "0018|5100"
    # DICOM Frame of Reference Module Attributes
    # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.4.html#table_C.7-6
    FrameOfReferenceUID = "0020|0052"
    # DICOM General Image Module Attributes
    # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.html#table_C.7-9
    ImageType = "0008|0008"
    InstanceNumber = "0020|0013"
    # DICOM Image Plane Module Attributes
    # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html#table_C.7-10
    ImagePositionPatient = "0020|0032"
    # DICOM Image Pixel Description Macro Attributes
    # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#table_C.7-11c
    PhotometricInterpretation = "0028|0004"
    BitsAllocated = "0028|0100"
    BitsStored = "0028|0101"
    HighBit = "0028|0102"
    PixelRepresentation = "0028|0103"
    # DICOM CT Image Module Attributes
    # See: http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.2.html#table_C.8-3
    RescaleIntercept = "0028|1052"
    RescaleSlope = "0028|1053"


@dataclass
class ImageWithHeader:
    """
    A 3D image with header
    """
    image: np.ndarray  # Z x Y x X
    header: ImageHeader

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)


class MedicalImageFileType(Enum):
    """
    Supported types of medical image formats
    """
    NIFTI_COMPRESSED_GZ = ".nii.gz"
    NIFTI = ".nii"


class NumpyFile(Enum):
    """
    Supported file extensions that indicate Numpy data.
    """
    NUMPY = ".npy"
    NUMPY_COMPRESSED = ".npz"


class DicomFileType(Enum):
    """
    Supported file extensions that indicate Dicom data.
    """
    Dicom = ".dcm"


VALID_NIFTI_EXTENSIONS_TUPLE = tuple([f.value for f in MedicalImageFileType])
VALID_NUMPY_EXTENSIONS_TUPLE = tuple([f.value for f in NumpyFile])
VALID_DICOM_EXTENSIONS_TUPLE = tuple([f.value for f in DicomFileType])


def _file_matches_extension(file: PathOrString, valid_extensions: Iterable[str]) -> bool:
    """
    Returns true if the given file name has any of the provided file extensions.

    :param file: The file name to check.
    :param valid_extensions: A tuple with all the extensions that are considered valid.
    :return: True if the file has any of the given extensions.
    """
    dot = "."
    extensions_with_dot = tuple(e if e.startswith(dot) else dot + e for e in valid_extensions)
    return str(file).lower().endswith(extensions_with_dot)


def is_nifti_file_path(file: PathOrString) -> bool:
    """
    Returns true if the given file name appears to belong to a compressed or uncompressed
    Nifti file. This is done based on extensions only. The file does not need to exist.

    :param file: The file name to check.
    :return: True if the file name indicates a Nifti file.
    """
    return _file_matches_extension(file, VALID_NIFTI_EXTENSIONS_TUPLE)


def is_numpy_file_path(file: PathOrString) -> bool:
    """
    Returns true if the given file name appears to belong to a Numpy file.
    This is done based on extensions only. The file does not need to exist.

    :param file: The file name to check.
    :return: True if the file name indicates a Numpy file.
    """
    return _file_matches_extension(file, VALID_NUMPY_EXTENSIONS_TUPLE)


def is_dicom_file_path(file: PathOrString) -> bool:
    """
    Returns true if the given file name appears to belong to a Dicom file.
    This is done based on extensions only. The file does not need to exist.

    :param file: The file name to check.
    :return: True if the file name indicates a Dicom file.
    """
    return _file_matches_extension(file, VALID_DICOM_EXTENSIONS_TUPLE)


def read_image_as_array_with_header(file_path: Path) -> Tuple[np.ndarray, ImageHeader]:
    """
    Read image with simpleITK as a ndarray.

    :param file_path:
    :return: Tuple of ndarray with image in Z Y X and Spacing in Z X Y
    """
    image: sitk.Image = sitk.ReadImage(str(file_path))
    img = sitk.GetArrayFromImage(image)  # This call changes the shape to ZYX
    spacing = reverse_tuple_float3(image.GetSpacing())
    # We keep origin and direction on the original shape since it is not used in this library
    # only for saving images correctly
    origin = image.GetOrigin()
    direction = image.GetDirection()

    return img, ImageHeader(origin=origin, direction=direction, spacing=spacing)


def load_nifti_image(path: PathOrString, image_type: Optional[Type] = float) -> ImageWithHeader:
    """
    Loads a single .nii, or .nii.gz image from disk. The image to load must be 3D.

    :param path: The path to the image to load.
    :return: A numpy array of the image and header data if applicable.
    :param image_type: The type to load the image in, set to None to not cast, default is float
    :raises ValueError: If the path is invalid or the image is not 3D.
    """

    def _is_valid_image_path(_path: Path) -> bool:
        """
        Validates a path for an image. Image must be .nii, or .nii.gz.
        :param _path: The path to the file.
        :return: True if it is valid, False otherwise
        """
        if _path.is_file():
            return is_nifti_file_path(_path)
        return False

    if isinstance(path, str):
        path = Path(path)
    if path is None or not _is_valid_image_path(path):
        raise ValueError("Invalid path to image: {}".format(path))

    img, header = read_image_as_array_with_header(path)

    # ensure a 3D image is loaded
    if not len(img.shape) == 3:
        raise ValueError("The loaded image should be 3D (image.shape: {})".format(img.shape))

    if image_type is not None:
        img = img.astype(dtype=image_type)

    return ImageWithHeader(image=img, header=header)


def load_numpy_image(path: PathOrString, image_type: Optional[Type] = None) -> np.ndarray:
    """
    Loads an array from a numpy file (npz or npy). The array is converted to image_type or untouched if None
    :param path: The path to the numpy file.
    :param image_type: The dtype to cast the array
    :return: ndarray
    """
    image = np.load(path)
    if type(image) is NpzFile:
        keys = list(image.keys())
        assert len(keys) == 1
        image = image[keys[0]]
    if image_type is not None:
        image = image.astype(dtype=image_type)
    return image


def load_dicom_image(path: PathOrString) -> np.ndarray:
    """
    Loads an array from a single dicom file.
    :param path: The path to the dicom file.
    """
    ds = dicom.dcmread(path)
    pixels = ds.pixel_array
    bits_stored = int(ds.BitsStored)  # type: ignore
    if ds.PhotometricInterpretation == PhotometricInterpretation.MONOCHROME1.value:
        pixel_repr = ds.PixelRepresentation
        if pixel_repr == 0:  # unsigned
            pixels = 2 ** bits_stored - 1 - pixels
        elif pixel_repr == 1:  # signed
            pixels = -1 * (pixels + 1)
        else:
            raise ValueError("Unknown value for DICOM tag 0028,0103 PixelRepresentation")
    # Return a float array, we may resize this in load_3d_images_and_stack, and interpolation will not work on int
    return pixels.astype(np.float)  # type: ignore


@dataclass(frozen=True)
class ImageAndSegmentations(Generic[TensorOrNumpyArray]):
    images: TensorOrNumpyArray


def is_png(file: PathOrString) -> bool:
    """
    Returns true if file is png
    """
    return _file_matches_extension(file, [".png"])


def load_image_in_known_formats(file: Path) -> ImageAndSegmentations[np.ndarray]:
    """
    Loads an image from a file in the given path. At the moment, this supports Nifti, numpy and dicom files.

    :param file: The path of the file to load.
    :return: a wrapper class that contains the images and segmentation if present
    """
    if is_nifti_file_path(file):
        return ImageAndSegmentations(images=load_nifti_image(path=file).image)
    elif is_numpy_file_path(file):
        return ImageAndSegmentations(images=load_numpy_image(path=file))
    elif is_dicom_file_path(file):
        return ImageAndSegmentations(images=load_dicom_image(path=file))
    elif is_png(file):
        image_with_header = load_image(path=file)
        return ImageAndSegmentations(images=image_with_header.image)
    else:
        raise ValueError(f"Unsupported image file type for path {file}")


def load_image(path: PathOrString, image_type: Optional[Type] = float) -> ImageWithHeader:
    """
    Loads an image with extension numpy or nifti
    :param path: The path to the file
    :param image_type: The type of the image
    """
    if is_nifti_file_path(path):
        return load_nifti_image(path, image_type)
    elif is_numpy_file_path(path):
        image = load_numpy_image(path, image_type)
        header = get_unit_image_header()
        return ImageWithHeader(image, header)
    elif is_png(path):
        import imageio
        image = imageio.imread(path).astype(np.float)  # type: ignore
        header = get_unit_image_header()
        return ImageWithHeader(image, header)
    raise ValueError(f"Invalid file type {path}")


def save_lines_to_file(file: Path, values: List[str]) -> None:
    """
    Writes an array of lines into a file, one value per line. End of line character is hardcoded to be `\n`.
    If the file exists already, it will be deleted.

    :param file: The path where to save the file
    :param values: A list of strings
    """
    if file.exists():
        file.unlink()

    lines = map(lambda l: l + "\n", values)
    file.write_text("".join(lines))


def reverse_tuple_float3(tuple: TupleFloat3) -> TupleFloat3:
    """
    Reverse a tuple of 3 floats.

    :param tuple: of 3 floats
    :return: a tuple of 3 floats reversed
    """
    return tuple[2], tuple[1], tuple[0]


def load_dicom_series(folder: Path) -> sitk.Image:
    """
    Load a DICOM series into a 3d sitk image.

    If the folder contains more than one series then the first will be loaded.

    :param folder: Path to folder containing DICOM series.
    :return: sitk.Image of the DICOM series.
    """
    reader = sitk.ImageSeriesReader()
    series_found = reader.GetGDCMSeriesIDs(str(folder))

    if not series_found:
        raise ValueError("Folder does not contain any DICOM series: {}".format(str(folder)))

    dicom_names = reader.GetGDCMSeriesFileNames(str(folder), series_found[0])
    reader.SetFileNames(dicom_names)

    return reader.Execute()


def load_dicom_series_and_save(folder: Path, file_name: Path) -> None:
    """
    Load a DICOM series into a 3d image and save as file_name.

    If the folder contains more than one series then the first will be loaded.
    The file format type is determined by SimpleITK based on the file name's suffix.
    List of supported file types is here:
    https://simpleitk.readthedocs.io/en/master/IO.html

    :param folder: Path to folder containing DICOM series.
    :param file_name: Path to save image.
    """
    image = load_dicom_series(folder)
    sitk.WriteImage(image, str(file_name))
