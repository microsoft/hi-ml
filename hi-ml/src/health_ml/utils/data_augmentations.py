from skimage import color
import cv2
import numpy as np
import torch


class HEDJitter(object):
    """
    A class to randomly perturb the HEAD color space value of an RGB image
    """
    def __init__(self, theta: float = 0.) -> None:   # HED_light: theta=0.05; HED_strong: theta=0.2

        self.theta = theta

    @staticmethod
    def adjust_hed(img: torch.Tensor, theta: float) -> torch.Tensor:
        """
        Randomly perturb the hematoxylin-Eosin-DAB (HED) color space value of an RGB image
        Steps involved in this process:
        1. separate the stains (RGB to HED color space conversion)
        2. perturb the stains independently
        3. convert the resulting stains back to RGB color space

        :param img: A Torch Tensor representing the image to be transformed
        :param theta: A float representing how much to jitter HED color space by
        :return: a Torch Tensor of stains transformed into RGB color space.
        """
        # alpha is chosen from a uniform distribution [1 - theta, 1 + theta]
        alpha = np.random.uniform(1 - theta, 1 + theta, (1, 3))
        # beta is chosen from a uniform distribution [-theta, theta]
        beta = np.random.uniform(-theta, theta, (1, 3))

        assert img.ndim == 4, "Expected a Tensor with 4 dimensions"
        # channel dim must be last for next function
        img = img.permute([0, 2, 3, 1]).numpy()
        s = color.rgb2hed(img)

        # the jitter formula (perturbations in HED color space) is **s' = \alpha * s + \beta**
        ns = alpha * s + beta

        nimg = color.hed2rgb(ns)
        nimg = np.clip(nimg, 0, 1)
        nimg = torch.Tensor(nimg).permute(0, 3, 1, 2)

        return nimg

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.adjust_hed(img, self.theta)


class StainNormalization(object):
    """
    A class to normalize the stain of an image given a reference image. Following
    Erik Reinhard,Bruce Gooch., “Color Transfer between Images,” IEEE ComputerGraphics and Applications.
    """
    def __init__(self) -> None:
        # mean and std per channel of a reference image
        self.reference_mean = np.array([148.60, 169.30, 105.97])
        self.reference_std = np.array([41.56, 9.01, 6.67])

    @staticmethod
    def stain_normalize(img: torch.Tensor, reference_mean: np.ndarray, reference_std: np.ndarray) -> torch.Tensor:
        """
        Normalize the stain of an image given a reference image

        Steps involved:
        1. mask all white pixels
        2. convert remaining pixels to lab space and normalize each channel
        3. add mean and std of reference image
        4. convert back to rgb and add white pixels back

        :param img: the image whose stain should be normalised
        :param reference_mean: the mean of the reference image, for normalisation
        :param reference_std: the standard deviation of the reference image, for normalisation
        :return: A Torch tensor representing the image with normalized stain
        """
        assert img.ndim == 4, "Expected a Tensor with 4 dimensions"
        # only 3 channels, color channel last, range 0 - 255
        img = img.permute([0, 2, 3, 1]).squeeze().numpy() * 255
        img = img.astype(np.uint8)  # type: ignore

        whitemask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        whitemask = whitemask > 215
        whitemask = np.repeat(whitemask[:, :, np.newaxis], 3, axis=2)

        imagelab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        imagelab_masked = np.ma.MaskedArray(imagelab, whitemask)  # type: np.ma.MaskedArray

        # Sometimes STD is near 0, add epsilon to avoid div by 0
        epsilon = 1e-11
        imagelab_masked_mean = imagelab_masked.mean(axis=(0, 1))
        imagelab_masked_std = imagelab_masked.std(axis=(0, 1)) + epsilon

        # Normalize and apply reference img statistics
        imagelab = (imagelab - imagelab_masked_mean) / imagelab_masked_std * reference_std + reference_mean
        imagelab = np.clip(imagelab, 0, 255)
        imagelab = imagelab.astype(np.uint8)
        nimg = cv2.cvtColor(imagelab, cv2.COLOR_LAB2RGB)

        # add back white pixels
        nimg[whitemask] = img[whitemask]
        # convert back to Tensor
        nimg = torch.Tensor(nimg).unsqueeze(0).permute(0, 3, 1, 2) / 255.

        return nimg

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.stain_normalize(img, self.reference_mean, self.reference_std)

test