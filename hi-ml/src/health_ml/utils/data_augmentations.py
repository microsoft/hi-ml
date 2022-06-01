#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF


class HEDJitter(object):
    """
    Randomly perturbe the HED color space value an RGB image.

    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix,
    taken from Ruifrok and Johnston (2001): "Quantification of histochemical staining by color deconvolution."
    Second, it perturbed the hematoxylin, eosin stains independently.
    Third, it transformed the resulting stains into regular RGB color space.

    Usage example:
        >>> transform = HEDJitter(0.05)
        >>> img = transform(img)
    """

    def __init__(self, theta: float = 0.) -> None:
        """
        :param theta: How much to jitter HED color space.
            HED_light: theta=0.05; HED_strong: theta=0.2.
            alpha is chosen from a uniform distribution [1-theta, 1+theta].
            beta is chosen from a uniform distribution [-theta, theta].
            The jitter formula is :math:`s' = \alpha * s + \beta`.
        """
        self.theta = theta
        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                                          [0.07, 0.99, 0.11],
                                          [0.27, 0.57, 0.78]])
        self.hed_from_rgb = torch.tensor([[1.87798274, -1.00767869, -0.55611582],
                                          [-0.06590806, 1.13473037, -0.1355218],
                                          [-0.60190736, -0.48041419, 1.57358807]])

    @staticmethod
    def adjust_hed(img: torch.Tensor,
                   theta: float,
                   stain_from_rgb_mat: torch.Tensor,
                   rgb_from_stain_mat: torch.Tensor
                   ) -> torch.Tensor:
        """
        Applies HED jitter to image.

        :param img: Input image.
        :param theta: Strength of the jitter. HED_light: theta=0.05; HED_strong: theta=0.2.
        :param stain_from_rgb_mat: Transformation matrix from HED to RGB.
        :param rgb_from_stain_mat: Transformation matrix from RGB to HED.
        """
        alpha = torch.FloatTensor(1, 3).uniform_(1 - theta, 1 + theta)
        beta = torch.FloatTensor(1, 3).uniform_(-theta, theta)

        # Only perturb the H (=0) and E (=1) channels
        alpha[0][-1] = 1.
        beta[0][-1] = 0.

        # Separate stains
        img = img.permute([0, 2, 3, 1])
        img = img + 2  # for consistency with skimage
        stains = -torch.log10(img) @ stain_from_rgb_mat
        stains = alpha * stains + beta  # perturbations in HED color space

        # Combine stains
        img = 10 ** (-stains @ rgb_from_stain_mat) - 2
        img = torch.clip(img, 0, 1)
        img = img.permute(0, 3, 1, 2)

        return img

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] != 3:
            raise ValueError("HED jitter can only be applied to images with 3 channels (RGB).")
        return self.adjust_hed(img, self.theta, self.hed_from_rgb, self.rgb_from_hed)


class StainNormalization(object):
    """Normalize the stain of an image given a reference image.

        Following Erik Reinhard, Bruce Gooch (2001): “Color Transfer between Images.”
        First, mask all white pixels.
        Second, convert remaining pixels to lab space and normalize each channel.
        Third, add mean and std of reference image.
        Fourth, convert back to rgb and add white pixels back.

        Usage example:
            >>> transform = StainNormalization()
            >>> img = transform(img)
    """

    def __init__(self) -> None:
        # mean and std per channel of a reference image
        self.reference_mean = np.array([148.60, 169.30, 105.97])
        self.reference_std = np.array([41.56, 9.01, 6.67])

    @staticmethod
    def stain_normalize(img: torch.Tensor, reference_mean: np.ndarray, reference_std: np.ndarray) -> torch.Tensor:
        """
        Applies stain normalization to image.

        :param img: Input image.
        :param reference_mean: Mean per channel of a reference image.
        :param reference_std: STD per channel of a reference image.
        """
        img = img.permute([0, 2, 3, 1]).squeeze().numpy() * 255  # only 3 channels, color channel last, range 0 - 255
        img = img.astype(np.uint8)  # type: ignore

        whitemask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        whitemask = whitemask > 215
        whitemask = np.repeat(whitemask[:, :, np.newaxis], 3, axis=2)

        imagelab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        imagelab_masked = np.ma.MaskedArray(imagelab, whitemask)  # type: np.ma.MaskedArray

        epsilon = 1e-11  # Sometimes STD is near 0, add epsilon to avoid div by 0
        imagelab_masked_mean = imagelab_masked.mean(axis=(0, 1))
        imagelab_masked_std = imagelab_masked.std(axis=(0, 1)) + epsilon

        # Normalize and apply reference img statistics
        imagelab = (imagelab - imagelab_masked_mean) / imagelab_masked_std * reference_std + reference_mean
        imagelab = np.clip(imagelab, 0, 255)
        imagelab = imagelab.astype(np.uint8)
        nimg = cv2.cvtColor(imagelab, cv2.COLOR_LAB2RGB)
        nimg[whitemask] = img[whitemask]  # add back white pixels
        nimg = torch.Tensor(nimg).unsqueeze(0).permute(0, 3, 1, 2) / 255.  # back to pytorch format

        return nimg

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.stain_normalize(img, self.reference_mean, self.reference_std)


class GaussianBlur(object):
    """
    Implements Gaussian blur as described in the SimCLR paper (https://arxiv.org/abs/2002.05709).

    Blur image using a Gaussian kernel with a randomly sampled STD.
    Slight modification of the code in pl_bolts to make it work with our transform pipeline.

    Usage example:
            >>> transform = GaussianBlur(kernel_size=int(224 * 0.1) + 1)
            >>> img = transform(img)
    """

    def __init__(self, kernel_size: int, p: float = 0.5, min: float = 0.1, max: float = 2.0) -> None:
        """
        :param kernel_size: Size of the Gaussian kernel, e.g., about 10% of the image size.
        :param p: Probability of applying blur.
        :param min: lower bound of the interval from which we sample the STD
        :param max: upper bound of the interval from which we sample the STD
        """
        self.min = min
        self.max = max
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = np.array(sample.squeeze())  # type: ignore
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
            sample = torch.Tensor(sample).unsqueeze(0)

        return sample


class RandomRotationByMultiplesOf90(object):
    """
    Rotation of input image by 0, 90, 180 or 270 degrees.

    Usage example:
            >>> transform = RandomRotationByMultiplesOf90()
            >>> img = transform(img)
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        angle = np.random.choice([0., 90., 180., 270.])

        if angle != 0.:
            sample = TF.rotate(sample, angle)

        return sample
