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

    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix.
    Second, it perturbed the hematoxylin, eosin stains independently.
    Third, it transformed the resulting stains into regular RGB color space.
    PyTorch version of: https://github.com/gatsby2016/Augmentation-PyTorch-Transforms/blob/master/myTransforms.py

    Usage example:
        >>> transform = HEDJitter(0.05)
        >>> img = transform(img)
    """

    def __init__(self, theta: float = 0.0) -> None:
        """
        :param theta: How much to jitter HED color space.
            HED_light: theta=0.05; HED_strong: theta=0.2.
            alpha is chosen from a uniform distribution [1-theta, 1+theta].
            beta is chosen from a uniform distribution [-theta, theta].
            The jitter formula is :math:`s' = \alpha * s + \beta`.
        """
        self.theta = theta
        self.rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
        self.hed_from_rgb = torch.tensor(
            [
                [1.87798274, -1.00767869, -0.55611582],
                [-0.06590806, 1.13473037, -0.1355218],
                [-0.60190736, -0.48041419, 1.57358807],
            ]
        )
        self.log_adjust = torch.log(torch.tensor(1e-6))

    def adjust_hed(self, img: torch.Tensor) -> torch.Tensor:
        """
        Applies HED jitter to image.

        :param img: Input image.
        """

        alpha = torch.FloatTensor(img.shape[0], 1, 1, 3).uniform_(1 - self.theta, 1 + self.theta)
        beta = torch.FloatTensor(img.shape[0], 1, 1, 3).uniform_(-self.theta, self.theta)

        # Separate stains
        img = img.permute([0, 2, 3, 1])
        img = torch.maximum(img, 1e-6 * torch.ones(img.shape))
        stains = (torch.log(img) / self.log_adjust) @ self.hed_from_rgb
        stains = torch.maximum(stains, torch.zeros(stains.shape))

        # perturbations in HED color space
        stains = alpha * stains + beta

        # Combine stains
        img = -(stains * (-self.log_adjust)) @ self.rgb_from_hed
        img = torch.exp(img)
        img = torch.clip(img, 0, 1)
        img = img.permute(0, 3, 1, 2)

        # Normalize
        imin = torch.amin(img, dim=[1, 2, 3], keepdim=True)
        imax = torch.amax(img, dim=[1, 2, 3], keepdim=True)
        img = (img - imin) / (imax - imin)

        return img

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        original_shape = img.shape
        if len(original_shape) == 3:
            img = img.unsqueeze(0)  # add batch dimension if missing
        # if the input is a bag of images, hed jitter needs to run on each image separately
        if img.shape[0] > 1:
            for i in range(img.shape[0]):
                img_tile = img[i]
                img[i] = self.adjust_hed(img_tile.unsqueeze(0))
            return img
        else:
            img = self.adjust_hed(img)
            if len(original_shape) == 3:
                return img.squeeze(0)
            return img


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
        nimg = torch.Tensor(nimg).unsqueeze(0).permute(0, 3, 1, 2) / 255.0  # back to pytorch format

        return nimg

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        original_shape = img.shape
        if len(original_shape) == 3:
            img = img.unsqueeze(0)  # add batch dimension if missing
        # if the input is a bag of images, stain normalization needs to run on each image separately
        if img.shape[0] > 1:
            for i in range(img.shape[0]):
                img_tile = img[i]
                img[i] = self.stain_normalize(img_tile.unsqueeze(0), self.reference_mean, self.reference_std)
            return img
        else:
            img = self.stain_normalize(img, self.reference_mean, self.reference_std)
            if len(original_shape) == 3:
                return img.squeeze(0)
            return img


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

    @staticmethod
    def apply_gaussian_blur(sample: torch.Tensor, kernel_size: int, p: float, min: float, max: float) -> torch.Tensor:
        """
        Applies Gaussian blur to image.

        :param img: Input image.
        :param kernel_size: Size of the Gaussian kernel, e.g., about 10% of the image size.
        :param p: Probability of applying blur.
        :param min: lower bound of the interval from which we sample the STD
        :param max: upper bound of the interval from which we sample the STD
        """
        if np.random.binomial(n=1, p=p):
            sigma = np.random.uniform(low=min, high=max)  # (max - min) * np.random.random_sample() + min
            sample = sample.permute([0, 2, 3, 1]).squeeze().numpy()  # only 3 channels, color channel last
            sample = cv2.GaussianBlur(sample, (kernel_size, kernel_size), sigma)
            sample = torch.Tensor(sample).unsqueeze(0).permute(0, 3, 1, 2)
        return sample

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        original_shape = img.shape
        if len(original_shape) == 3:
            img = img.unsqueeze(0)  # add batch dimension if missing
        # if the input is a bag of images, gaussian blur needs to run on each image separately
        if img.shape[0] > 1:
            for i in range(img.shape[0]):
                img_tile = img[i]
                img[i] = self.apply_gaussian_blur(
                    sample=img_tile.unsqueeze(0), kernel_size=self.kernel_size, p=self.p, min=self.min, max=self.max
                )
            return img
        else:
            img = self.apply_gaussian_blur(
                sample=img, kernel_size=self.kernel_size, p=self.p, min=self.min, max=self.max
            )
            if len(original_shape) == 3:
                return img.squeeze(0)
            return img


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
        angle = np.random.choice([0.0, 90.0, 180.0, 270.0])

        if angle != 0.0:
            sample = TF.rotate(sample, angle)

        return sample
