from skimage import color
import cv2
import numpy as np
import torch


class HEDJitter(object):
    """Randomly perturbe the HED color space value an RGB image.
    First, it disentangled the hematoxylin and eosin color channels by color deconvolution method using a fixed matrix.
    Second, it perturbed the hematoxylin, eosin and DAB stains independently.
    Third, it transformed the resulting stains into regular RGB color space.
    Args:
        theta (float): How much to jitter HED color space,
         alpha is chosen from a uniform distribution [1-theta, 1+theta]
         beta is chosen from a uniform distribution [-theta, theta]
         the jitter formula is **s' = \alpha * s + \beta**
    """
    def __init__(self, theta: float = 0.) -> None:   # HED_light: theta=0.05; HED_strong: theta=0.2
        self.theta = theta

    @staticmethod
    def adjust_hed(img: torch.Tensor, theta: float) -> torch.Tensor:
        alpha = np.random.uniform(1-theta, 1+theta, (1, 3))
        beta = np.random.uniform(-theta, theta, (1, 3))

        img = img.permute([0, 2, 3, 1]).numpy()  # channel dim must be last for next function
        s = color.rgb2hed(img)
        ns = alpha * s + beta  # perturbations in HED color space
        nimg = color.hed2rgb(ns)
        nimg = np.clip(nimg, 0, 1)
        nimg = torch.Tensor(nimg).permute(0, 3, 1, 2)  # back to pytorch format

        return nimg

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.adjust_hed(img, self.theta)


class StainNormalization(object):
    """Normalize the stain of an image given a reference image. Following
     Erik Reinhard,Bruce Gooch., “Color Transfer between Images,” IEEE ComputerGraphics and Applications. 
    First, mask all white pixels.
    Second, convert remaining pixels to lab space and normalize each channel.
    Third, add mean and std of reference image.
    Fourth, convert back to rgb and add white pixels back.
    """
    def __init__(self) -> None:
        # mean and std per channel of a reference image
        self.reference_mean = np.array([148.60, 169.30, 105.97])
        self.reference_std = np.array([41.56, 9.01, 6.67])

    @staticmethod
    def stain_normalize(img: torch.Tensor, reference_mean: np.ndarray, reference_std: np.ndarray) -> torch.Tensor:
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


