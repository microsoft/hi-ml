from health_multimodal.image import ImageModel
from health_multimodal.image import get_biovil_resnet as _biovil_resnet


def biovil_resnet(pretrained: bool = False) -> ImageModel:
    """Get BioViL image encoder.

    :param pretrained: Load pretrained weights from a checkpoint.
    """
    model = _biovil_resnet(pretrained=pretrained)
    return model
