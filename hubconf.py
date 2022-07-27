# autopep8: off
dependencies = ["torch==1.9.0", "torchvision==0.10.0"]

import sys
from pathlib import Path
repo_dir = Path(__file__).parent
multimodal_src_dir = repo_dir / "hi-ml-multimodal" / "src" / "health_multimodal"
sys.path.append(str(multimodal_src_dir))

from image import ImageModel
from image import get_biovil_resnet as _biovil_resnet
# autopep8: on


def biovil_resnet(pretrained: bool = False) -> ImageModel:
    """Get BioViL image encoder.

    :param pretrained: Load pretrained weights from a checkpoint.
    """
    model = _biovil_resnet(pretrained=pretrained)
    return model
