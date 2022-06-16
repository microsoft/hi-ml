from transformers import AutoModel, AutoTokenizer

from health_multimodal.text.inference_engine import TextInferenceEngine
from health_multimodal.image.inference_engine import ImageInferenceEngine
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine

from health_multimodal.image.model.model import ImageModel
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

# Load the text inference engine
URL = "microsoft/BiomedVLP-CXR-BERT-specialized"
text_inference = TextInferenceEngine(
    tokenizer=AutoTokenizer.from_pretrained(URL, trust_remote_code=True),
    text_model=AutoModel.from_pretrained(URL, trust_remote_code=True))

# Load the image inference engine
image_inference = ImageInferenceEngine(
    image_model=ImageModel(img_model_type="resnet50", joint_feature_size=128),
    transforms=create_chest_xray_transform_for_inference(resize=512, center_crop_size=480))

# Instantiate the joint inference engine
image_text_inference = ImageTextInferenceEngine(image_inference_engine=image_inference,
                                                text_inference_engine=text_inference)

# Input text prompts and image
text_prompts = ["There is no pneumonia"]
IMAGE_DIR = "/datasetdrive/MIMIC-CXR-V2-512-NIFTI/files/p10/p10999737/s52341872/"
IMAGE_PATH = IMAGE_DIR + "5aea5877-40b40fee-5bccd163-ca1bf0ce-a95c213d.nii.gz"

sim_map = image_text_inference.get_similarity_map_from_raw_data(image_path=IMAGE_PATH, query_text=text_prompts[0])
