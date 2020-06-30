import io
from PIL import Image
import logging
import torchvision.transforms as transforms
from src.config import Config

logger = logging.getLogger(__name__)


def transform_image_to_imagenet(image_bytes):
    """
    Preprocess image to match size of standard ImageNet images, e.g. 224x224
    :param image_bytes:
    :return:
    """
    my_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    if image.getbands() != ('R', 'G', 'B'):
        logger.debug(f'Converting the image to RGB scheme. Original: {image.getbands()}')
        image = image.convert('RGB')
    logger.debug(f'Image shape: {image.size}')
    return my_transforms(image).unsqueeze(0)


def transform_image(image_bytes, model_name, conf):
    logger.debug(f'Model name: {model_name}')
    if model_name in conf.torchvision_models:
        return transform_image_to_imagenet(image_bytes)
    elif model_name in conf.custom_models:
        raise NotImplementedError("No preprocessor for the specified model defined")
    else:
        raise ValueError("Transformation for the specified model is not available")
