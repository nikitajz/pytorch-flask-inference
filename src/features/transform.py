import io
import logging

import torchvision
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)


def transform_image_to_imagenet(image_bytes):
    """
    Preprocess image to match size of standard ImageNet images 224x224, convert to Pytorch tensor and normalize.

    Args:
        image_bytes (bytes): Image to process

    Returns:
        Tensor: 4-dimensional PyTorch tensor where first dimension is batch (of size 1).
    """
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
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


def transform_image(image_bytes, model_name):
    """
    Apply transformation corresponding to the model. For torchvision models see func `transform_image_to_imagenet`.
    Args:
        image_bytes (bytes): Image to process
        model_name (str): Model name to be applied to transformed image.

    Returns:
        Tensor: 4-dimensional PyTorch tensor where first dimension is batch (of size 1).
    """
    logger.debug(f'Model name: {model_name}')
    if hasattr(torchvision.models, model_name):
        return transform_image_to_imagenet(image_bytes)
    elif model_name == "custom_model_name":
        raise NotImplementedError("No preprocessor for the specified model defined")
    else:
        raise ValueError("Transformation for the specified model is not available")
