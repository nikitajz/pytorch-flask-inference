import pytest
import torch
from PIL import Image

from src.features.transform import transform_image_to_imagenet, TransformImage
from tests.utils import convert_image_to_bytes


@pytest.fixture
def example_large_image():
    img = Image.open('tests/data/cat.jpg', mode='r')
    return img


def test_transform_image_to_imagenet(example_large_image):
    image = example_large_image
    init_size = image.size
    image_bytes = convert_image_to_bytes(image)
    resized_image_tensor = transform_image_to_imagenet(image_bytes)
    transformed_size = list(resized_image_tensor.size())[1:]  # drop channels dims
    assert len(init_size) == len(transformed_size), "Make sure to compare only image size (remove other dims)"
    assert [dim == 224 for dim in transformed_size], "Transformed image should be of size 224x224"
    assert isinstance(resized_image_tensor, torch.Tensor), "Transformed image should be a torch.Tensor"


def test_transform_image_custom_model(example_large_image):
    # change to custom model size once implemented
    image_bytes = convert_image_to_bytes(example_large_image)
    with pytest.raises(NotImplementedError):
        transform_image = TransformImage("custom_model_name")
        transform_image(image_bytes)


def test_transform_image_k_model(example_large_image):
    image_bytes = convert_image_to_bytes(example_large_image)
    with pytest.raises(ValueError):
        transform_image = TransformImage("unknown_model_name")
        transform_image(image_bytes)
