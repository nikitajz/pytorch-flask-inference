import io

import torch.nn as nn


def convert_image_to_bytes(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    return image_bytes


class MockModel(nn.Module):
    def __init__(self, *args, pretrained=False, **kwargs):
        super().__init__()
        self.pretrained = pretrained

    @staticmethod
    def _get_name():
        return "model_name"

    def forward(self, tensor):
        return tensor
