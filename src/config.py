# TODO: replace with proper config file, e.g. hydra: https://hydra.cc/docs/intro/
class Config:
    model_name = 'resnext101_32x8d'
    device = 'cuda:0'  # 'cpu'
    batch_size = 64
    num_workers = 4
    # specify custom models here and add corresponding condition in `load_model`, `transform_image` and `class_mapping`
    custom_models = []
    # only a few models listed here for simplicity
    torchvision_models = ['resnext50_32x4d', 'resnext101_32x8d', 'vgg16', 'vgg16_bn']
    allowed_models = []

    def __init__(self):
        self.allowed_models = self.torchvision_models + self.custom_models
