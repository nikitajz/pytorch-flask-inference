# TODO: replace with proper config file, e.g. hydra: https://hydra.cc/docs/intro/
class Config:
    model_name = 'resnext101_32x8d'
    device = 'cpu'  # 'cuda:0'
    # specify custom models here and add corresponding condition in `load_model`
    custom_models = []
    # only a few models listed here for simplicity
    torchvision_models = ['resnext101_32x8d', 'resnext101_32x8d', 'vgg16', 'vgg16_bn']
    allowed_models = []

    def __init__(self):
        self.allowed_models = self.allowed_models + self.torchvision_models