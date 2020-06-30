import json


def load_class_mapping(model_name, conf):
    if model_name in conf.torchvision_models:
        return json.load(open('data/imagenet_class_mapping.json'))
    elif model_name in conf.custom_models:
        raise NotImplementedError("No class mapping for the specified model defined")
    else:
        raise ValueError("Class mapping for the specified model is not available")