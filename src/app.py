import logging

from flask import Flask, jsonify, request

from src.config import Config
from src.features.transform import transform_image
from src.models.class_mapping import load_class_mapping
from src.models.model_loader import load_model

logger = logging.getLogger(__name__)
app = Flask(__name__)


@app.route('/')
def health_check():
    return {"status": "ok"}


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        prediction_result = get_prediction(image_bytes=img_bytes, model_name=cfg.model_name, device=cfg.device)
        return jsonify(prediction_result)
    else:
        raise ValueError("Incorrect request type, use POST method.")


def get_prediction(image_bytes, model_name, device):
    """
    For provided image and model name, perform necessary transformations, apply model (forward pass) and select best
    prediction.
    Args:
        image_bytes (bytes): Image to process
        model_name (str): Model name to apply corresponding transformation.
        device (str or torch.device): device to run prediction on

    Returns:
        dict: {class_idx: <class_idx>, class_name: <class_name>}
    """
    tensor = transform_image(image_bytes=image_bytes, model_name=model_name)
    tensor = tensor.to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_class_idx = str(y_hat.item())
    predicted_class_name = class_mapping[predicted_class_idx]
    logger.info(f"Predicted class: {predicted_class_name} ({predicted_class_idx})")
    return {'class_idx': predicted_class_idx, 'class_name': predicted_class_name}


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    cfg = Config()
    model = load_model(model_name=cfg.model_name, conf=cfg)
    model.eval()
    model.to(cfg.device)
    class_mapping = load_class_mapping(cfg.model_name, cfg)
    app.run()
