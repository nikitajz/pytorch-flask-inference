from flask import Flask, jsonify, request
import logging

from src.models.class_mapping import load_class_mapping
from src.models.model_loader import load_model
from src.features.transform import transform_image
from src.config import Config

logger = logging.getLogger(__name__)
app = Flask(__name__)
cfg = Config()
model = load_model(model_name=cfg.model_name, conf=cfg)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes, model_name=cfg.model_name, conf=cfg)
        return jsonify({'class_id': class_id, 'class_name': class_name})
    else:
        raise ValueError("Incorrect request type, use POST method.")


def get_prediction(image_bytes, model_name, conf):
    tensor = transform_image(image_bytes=image_bytes, model_name=model_name, conf=conf)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    predicted_class = class_mapping[predicted_idx]
    logger.info(f"Predicted class: {predicted_class[1]}({predicted_idx})")
    return predicted_class


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)

    class_mapping = load_class_mapping(cfg.model_name, cfg)
    app.run()

    model.eval()
    model.to(cfg.device)
