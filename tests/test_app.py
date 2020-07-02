import json

import numpy as np
import pytest
import torch
from PIL import Image
from mock import patch

from src import app
from src.config import Config
from tests.utils import convert_image_to_bytes, MockModel


@pytest.fixture
def client():
    app.app.config['TESTING'] = True
    with app.app.test_client() as client:
        yield client


@pytest.fixture
def example_large_image():
    img = Image.open('tests/data/cat.jpg', mode='r')
    return img


def test_health_check(client):
    response = client.get('/status')
    status = json.loads(response.get_data(as_text=True))
    assert status == {"status": "ok"}


def test_get_prediction(monkeypatch, example_large_image):
    with patch('src.features.transform.transform_image') as transform_image:
        transform_image.return_value = torch.rand(1, 3, 244, 244)
        img_bytes = convert_image_to_bytes(example_large_image)
        mock_outputs = np.random.uniform(0, 0.7, 1000)
        mock_outputs[3] = 0.99  # "tiger_shark"
        mock_outputs = torch.Tensor(mock_outputs).unsqueeze(0)
        app.model = MockModel()

        def mock_return(*args):
            return mock_outputs

        monkeypatch.setattr(app.model, "forward", mock_return)
        app.class_mapping = json.load(open('src/data/imagenet_class_mapping.json'))
        expected_prediction = {'class_id': "n01491361", 'class_name': "tiger_shark"}
        actual_prediction = app.get_prediction(img_bytes, "vgg11", "cpu")
        assert expected_prediction == actual_prediction


def test_upload_file(client, example_large_image):
    app.cfg = Config()
    response = client.get('/')
    assert response.status_code == 200
    assert b'Upload image to classify' in response.data

# TODO: add test for `predict`
