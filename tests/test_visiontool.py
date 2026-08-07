from app.tools.vision import VisionTool, VisionResult
import pytest
import torch
from PIL import Image
import numpy as np

class FakeModel:
    def eval(self):
        pass

    def get_features(self, tensor):
        # fake feature vector: batch_size=1, dim=3
        return torch.tensor([
            [1.0, 2.0, 3.0]
        ])

    def classify(self, features):
        # class index, confidence
        return torch.tensor(0), torch.tensor(0.95)

def test_vision_analyze(monkeypatch, tmp_path):
    # create fake image
    image_path = tmp_path / "test.jpg"

    image = Image.new(
        "RGB",
        (224, 224),
        color="white"
    )
    image.save(image_path)

    # replace real model loading
    monkeypatch.setattr(
        "app.tools.vision.load_fabric_model",
        lambda model_path, device: FakeModel()
    )

    vision_tool = VisionTool(
        model_path="fake.pt",
        device="cpu"
    )

    result = vision_tool.analyze(str(image_path))
    assert isinstance(result, VisionResult)
    assert result.prediction in vision_tool.class_names
    assert result.confidence == pytest.approx(0.95)
    assert isinstance(result.embedding, np.ndarray)
    assert result.embedding.shape == (3,)
        
def test_vision_invalid_image(monkeypatch):
    monkeypatch.setattr(
        "app.tools.vision.load_fabric_model",
        lambda model_path, device: FakeModel()
    )

    vision_tool = VisionTool(
        model_path="fake.pt",
        device="cpu"
    )

    with pytest.raises(Exception):
        vision_tool.analyze("does_not_exist.jpg")