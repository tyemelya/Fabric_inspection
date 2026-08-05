from dataclasses import dataclass
import numpy as np
from app.ml.loader import load_fabric_model
import torch
from PIL import Image
from app.ml.data import get_transforms
import torch.nn.functional as F
import json

@dataclass
class VisionResult:
    prediction: str
    confidence: float
    embedding: np.ndarray
    
class VisionTool:
    def __init__(self, model_path, device):
        self.model = load_fabric_model(
            model_path,
            device
        )
        
        with open("data/class_names.json", "r") as f:
            self.class_names = json.load(f)
    
        self.device = device
        self.transform = get_transforms()
        self.model.eval()

    def get_embedding(self, image_path):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            feats = F.normalize(
                self.model(tensor),
                dim=1
            ).cpu().numpy()

        return feats.squeeze()    
    
    def analyze(self, image_path):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        tensor = tensor.unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            features = self.model.get_features(tensor)
            label, confidence = self.model.classify(
                features
            )
        embedding = F.normalize(
            features,
            dim=1
        )
        
        return VisionResult(
            prediction=self.class_names[label.item()],
            confidence=confidence.item(),
            embedding=embedding.squeeze(0).cpu().numpy()
        )