import torch
import torch.nn.functional as F
import torch.nn as nn

class FabricDINOModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def get_features(self, x):
        """
        Raw DINO features used by classifier.
        """
        return self.backbone(x)
    
    def get_embedding(self, x):
        """
        Normalized features used by ANNOY.
        """
        features = self.backbone(x)

        return F.normalize(
            features,
            dim=1
        )

    def classify(self, features):
        logits = self.classifier(features)

        probabilities = torch.softmax(
            logits,
            dim=1
        )
        confidence, label = torch.max(
            probabilities,
            dim=1
        )

        return label, confidence

    def forward(self, x):
        return self.get_embedding(x)