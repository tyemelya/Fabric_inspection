from app.ml.models import add_lora, load_dinov2, freeze_backbone, LinearHead
import torch
from app.ml.inference_model import FabricDINOModel

def load_fabric_model(checkpoint_path: str, device: str):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Infer classifier dimensions from saved weights
    classifier_state = checkpoint["classifier_state_dict"]
    num_classes = classifier_state["classifier.weight"].shape[0]
    input_dim = classifier_state["classifier.weight"].shape[1]
    
    backbone = load_dinov2(device)
    freeze_backbone(backbone)
    backbone = add_lora(backbone)


    classifier = LinearHead(
        in_dim=input_dim,
        num_classes=num_classes
    )
    backbone.load_state_dict(
        checkpoint["backbone_state_dict"]
    )
    classifier.load_state_dict(
        checkpoint["classifier_state_dict"]
    )

    model = FabricDINOModel(
        backbone,
        classifier
    )

    model.to(device)
    model.eval()

    return model