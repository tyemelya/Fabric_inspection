import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

def load_dinov2(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)

    return model

def freeze_backbone(model):
    model.eval()
    
    for p in model.parameters():
        p.requires_grad = False
        
    return model

def unfreeze_backbone(model):
    model.train()
    
    for p in model.parameters():
        p.requires_grad = True
        
    return model

#Linear head
class LinearHead(nn.Module):
    def __init__(self, in_dim=768, num_classes=None):
        super().__init__()
        
        if num_classes is None:
            raise ValueError("num_classes must be specified.")
        
        self.classifier = nn.Linear(in_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)
    
def add_lora(
    model,
    r=16,
    alpha=32,
    dropout=0.1,
):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "qkv",
            "proj",
        ],
    )

    model = get_peft_model(
        model,
        config,
    )

    model.print_trainable_parameters()

    return model