from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_data_loader(
    root, batch_size=32
    ):
    
    transform = get_transforms()

    dataset = ImageFolder(root, transform=transform)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"Datset length: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    
    return data_loader, dataset