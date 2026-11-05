import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the PyTorch CIFAR repository to the path
CIFAR_REPO_PATH = os.path.join(os.path.dirname(__file__), 'PyTorch_CIFAR10-master')
sys.path.append(CIFAR_REPO_PATH)

DATA_DIR   = './data'
BATCH_SIZE = 1
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")
print(f"Data directory: {DATA_DIR}/")

def get_dataset(dataset_name: str):
    name = dataset_name.lower()
    if name == 'cifar10':
        # Use CIFAR-10 specific preprocessing (32x32, no resizing)
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2471, 0.2435, 0.2616)),
        ])
        ds = datasets.CIFAR10(root=DATA_DIR, train=False,
                              download=True, transform=tf)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return ds, loader, tf, num_classes

def get_model(model_name: str, dataset_name: str, num_classes: int):
    dset  = dataset_name.lower()

    if dset == 'cifar10':
        if model_name == 'resnet18':
            from cifar10_models.resnet import resnet18
            model = resnet18(pretrained=True, device=DEVICE)
            actual_model_name = "resnet18"
        elif model_name == 'resnet34':
            from cifar10_models.resnet import resnet34
            model = resnet34(pretrained=True, device=DEVICE)
            actual_model_name = "resnet34"
        elif model_name == 'resnet50':
            from cifar10_models.resnet import resnet50
            model = resnet50(pretrained=True, device=DEVICE)
            actual_model_name = "resnet50"
        elif model_name == 'vgg11_bn':
            from cifar10_models.vgg import vgg11_bn
            model = vgg11_bn(pretrained=True, device=DEVICE)
            actual_model_name = "vgg11_bn"
        elif model_name == 'vgg13_bn':
            from cifar10_models.vgg import vgg13_bn
            model = vgg13_bn(pretrained=True, device=DEVICE)
            actual_model_name = "vgg13_bn"
        elif model_name == 'vgg16_bn':
            from cifar10_models.vgg import vgg16_bn
            model = vgg16_bn(pretrained=True, device=DEVICE)
            actual_model_name = "vgg16_bn"
        elif model_name == 'vgg19_bn':
            from cifar10_models.vgg import vgg19_bn
            model = vgg19_bn(pretrained=True, device=DEVICE)
            actual_model_name = "vgg19_bn"
        elif model_name == 'densenet121':
            from cifar10_models.densenet import densenet121
            model = densenet121(pretrained=True, device=DEVICE)
            actual_model_name = "densenet121"
        elif model_name == 'densenet161':
            from cifar10_models.densenet import densenet161
            model = densenet161(pretrained=True, device=DEVICE)
            actual_model_name = "densenet161"
        elif model_name == 'densenet169':
            from cifar10_models.densenet import densenet169
            model = densenet169(pretrained=True, device=DEVICE)
            actual_model_name = "densenet169"
        elif model_name == 'googlenet':
            from cifar10_models.googlenet import googlenet
            model = googlenet(pretrained=True, device=DEVICE)
            actual_model_name = "googlenet"
        elif model_name == 'inception_v3':
            from cifar10_models.inception import inception_v3
            model = inception_v3(pretrained=True, device=DEVICE)
            actual_model_name = "inception_v3"
        elif model_name == 'mobilenet_v2':
            from cifar10_models.mobilenetv2 import mobilenet_v2
            model = mobilenet_v2(pretrained=True, device=DEVICE)
            actual_model_name = "mobilenet_v2"
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Test model with appropriate input size
    with torch.no_grad():
        if dset == 'cifar10':
            x = torch.randn(1, 3, 32, 32).to(DEVICE)  # CIFAR-10 size
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        out = model(x)
        if out.shape[1] != num_classes:
            raise RuntimeError(f"Model output ({out.shape[1]}) ≠ expected classes ({num_classes})")

    return model.to(DEVICE).double().eval(), actual_model_name

def load_data_and_model(dataset_name='cifar10', model_name='resnet18'):
    ds, loader, transform, num_classes = get_dataset(dataset_name)
    model, actual_model_name = get_model(model_name, dataset_name, num_classes)
    print(f">>> Loaded {dataset_name} dataset ({len(ds)} samples, {num_classes} classes)")
    print(f">>> Loaded {actual_model_name} model with {num_classes}-way output")
    return ds, loader, transform, model

def load_dataset(dataset_name='cifar10'):
    ds, loader, transform, _ = get_dataset(dataset_name)
    return ds, loader, transform

def load_model(model_name='resnet18', dataset_name='cifar10'):
    _, _, _, num_classes = get_dataset(dataset_name)
    model, _ = get_model(model_name, dataset_name, num_classes)
    return model

if __name__ == '__main__':
    # Test configurations using the local CIFAR models
    test_configs = [
        ('cifar10', 'resnet18'),
        ('cifar10', 'resnet34'),
        ('cifar10', 'vgg11_bn'),
        ('cifar10', 'densenet121'),
        ('cifar10', 'googlenet'),
    ]
    
    for dataset_name, model_name in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing {model_name} on {dataset_name}...")
        print(f"{'='*50}")
        
        try:
            dataset, dataloader, transform, model = load_data_and_model(dataset_name, model_name)
            
            # Test a few samples
            for imgs, labels in dataloader:
                preds = model(imgs.to(DEVICE))
                print(f"Input shape: {imgs.shape}")
                print(f"Predictions shape: {preds.shape}")
                print(f"Predicted classes: {preds.argmax(1)}")
                print(f"Actual classes: {labels}")
                break
            
        except Exception as e:
            print(f"Error testing {model_name} on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()