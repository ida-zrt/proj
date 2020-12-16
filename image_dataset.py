import torch
from torchvision import datasets, transforms
import configs as cfg

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize(size=(165, 370)),
        # transforms.RandomResizedCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(size=(165, 370)),
        # transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# ImageFolder: label is the folder name, suitable for classification
data_dir = cfg.image_folder

image_datasets = {
    'train':
    datasets.ImageFolder(data_dir + 'train',
                         transform=data_transforms['train']),
    'val':
    datasets.ImageFolder(data_dir + 'val', transform=data_transforms['val'])
}

# dataloader
dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True,
                                num_workers=4),
    'val':
    torch.utils.data.DataLoader(image_datasets['val'],
                                batch_size=4,
                                shuffle=True,
                                num_workers=4)
}

# some information for convenience
dataset_sizes = {
    'train': len(image_datasets['train']),
    'val': len(image_datasets['val'])
}
class_names = image_datasets['train'].classes
