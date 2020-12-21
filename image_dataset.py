import torch
from torchvision import datasets, transforms
import configs as cfg

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize(256),
        # 大部分特征在中心位置，中心裁剪会比随机裁剪好一些transforms.RandomResizedCrop(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
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
    datasets.ImageFolder(data_dir + 'val',
                         transform=data_transforms['val'])
}

big_small_datasets = {
    'small':
    datasets.ImageFolder(data_dir + 'small', transform=data_transforms['val']),
    'medium':
    datasets.ImageFolder(data_dir + 'medium', transform=data_transforms['val']),
    'big':
    datasets.ImageFolder(data_dir + 'big', transform=data_transforms['val'])
}

big_small_dataloaders = {
    'small':
    torch.utils.data.DataLoader(big_small_datasets['small'],
                                batch_size=32,
                                shuffle=True,
                                num_workers=4),
    'medium':
    torch.utils.data.DataLoader(big_small_datasets['medium'],
                                batch_size=32,
                                shuffle=True,
                                num_workers=4),
    'big':
    torch.utils.data.DataLoader(big_small_datasets['big'],
                                batch_size=32,
                                shuffle=True,
                                num_workers=4)
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
