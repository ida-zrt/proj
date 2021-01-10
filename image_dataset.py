import torch
from torchvision import datasets, transforms
import configs as cfg
import cv2 as cv
import glob
import numpy as np
import os

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

# remove old pyrdown images
try:
    old_pyd = np.load('pydImage.npy').tolist()
    for name in old_pyd:
        try:
            os.remove(name)
        except:
            pass
except:
    pass

# ImageFolder: label is the folder name, suitable for classification
data_dir = cfg.image_folder

# pyrdown operation to make image pyrmiads
if cfg.pydaction:
    temp = []
    for cls in ['normal', 'phone', 'smoke']:
        for name in glob.glob(data_dir + cls + '/*'):
            img = cv.imread(name)
            imgSize = img.shape[0] * img.shape[1]
            actcode = np.random.randint(0, 4)
            if actcode == 0 or imgSize < 224 * 224:
                continue
            else:
                for _ in range(actcode):
                    # img2 = cv.pyrDown(img)
                    pass
            newname = name[:-4] + f'_{actcode}.jpg'
            # cv.imwrite(newname, img2)
            temp.append(newname)
            print(f'pyrdown level {actcode}, saved to {newname}')

    np.save('pydImage', temp)

image_datasets = {
    'train':
    datasets.ImageFolder(data_dir + 'train',
                         transform=data_transforms['train']),
    'val':
    datasets.ImageFolder(data_dir + 'val', transform=data_transforms['val'])
}

big_small_datasets = {
    'small':
    datasets.ImageFolder(data_dir + 'small', transform=data_transforms['val']),
    'medium':
    datasets.ImageFolder(data_dir + 'medium',
                         transform=data_transforms['val']),
    'big':
    datasets.ImageFolder(data_dir + 'big', transform=data_transforms['val'])
}

big_small_dataloaders = {
    'small':
    torch.utils.data.DataLoader(big_small_datasets['small'],
                                batch_size=32,
                                shuffle=False,
                                num_workers=4),
    'medium':
    torch.utils.data.DataLoader(big_small_datasets['medium'],
                                batch_size=32,
                                shuffle=False,
                                num_workers=4),
    'big':
    torch.utils.data.DataLoader(big_small_datasets['big'],
                                batch_size=32,
                                shuffle=False,
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
                                shuffle=False,
                                num_workers=4),
    'val_random':
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
