# 采用字典组织数据类型，可以提升代码可读性
# 可以利用字典将traindataset, testdataset组合在一起

# imageFolder: root/classname/image.jpg
# imageFolder(root, transform)
# 它的常用属性
# classes (list): List of the class names.
# class_to_idx (dict): Dict with items (class_name, class_index).
# imgs (list): List of (image path, class_index) tuples

# lr_scheduler: learning rate scheduler
# 貌似是可以减小lr

# num_epoches: 指的是遍历所有数据的次数

# with torch.set_grad_enabled( mode = phase == 'train')
# 此代码将根据 phase=='train'的情况，确定是否启动梯度计算
# 因此: 可以用一个变量 phase 取 ['train','val']
# 从而模型可以边训练，边eval，通常在便利完一次数据集之后eval
# 并计算 epoch_loss, epoch_acc

# model.train() 同样也可以接受一个bool参数
# model.training 可以记录模型是否正在训练
# 因此可以做到实时的模式转换

# copy.deepcopy(model.state_dict()) 可以用于在内存中保存模型参数

# maxval, idx = torch.max(tensor, dim)同时返回最大值和索引值

# 模型在定义的时候，一般在init函数中写好模型需要的各层名称，
# forward函数中写好如何传递模型
# 这样做的好处，是写好模型后可以直接通过 属性访问 修改某一层的参数

# 使用to(device) 确保代码可用性

# 迁移学习: 用已有的weight初始化, freeze一部分, tune一部分

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()

data_transforms = {
    'train': transforms.Compose(
        [transforms.Resize(256),
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])]),
    'val': transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])])
}

data_dir = 'data/hymenoptera_data/'
image_datasets = {
    'train': datasets.ImageFolder('data/hymenoptera_data/train/', transform=data_transforms['train']),
    'val': datasets.ImageFolder('data/hymenoptera_data/val/', transform=data_transforms['val'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'],
                                         batch_size=32, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets['val'],
                                       batch_size=4, shuffle=True, num_workers=4)
}

dataset_sizes = {'train': len(image_datasets['train']),
                 'val': len(image_datasets['val'])}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


inputs, classes = iter(dataloaders['train']).next()

out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed//60, time_elapsed % 60))

    print('Best val Acc: {:.4f}'.format(best_acc))

    torch.save(model.state_dict(), './weights/last.pth')
    torch.save(best_model_wts, './weights/best.pth')
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# # load resnet18 模型，并修改最后一层 output为2
# # freeze=None, tune=ALL
# model_ft = models.resnet18(pretrained=False)
# model_ft.load_state_dict(torch.load('weights/resnet18-5c106cde.pth'))
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, 2)

# model_ft = model_ft.to(device)

# criterion = nn.CrossEntropyLoss()

# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=25)

# torch.save(model_ft.state_dict(), './weights/model_ft_last.pth')
# model_ft.load_state_dict(torch.load('weights/best.pth'))
# torch.save(model_ft.state_dict(), './weights/model_ft_best.pth')
# model_ft = model_ft.to(device)
# visualize_model(model_ft)

# ************************************************************************* #
# freeze=All - lastLayer, tune = lastLayer
# changed some layers to improve performace
model_conv = torchvision.models.resnet18(pretrained=False)
model_conv.load_state_dict(torch.load('./weights/resnet18-5c106cde.pth'))
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Sequential()
model_conv.fc.add_module(
    'fc1', nn.Linear(num_ftrs, 256)
)

model_conv.fc.add_module(
    'dropout', nn.Dropout(0.5)
)

model_conv.fc.add_module(
    'fc2', nn.Linear(256, 2)
)

model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

torch.save(model_conv.state_dict(), './weights/model_conv_last.pth')
model_conv.load_state_dict(torch.load('weights/best.pth'))
torch.save(model_conv.state_dict(), './weights/model_conv_best.pth')
