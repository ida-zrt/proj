# TODO: add train curves

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import torch
from image_dataset import dataloaders, dataset_sizes
import configs as cfg

device = cfg.device
num_epochs = cfg.num_epochs


def train_model(model,
                criterion,
                optimizer,
                scheduler,
                num_epochs=num_epochs,
                model_name=cfg.model_name):
    since = time.time()

    best_model = copy.deepcopy(model)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

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
                best_model = copy.deepcopy(model)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:.4f}'.format(best_acc))

    torch.save(model, './weights/{}_last.pth'.format(model_name))
    torch.save(best_model, './weights/{}_best.pth'.format(model_name))
    return model


# ************************************************************************* #
# freeze=All - lastLayer, tune = lastLayer
# changed some layers to improve performace
model_name = cfg.model_name

model = cfg.getmodel(model_name=model_name,
                     pretrained=True,
                     local_weight=True,
                     weights='./weights/resnet18-5c106cde.pth')

# custom configurations:
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential()
model.fc.add_module('fc1', nn.Linear(num_ftrs, 256))

model.fc.add_module('dropout', nn.Dropout(0.5))

model.fc.add_module('fc2', nn.Linear(256, 2))

model = model.to(device)

# loss func, optimizers and hyper-params
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model = train_model(model,
                    criterion,
                    optimizer_conv,
                    exp_lr_scheduler,
                    num_epochs=cfg.num_epochs)
