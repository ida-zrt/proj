import torch
import numpy as np
import os
import configs as cfg
from image_dataset import dataloaders, big_small_dataloaders

device = cfg.device

for model_name in cfg.pretrained_weights_path_dict.keys():
    if cfg.debugMode:
        model_name = 'resnet18_testmodel'
        n = 2
    else:
        n = 3
    trainedModel = torch.load(f'./weights/{model_name}_best.pth')
    trainedModel.to(cfg.device)

    outputList = []
    labelList = []
    running_corrects = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = trainedModel(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

            labels = torch.nn.functional.one_hot(labels, n)

            outputList.append(outputs.to('cpu').numpy())
            labelList.append(labels.to('cpu').numpy())

        epoch_acc = running_corrects.double() / len(dataloaders['val'].dataset)

    print(f'acc on all: {epoch_acc.item()}')
    outputList = np.concatenate(outputList, axis=0)
    labelList = np.concatenate(labelList, axis=0)
    np.save(f'./results/{model_name}_results', outputList)
    np.save(f'./results/{model_name}_labels', labelList)

    for set_name in ['small', 'medium', 'big']:
        outputList = []
        labelList = []
        running_corrects = 0
        with torch.no_grad():
            for i, (inputs,
                    labels) in enumerate(big_small_dataloaders[set_name]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = trainedModel(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                labels = torch.nn.functional.one_hot(labels, n)

                outputList.append(outputs.to('cpu').numpy())
                labelList.append(labels.to('cpu').numpy())

            epoch_acc = running_corrects.double() / len(
                big_small_dataloaders[set_name].dataset)

        print(f'acc on {set_name}: {epoch_acc.item()}')
        outputList = np.concatenate(outputList, axis=0)
        labelList = np.concatenate(labelList, axis=0)
        np.save(f'./results/{model_name}_{set_name}_results', outputList)
        np.save(f'./results/{model_name}_{set_name}_labels', labelList)

    if cfg.debugMode:
        break
