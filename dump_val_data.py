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
import configs as cfg
from image_dataset import dataloaders

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
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = torch.nn.functional.one_hot(labels, n)

            outputs = trainedModel(inputs)
            outputList.append(outputs.to('cpu').numpy())
            labelList.append(labels.to('cpu').numpy())
    
    outputList = np.concatenate(outputList, axis=0)
    labelList = np.concatenate(labelList, axis=0)
    np.save(f'./results/{model_name}_results', outputList)
    if not os.path.exists('./results/labels.npy'):
        np.save('./results/labels', labelList)

    if cfg.debugMode:
        break
