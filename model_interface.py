from PIL import Image
import torch
import numpy as np
from image_dataset import data_transforms, class_names
from configs import device
import sys

test_data = './data/smoke_data/val/smoke/6723.jpg'

image = (Image.open(test_data))

image_tensor = torch.unsqueeze(data_transforms['val'](image), 0).to(device)

model = torch.load('./weights/resnet18_testmodel_best.pth')
model = model.to(device)
with torch.no_grad:
    out = model(image_tensor)
    _, pred = torch.max(out, 1)

print(f'predicted label: {class_names[pred.cpu().data]}')
