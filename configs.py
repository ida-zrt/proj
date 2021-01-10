import torch
import torchvision
debugMode = False
if debugMode:
    image_folder = './data/hymenoptera_data/'
else:
    image_folder = './data/smoke_data/'

train_val_split = [0.8, 0.2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
num_epochs = 50

# a model in torchvision package
# test it with torchvision.models.model_name(pretrained)
model_name = 'resnet18'
defaut_weight = './weights/{}_best.pth'.format(model_name)
pydaction = False

# model zoo
pretrained_weights_path_dict = {
    'mobilenet_v2':
    './weights/pretrained_weights/mobilenet_v2/mobilenet_v2-b0353104.pth',
    'resnet18':
    './weights/pretrained_weights/resnet/resnet18-5c106cde.pth',
    'resnet34':
    './weights/pretrained_weights/resnet/resnet34-333f7ec4.pth',
    # 'resnet50':
    # './weights/pretrained_weights/resnet/resnet50-19c8e357.pth',
    # 'resnet101':
    # './weights/pretrained_weights/resnet/resnet101-5d3b4d8f.pth',
    # 'resnet152':
    # './weights/pretrained_weights/resnet/resnet152-b121ed2d.pth',
    # 'resnext50_32x4d':
    # './weights/pretrained_weights/resnext/resnext50_32x4d-7cdf4587.pth',
    # 'resnext101_32x8d':
    # './weights/pretrained_weights/resnext/resnext101_32x8d-8ba56ff5.pth'
}

noFreezeList = [
    'resnet18', 'resnet34', 'resnet50', 'mobilenet_v2', 'resnext50_32x4d'
]


def getmodel(model_name=model_name,
             pretrained=False,
             local_weight=False,
             weights=defaut_weight):
    if local_weight:
        if pretrained:
            model = getattr(torchvision.models, model_name)(pretrained=False)
            model.load_state_dict(torch.load(weights))
        else:
            model = torch.load(weights)

    else:
        model = getattr(torchvision.models, model_name)(pretrained)
    return model
