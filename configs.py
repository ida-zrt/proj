import torch
import torchvision

image_folder = './data/hymenoptera_data/'

train_val_split = [0.8, 0.2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 10

# a model in torchvision package
# test it with torchvision.models.model_name(pretrained)
model_name = 'resnet18'
defaut_weight = './weights/{}_best.pth'.format(model_name)


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
