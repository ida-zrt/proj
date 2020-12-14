# TODO: plot wrong classifications
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from image_dataset import dataloaders, class_names
import configs as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.ion()


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated


inputs, classes = iter(dataloaders['train']).next()

out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])


def visualize_model(model, num_images=6, model_name=cfg.model_name):
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
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    break

            if images_so_far == num_images:
                break

        model.train(mode=was_training)
    fig.savefig('{}_test.png'.format(model_name), dpi=100)


model = cfg.getmodel(cfg.model_name, pretrained=False, local_weight=True)
model = model.to(device)
visualize_model(model)
