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


def visualize_model(model_name=cfg.model_name, num_images=6):
    model = torch.load(f'./weights/{model_name}_best.pth')
    model.to(cfg.device)
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


def visualize_model_err(model_name=cfg.model_name, num_images=6):
    model = torch.load(f'./weights/{model_name}_best.pth')
    model.to(cfg.device)
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

            index = (preds.to('cpu').numpy() != labels.data)

            err_images = inputs.cpu()[index, :, :, :]

            for j in range(err_images.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(err_images.cpu().data[j])

                if images_so_far == num_images:
                    break

            if images_so_far == num_images:
                break

        model.train(mode=was_training)
    fig.savefig('{}_err_img.png'.format(model_name), dpi=100)


# 模型可视化
for model_name in cfg.pretrained_weights_path_dict.keys():
    visualize_model(model_name)
    visualize_model_err(model_name)
