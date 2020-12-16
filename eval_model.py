# give model results (raw results)
import numpy as np
import configs as cfg

labels = np.load('./results/labels.npy')
for model_name in cfg.pretrained_weights_path_dict.keys():
    if cfg.debugMode:
        model_name = "resnet18_testmodel"
        n = 2  # num of classes
    else:
        n = 3
    results = np.load(f'./results/{model_name}_results.npy')

    # put your draw codes here:

    # save picture
    save_name_ROC = f'./results/{model_name}_ROC.png'
    save_name_PR = f'./results/{model_name}_PR.png'

    if cfg.debugMode:
        break
