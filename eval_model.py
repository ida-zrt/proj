# give model results (raw results)
import numpy as np
import configs as cfg
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics
from image_dataset import class_names

for set_type in ['all', 'small', 'medium']:
    if set_type.startswith('a'):
        set_name = ''
    else:
        set_name = set_type + '_'
    if set_type == 'medium':
        set_type = 'big'
    for model_name in cfg.pretrained_weights_path_dict.keys():
        labels_x = np.load(f'./results/{model_name}_{set_name}labels.npy')
        if cfg.debugMode:
            model_name = "resnet18_testmodel"
            n = 2  # num of classes
        else:
            n = 3

        results = np.load(f'./results/{model_name}_{set_name}results.npy')

        # put your draw codes here:
        y_true = np.array(labels_x)
        y_score = np.array(results)
        # draw picture of ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_score[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            print(
                f'Auc score for class {class_names[i]} prediction: {roc_auc[i]} ({model_name} on set {set_type})'
            )

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
            y_true.ravel(), y_score.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
        print(
            f'micro-average auc score: {roc_auc["micro"]} ({model_name} on set {set_type})'
        )

        # Plot all ROC curves
        lw = 2
        plt.figure()
        plt.plot(fpr["micro"],
                 tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]),
                 color='deeppink',
                 linestyle=':',
                 linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n), colors):
            plt.plot(fpr[i],
                     tpr[i],
                     color=color,
                     lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC of {model_name} on {set_type}')
        plt.legend(loc="lower right")
        plt.savefig(f'./results/{model_name}_{set_type}_ROC.png')
        # plt.show()

        # draw picture of PR
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n):
            precision[i], recall[i], _ = metrics.precision_recall_curve(
                y_true[:, i], y_score[:, i])
            average_precision[i] = metrics.average_precision_score(
                y_true[:, i], y_score[:, i], average='macro', pos_label=1)
            print(
                f'AP score for class {i} prediction: {roc_auc[i]} ({model_name} on set {set_type})'
            )

        # 计算微定义AP
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall[
            "micro"], _ = metrics.precision_recall_curve(
                y_true.ravel(), y_score.ravel())
        average_precision["micro"] = metrics.average_precision_score(
            y_true, y_score, average="micro")
        print(
            'Average precision score, micro-averaged over all classes: {0:0.2f} ({1} on set {2})'
            .format(average_precision["micro"], model_name, set_type))

        # setup plot details
        colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []

        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append(
            'micro-average Precision-recall (area = {0:0.2f})'.format(
                average_precision["micro"]))

        for i, color in zip(range(n), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.15)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR of {model_name} on {set_type}')
        # plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
        plt.legend(lines, labels)
        plt.savefig(f'./results/{model_name}_{set_type}_PR.png')
        # plt.show()

        # save picture

        if cfg.debugMode:
            break

for set_type in ['all', 'small', 'medium']:
    if set_type.startswith('a'):
        set_name = ''
    else:
        set_name = set_type + '_'

    if set_type == 'medium':
        set_type = 'big'

    for model_name in cfg.pretrained_weights_path_dict.keys():
        labels_x = np.load(f'./results/{model_name}_{set_name}labels.npy')
        if cfg.debugMode:
            model_name = "resnet18_testmodel"
            n = 2  # num of classes
        else:
            n = 3

        results = np.load(f'./results/{model_name}_{set_name}results.npy')
        preds = np.argmax(results, axis=1)
        lb = np.argmax(labels_x, axis=1)
        corrects = np.sum(preds == lb)
        total = preds.shape[0]

        print(f'Accuracy: {corrects/total} ({model_name} on set {set_type})')

        for i in range(3):
            lb_class = lb[lb == i]
            preds_class = preds[lb == i]
            acc = np.sum(lb_class == preds_class) / preds_class.shape[0]
            print(
                f'Accuracy on class {class_names[i]}: {acc} ({model_name} on set {set_type})'
            )
