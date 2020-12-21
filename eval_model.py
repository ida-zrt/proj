# give model results (raw results)
import numpy as np
import configs as cfg
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics
labels = np.load('./results/labels.npy')
for model_name in cfg.pretrained_weights_path_dict.keys():
    if cfg.debugMode:
        model_name = "resnet18_testmodel"
        n = 2  # num of classes
    else:
        n = 3
    results = np.load(f'./results/{model_name}_results.npy')

    # put your draw codes here:
    y_true = np.array(labels)
    y_score = np.array(results)
    # draw picture of ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('./results/{model_name}_ROC.png')
    plt.show()

    # draw picture of PR
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true[:, i], y_score[:, i])
        average_precision[i] = metrics.average_precision_score(y_true[:, i], y_score[:, i],
                                                               average='macro', pos_label=1)

    # 计算微定义AP
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_true.ravel(), y_score.ravel())
    average_precision["micro"] = metrics.average_precision_score(y_true, y_score, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    # setup plot details
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'.format(average_precision["micro"]))

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
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig('./results/{model_name}_PR.png')
    plt.show()

    # save picture
    save_name_ROC = f'./results/{model_name}_ROC.png'
    save_name_PR = f'./results/{model_name}_PR.png'

    if cfg.debugMode:
        break
