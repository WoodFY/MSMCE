import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

from exp.metrics import compute_confusion_matrix, compute_roc_auc
from utils.data_loader import get_file_paths


def plot_metrics(metrics, titles, plot_title, save_dir):
    n = len(metrics)
    plt.figure(figsize=(15, 5 * (n // 2 + n % 2)))
    for i, (train_metrics, valid_metrics) in enumerate(metrics):
        plt.subplot(n // 2 + n % 2, 2, i + 1)
        plt.plot(train_metrics, label='Train')
        plt.plot(valid_metrics, label='Valid')
        plt.title(titles[i])
        plt.xlabel('Epoch')
        plt.ylabel(titles[i])
        plt.legend()
    plt.suptitle(plot_title.split('.')[0])
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    metrics_file_path = os.path.join(save_dir, f"{plot_title.split('.')[0]}_train_valid_metrics_plot_{time_stamp}.png")
    plt.savefig(metrics_file_path)
    plt.show()
    plt.close()


def plot_confusion_matrix(targets, predicts, label_mapping, plot_title, save_dir, cm=None):
    if cm is None:
        cm, class_labels = compute_confusion_matrix(targets, predicts, label_mapping)
    else:
        class_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]

    # visualize confusion matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.title(f"{plot_title.split('.')[0]} Confusion Matrix")

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    cm_file_path = os.path.join(save_dir, f"{plot_title.split('.')[0]}_confusion_matrix_plot_{time_stamp}.png")
    plt.savefig(cm_file_path)
    plt.show()
    plt.close()


def plot_roc_auc_curve(targets, predicts, label_mapping, plot_title, save_dir):
    """
    Plot ROC curve

    Args:
        fpr: false positive rate
        tpr: true positive rate
        auc_scores: area under the curve
        num_classes: number of classes

    Returns:
        None
    """
    num_classes = len(label_mapping)
    fpr, tpr, auc_scores = compute_roc_auc(targets, predicts, num_classes)
    class_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]

    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(class_labels):
        plt.plot(fpr[i], tpr[i], label=f'{class_label} (AUC = {auc_scores[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{plot_title.split('.')[0]} ROC Curve")
    plt.legend(loc='lower right')

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    roc_auc_file_path = os.path.join(save_dir, f"{plot_title.split('.')[0]}_roc_auc_plot_{time_stamp}.png")
    plt.savefig(roc_auc_file_path)
    plt.show()
    plt.close()


def plot_boxplot(metrics_file_paths: list, metric: str, dataset_name: str, save_dir: str):
    """
    读取多个CSV文件并绘制对应模型的箱线图。

    Args:
        metrics_file_paths (list): 包含每个CSV文件路径的列表
        metric (str): 需要绘制的参数名称 (如 'Accuracy', 'Precision', 'Recall', 'F1 Score')
    """

    all_data = []

    for file_path in metrics_file_paths:
        model_name = os.path.basename(file_path).split('_')[0].replace('MLPEmbedding', 'ME-')

        # read csv file
        df = pd.read_csv(file_path)
        df['Model'] = model_name

        all_data.append(df[[metric, 'Model']])

    df_all = pd.concat(all_data)

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("hsv", len(df_all['Model'].unique()))
    # palette = sns.color_palette("coolwarm", 7)
    sns.boxplot(x='Model', y=metric, data=df_all, palette=palette)
    plt.title(f'{metric} Boxplot Across Models')
    plt.ylabel(metric)

    # 标注分区
    # plt.axvspan(4.5, 6.5, color='pink', alpha=0.2)  # 用浅红色突出最后两组模型

    # 调整x轴标签显示
    plt.xticks(rotation=45, fontsize=10)  # ha='right' 可以让标签靠右对齐，避免重叠

    # 调整底部以防标签被切掉
    plt.subplots_adjust(bottom=0.25)  # 调整底部空间，确保标签显示完全

    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    boxplot_file_path = os.path.join(save_dir, f"{dataset_name}_{metric}_boxplot_{time_stamp}.png")
    plt.savefig(boxplot_file_path)
    plt.show()
    plt.close()
