import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.utils import resample as bootstrap_resample


def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy.

    :param y_true: true labels
    :param y_pred: predicted labels
    """
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def compute_precision(y_true, y_pred, average='macro', zero_division=0):
    """
    Compute precision.

    :param y_true: true labels
    :param y_pred: predicted labels
    """
    return precision_score(y_true=y_true, y_pred=y_pred, average=average, zero_division=zero_division)


def compute_recall(y_true, y_pred, average='macro'):
    """
    Compute recall.

    :param y_true: true labels
    :param y_pred: predicted labels
    """
    return recall_score(y_true=y_true, y_pred=y_pred, average=average)


def compute_f1(y_true, y_pred, average='macro'):
    """
    Compute f1 score.

    :param y_true: true labels
    :param y_pred: predicted labels
    """
    return f1_score(y_true=y_true, y_pred=y_pred, average=average)


def compute_confusion_matrix(y_true, y_pred, label_mapping):
    """
    Compute confusion matrix.

    :param y_true: true labels
    :param y_pred: predicted labels
    """
    class_labels = [label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])]
    targets_mapped = [class_labels[target] for target in y_true]
    predicts_mapped = [class_labels[predict] for predict in y_pred]
    cm = confusion_matrix(targets_mapped, predicts_mapped, labels=class_labels)
    return cm, class_labels


def compute_specificity(cm, label_mapping):
    """
    Compute specificity

    :param cm: confusion matrix
    :param label_mapping: mapping from label to class name
    """
    specificity = {}
    for i, label in enumerate(label_mapping):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tp = cm[i, i]
        specificity[label] = round(tn / (tn + fp), 4)
    return specificity


def compute_roc_auc(y_true, y_pred, num_classes):
    """
    Compute ROC-AUC score

    :param y_true: true labels
    :param y_pred: predicted probabilities
    :param num_classes: number of classes
    """
    targets_one_hot = np.eye(num_classes)[y_true]

    fpr, tpr = {}, {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def calculate_metrics_statistics(metrics_list):
    """
    Calculate statistics of metrics

    :param metrics_list: [(accuracy, precision, recall, f1 score), (...), ...]
    """
    accuracies = [metrics['Accuracy'] for metrics in metrics_list]
    precisions = [metrics['Precision'] for metrics in metrics_list]
    recalls = [metrics['Recall'] for metrics in metrics_list]
    f1_scores = [metrics['F1 Score'] for metrics in metrics_list]

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    mean_precision = np.mean(precisions)
    std_precision = np.std(precisions)

    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)

    mean_f1_score = np.mean(f1_scores)
    std_f1_score = np.std(f1_scores)

    metrics_statistics = {
        'Accuracy': f"{mean_accuracy:.3f} ± {std_accuracy:.2f}",
        'Precision': f"{mean_precision:.3f} ± {std_precision:.2f}",
        'Recall': f"{mean_recall:.3f} ± {std_recall:.2f}",
        'F1 Score': f"{mean_f1_score:.3f} ± {std_f1_score:.2f}"
    }

    return metrics_statistics


def calculate_bootstrap_ci(data, n_bootstraps=1000, alpha=0.05, random_seed=3407):
    """
    Calculate bootstrap confidence interval.
    """
    if len(data) < 2:
        return np.nan, np.nan
    if random_seed:
        np.random.seed(random_seed)
    bootstrap_means = []
    for _ in range(n_bootstraps):
        sample = bootstrap_resample(data)
        bootstrap_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrap_means, 100 * (alpha / 2.0))
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2.0))
    return lower_bound, upper_bound
