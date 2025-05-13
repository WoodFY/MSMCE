import os
import joblib

from utils.metrics import compute_accuracy, compute_precision, compute_recall, compute_f1, compute_confusion_matrix, compute_specificity
from utils.metrics_visualization import plot_confusion_matrix, plot_roc_auc_curve


def train_test_ml(exp_dir, exp_model_name, model, train_set, test_set, label_mapping, metrics_visualization=True):
    """
    Train a machine learning model and evaluate it on the test set.
    """
    X_train, y_train = train_set
    X_test, y_test = test_set

    model.fit(X_train, y_train)
    save_model_path = os.path.join(exp_dir, f'{exp_model_name}.pkl')
    joblib.dump(model, save_model_path)

    y_pred = model.predict(X_test)

    test_accuracy = compute_accuracy(y_test, y_pred)
    test_precision = compute_precision(y_test, y_pred)
    test_recall = compute_recall(y_test, y_pred)
    test_f1 = compute_f1(y_test, y_pred)
    print('==========================================================================================')
    print('Test Acc: {:.4f} | Test Precision: {:.4f} | Test Recall (Sensitivity): :{:.4f} | Test F1: {:.4f}'.format(
        test_accuracy, test_precision, test_recall, test_f1)
    )

    confusion_matrix, _ = compute_confusion_matrix(y_test, y_pred, label_mapping)
    # print(f'Confusion Matrix: \n{matrix}')
    specificity = compute_specificity(confusion_matrix, label_mapping)
    print(f'Specificity for each class: {specificity}')
    print('==========================================================================================')

    if metrics_visualization:
        plot_confusion_matrix(y_test, y_pred, label_mapping, exp_model_name, exp_dir, cm=confusion_matrix)
        plot_roc_auc_curve(y_test, model.predict_proba(X_test), label_mapping, exp_model_name, exp_dir)

    return test_accuracy, test_precision, test_recall, test_f1