import os
import torch
import numpy as np

from tqdm import tqdm

from metrics import compute_accuracy, compute_precision, compute_recall, compute_f1, compute_confusion_matrix, compute_specificity
from utils.metrics_visualization import plot_metrics, plot_confusion_matrix, plot_roc_auc_curve


def train(
        model,
        exp_dir,
        exp_name,
        train_loader,
        valid_loader,
        criterion,
        optimizers,
        schedulers,
        early_stopping,
        epochs,
        device,
        is_early_stopping=False,
        is_metrics_visualization=True
):
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    min_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        all_train_loss = []
        all_train_accuracy = []
        train_progress_bar = tqdm(train_loader, desc='Training'.ljust(10), leave=True)
        for spectra, labels in train_progress_bar:
            spectra, labels = spectra.to(device), labels.to(device)
            for optimizer in optimizers:
                optimizer.zero_grad()

            predicts = model(spectra)
            loss = criterion(predicts, labels)

            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            all_train_loss.append(loss.item())
            accuracy = compute_accuracy(labels.cpu().numpy(), predicts.argmax(dim=1).cpu().numpy())
            all_train_accuracy.append(accuracy)

            train_progress_bar.set_description('Epoch: {} | Loss: {:.4f} | Accuracy: {:.4f}'.format(epoch, loss.item(), accuracy))

        model.eval()
        all_valid_loss = []
        all_valid_accuracy = []
        valid_progress_bar = tqdm(valid_loader, desc='Validation'.ljust(10), leave=True)
        with torch.no_grad():
            for spectra, labels in valid_progress_bar:
                spectra, labels = spectra.to(device), labels.to(device)

                predicts = model(spectra)
                loss = criterion(predicts, labels)

                all_valid_loss.append(loss.item())
                accuracy = compute_accuracy(labels.cpu().numpy(), predicts.argmax(dim=1).cpu().numpy())
                all_valid_accuracy.append(accuracy)

                valid_progress_bar.set_description('Epoch: {} | Loss: {:.4f} | Accuracy: {:.4f}'.format(epoch, loss.item(), accuracy))

        train_loss = np.mean(all_train_loss)
        train_accuracy = np.mean(all_train_accuracy)
        valid_loss = np.mean(all_valid_loss)
        valid_accuracy = np.mean(all_valid_accuracy)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        print('Epoch: {} | Train Loss: {:.4f} | Train Acc: {:.4f} | Valid Loss: {:.4f} | Valid Acc: {:.4f}'.format(
            epoch, train_loss, train_accuracy, valid_loss, valid_accuracy))

        # scheduler.step(valid_loss)
        for scheduler in schedulers:
            scheduler.step(valid_loss)

        if is_early_stopping:
            early_stopping(valid_loss, model, save_model_dir=exp_dir, save_model_name=exp_name)
            if early_stopping.early_stop:
                print('Early Stopping.')

                if is_metrics_visualization:
                    metrics = [
                        (train_losses, valid_losses),
                        (train_accuracies, valid_accuracies)
                    ]
                    plot_metrics(metrics, ['Loss', 'Accuracy'], exp_name, exp_dir)

                break

        else:
            save_model_path = os.path.join(exp_dir, exp_name)

            if valid_loss < min_loss:
                min_loss = valid_loss
                torch.save(model.state_dict(), str(save_model_path))

    if is_metrics_visualization:
        metrics = [
            (train_losses, valid_losses),
            (train_accuracies, valid_accuracies),
        ]
        plot_metrics(metrics, ['Loss', 'Accuracy'], exp_name, exp_dir)


def test(
        model,
        exp_dir,
        exp_name,
        test_loader,
        criterion,
        label_mapping,
        device,
        metric_args=None,
        is_metrics_visualization=True
):
    save_model_path = os.path.join(exp_dir, exp_name)

    if save_model_path is None:
        raise FileNotFoundError('No model checkpoint file found in the directory.')

    model.load_state_dict(torch.load(str(save_model_path)))
    model.eval()

    all_test_loss = []
    all_labels = []
    all_predicts = []
    all_predicts_proba = []
    test_progress_bar = tqdm(test_loader, desc='Testing'.ljust(10), leave=True)
    with torch.no_grad():
        for spectra, labels in test_progress_bar:
            spectra, labels = spectra.to(device), labels.to(device)

            predicts = model(spectra)
            loss = criterion(predicts, labels)

            all_test_loss.append(loss.item())
            accuracy = compute_accuracy(labels.cpu().numpy(), predicts.argmax(dim=1).cpu().numpy())

            all_labels.extend(labels.cpu().numpy())
            all_predicts.extend(predicts.argmax(dim=1).cpu().numpy())
            all_predicts_proba.extend(torch.softmax(predicts, dim=1).cpu().numpy())

            test_progress_bar.set_description('Loss: {:.4f} | Accuracy: {:.4f}'.format(loss.item(), accuracy))

    test_loss = np.mean(all_test_loss)
    test_accuracy = compute_accuracy(all_labels, all_predicts)
    test_precision = compute_precision(all_labels, all_predicts)
    test_recall = compute_recall(all_labels, all_predicts)
    test_f1 = compute_f1(all_labels, all_predicts)
    print('Test Loss: {:.4f} | Test Acc: {:.4f}'.format(test_loss, test_accuracy))
    print('Test Precision: {:.4f} | Test Recall (Sensitivity): :{:.4f} | Test F1: {:.4f}'.format(test_precision, test_recall, test_f1))

    confusion_matrix, _ = compute_confusion_matrix(all_labels, all_predicts, label_mapping)
    # print(f'Confusion Matrix: \n{matrix}')

    specificity = compute_specificity(confusion_matrix, label_mapping)
    print(f'Specificity for each class: {specificity}')

    if is_metrics_visualization:
        plot_confusion_matrix(all_labels, all_predicts, label_mapping, exp_name, exp_dir, cm=confusion_matrix)
        plot_roc_auc_curve(np.array(all_labels), np.array(all_predicts_proba), label_mapping, exp_name, exp_dir)

    return test_accuracy, test_precision, test_recall, test_f1