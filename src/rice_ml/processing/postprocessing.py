import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.

    Args:
        y_true (np.ndarray)
        y_pred (np.ndarray)

    Returns:
        float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix.

    Args:
        y_true (np.ndarray)
        y_pred (np.ndarray)
        labels (list or None): specify label order

    Returns:
        np.ndarray (n_classes x n_classes)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))

    label_to_index = {label: i for i, label in enumerate(labels)}
    n_classes = len(labels)

    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        i = label_to_index[true]
        j = label_to_index[pred]
        cm[i, j] += 1

    return cm