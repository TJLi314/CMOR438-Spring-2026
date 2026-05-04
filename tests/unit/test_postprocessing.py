import numpy as np
import pytest

from rice_ml import accuracy_score, confusion_matrix


# ----------------------------
# accuracy_score Tests
# ----------------------------

def test_accuracy_score_perfect():
    y_true = np.array([1, -1, 1, -1])
    y_pred = np.array([1, -1, 1, -1])

    assert accuracy_score(y_true, y_pred) == 1.0


def test_accuracy_score_partial():
    y_true = np.array([1, -1, 1, -1])
    y_pred = np.array([1, 1, 1, -1])

    assert accuracy_score(y_true, y_pred) == 0.75


def test_accuracy_score_all_wrong():
    y_true = np.array([1, 1])
    y_pred = np.array([-1, -1])

    assert accuracy_score(y_true, y_pred) == 0.0


# ----------------------------
# confusion_matrix Tests
# ----------------------------

def test_confusion_matrix_binary():
    y_true = np.array([1, 1, -1, -1])
    y_pred = np.array([1, -1, 1, -1])

    cm = confusion_matrix(y_true, y_pred, labels=[1, -1])

    expected = np.array([
        [1, 1],  # true 1
        [1, 1]   # true -1
    ])

    assert np.array_equal(cm, expected)


def test_confusion_matrix_multiclass():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 2, 1, 1, 0])

    cm = confusion_matrix(y_true, y_pred)

    expected = np.array([
        [2, 0, 0],
        [0, 1, 1],
        [0, 1, 0]
    ])

    assert np.array_equal(cm, expected)


def test_confusion_matrix_labels_order():
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    expected = np.array([
        [0, 1],
        [1, 0]
    ])

    assert np.array_equal(cm, expected)


def test_confusion_matrix_single_class():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])

    cm = confusion_matrix(y_true, y_pred)

    assert cm.shape == (1, 1)
    assert cm[0, 0] == 3