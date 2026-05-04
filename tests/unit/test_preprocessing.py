import numpy as np
import pytest

from rice_ml import StandardScaler, train_test_split, LabelEncoder


# ----------------------------
# StandardScaler Tests
# ----------------------------

def test_standard_scaler_fit():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    scaler.fit(X)

    assert np.allclose(scaler.mean_, np.array([3, 4]))
    assert np.allclose(scaler.std_, np.array([np.std([1,3,5]), np.std([2,4,6])]))


def test_standard_scaler_transform():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Mean should be ~0
    assert np.allclose(np.mean(X_scaled, axis=0), np.zeros(2))

    # Std should be ~1
    assert np.allclose(np.std(X_scaled, axis=0), np.ones(2))


def test_standard_scaler_constant_feature():
    X = np.array([[1, 2], [1, 3], [1, 4]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # First column constant → should become zeros
    assert np.allclose(X_scaled[:, 0], 0)


def test_standard_scaler_transform_without_fit():
    scaler = StandardScaler()
    X = np.array([[1, 2]])

    with pytest.raises(Exception):
        scaler.transform(X)


# ----------------------------
# train_test_split Tests
# ----------------------------

def test_train_test_split_shapes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_train_test_split_no_shuffle():
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # No shuffle → first part should be test
    assert np.array_equal(X_test.flatten(), np.array([0, 1, 2]))
    assert np.array_equal(X_train.flatten(), np.array([3,4,5,6,7,8,9]))


def test_train_test_split_reproducibility():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    split1 = train_test_split(X, y, random_state=42)
    split2 = train_test_split(X, y, random_state=42)

    for a, b in zip(split1, split2):
        assert np.array_equal(a, b)


def test_train_test_split_stratify():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0]*5 + [1]*5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    # Check class balance
    assert np.sum(y_train == 0) == 3
    assert np.sum(y_train == 1) == 3
    assert np.sum(y_test == 0) == 2
    assert np.sum(y_test == 1) == 2
    
def test_label_encoder_fit():
    y = np.array(["cat", "dog", "cat", "bird"])

    le = LabelEncoder()
    le.fit(y)

    assert set(le.classes_) == {"cat", "dog", "bird"}
    assert len(le.classes_) == 3


def test_label_encoder_transform():
    y = np.array(["cat", "dog", "cat", "bird"])

    le = LabelEncoder()
    le.fit(y)

    encoded = le.transform(y)

    # Ensure deterministic mapping
    assert len(np.unique(encoded)) == 3
    assert set(encoded) == {0, 1, 2}


def test_label_encoder_fit_transform_equivalence():
    y = np.array(["a", "b", "a", "c"])

    le = LabelEncoder()

    a = le.fit_transform(y)

    le2 = LabelEncoder()
    le2.fit(y)
    b = le2.transform(y)

    assert np.array_equal(a, b)


def test_label_encoder_inverse_transform():
    y = np.array(["red", "blue", "green", "red"])

    le = LabelEncoder()
    encoded = le.fit_transform(y)
    decoded = le.inverse_transform(encoded)

    assert np.array_equal(decoded, y)


def test_label_encoder_transform_before_fit():
    le = LabelEncoder()
    y = np.array(["a", "b"])

    with pytest.raises(ValueError):
        le.transform(y)


def test_label_encoder_inverse_before_fit():
    le = LabelEncoder()

    with pytest.raises(ValueError):
        le.inverse_transform(np.array([0, 1]))


def test_label_encoder_consistency():
    y = np.array(["x", "y", "z", "x", "y"])

    le = LabelEncoder()
    encoded = le.fit_transform(y)
    decoded = le.inverse_transform(encoded)

    assert np.array_equal(decoded, y)