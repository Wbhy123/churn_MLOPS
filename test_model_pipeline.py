"""Unit tests for every function of model_pipeline."""
import pandas as pd
import pytest

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


@pytest.fixture
def csv_path(tmp_path):
    """Create a small fake churn CSV and return its path."""
    n = 50
    df = pd.DataFrame({
        "RowNumber": range(n),
        "CustomerId": range(n),
        "Surname": ["x"] * n,
        "CreditScore": [600 + i for i in range(n)],
        "Geography": ["France"] * n,
        "Gender": ["Male", "Female"] * (n // 2),
        "Age": [30 + i % 20 for i in range(n)],
        "Exited": [0, 1] * (n // 2),
    })
    path = tmp_path / "fake.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def data(csv_path):
    """Prepared data: x_train, x_test, y_train, y_test."""
    return prepare_data(csv_path)


@pytest.fixture
def model(data):
    """A trained model."""
    x_train, _, y_train, _ = data
    return train_model(x_train, y_train)


def test_prepare_data_split(data):
    """The split is 80% train and 20% test."""
    x_train, x_test, y_train, y_test = data
    assert len(x_train) == 40 and len(x_test) == 10
    assert len(y_train) == 40 and len(y_test) == 10


def test_prepare_data_columns(data):
    """Unused columns are removed and Gender is numeric."""
    x_train, _, _, _ = data
    for col in ["Exited", "Surname", "Geography", "RowNumber", "CustomerId"]:
        assert col not in x_train.columns
    assert pd.api.types.is_numeric_dtype(x_train["Gender"])


def test_train_model(model, data):
    """The trained model returns one prediction per test row."""
    _, x_test, _, _ = data
    assert len(model.predict(x_test)) == len(x_test)


def test_evaluate_model(model, data):
    """Accuracy is between 0 and 1 and the matrix is 2x2."""
    _, x_test, _, y_test = data
    accuracy, matrix = evaluate_model(model, x_test, y_test)
    assert 0 <= accuracy <= 1
    assert matrix.shape == (2, 2)
    assert matrix.sum() == len(y_test)


def test_save_model(model, tmp_path):
    """save_model creates the file on disk."""
    path = tmp_path / "model.joblib"
    save_model(model, str(path))
    assert path.exists()


def test_load_model(model, data, tmp_path):
    """A loaded model gives the same predictions as the saved one."""
    _, x_test, _, _ = data
    path = str(tmp_path / "model.joblib")
    save_model(model, path)
    loaded = load_model(path)
    assert list(loaded.predict(x_test)) == list(model.predict(x_test))
