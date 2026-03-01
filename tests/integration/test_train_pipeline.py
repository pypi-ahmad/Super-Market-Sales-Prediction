import os
import pickle

import numpy as np
import pandas as pd

import train_automl
from tests.helpers import FakeAutoML


def _sample_training_dataframe(rows=20):
    data = {
        "Invoice ID": [f"INV-{i}" for i in range(rows)],
        "Branch": ["A", "B", "C", "A", "B"] * (rows // 5),
        "City": ["Yangon", "Naypyitaw", "Mandalay", "Yangon", "Naypyitaw"] * (rows // 5),
        "Customer type": ["Member", "Normal", "Member", "Normal", "Member"] * (rows // 5),
        "Gender": ["Female", "Male", "Female", "Male", "Female"] * (rows // 5),
        "Product line": ["Health and beauty", "Food and beverages", "Electronic accessories", "Sports and travel", "Home and lifestyle"] * (rows // 5),
        "Unit price": [10.0 + i for i in range(rows)],
        "Quantity": [1 + (i % 5) for i in range(rows)],
        "Tax 5%": [1.0 + i * 0.1 for i in range(rows)],
        "Date": ["1/1/2019"] * rows,
        "Time": ["10:00"] * rows,
        "Payment": ["Cash", "Ewallet", "Credit card", "Cash", "Ewallet"] * (rows // 5),
        "cogs": [20.0 + i for i in range(rows)],
        "gross margin percentage": [4.761904762] * rows,
        "gross income": [1.0 + i * 0.1 for i in range(rows)],
        "Rating": [5.0 + (i % 5) for i in range(rows)],
    }
    df = pd.DataFrame(data)
    df["Total"] = df["cogs"] + df["Tax 5%"]
    return df


def test_training_pipeline_creates_expected_artifact(tmp_path, monkeypatch):
    source_path = tmp_path / "train.csv"
    model_path = tmp_path / "automl_model.pkl"

    df = _sample_training_dataframe(rows=20)
    df.to_csv(source_path, index=False)

    monkeypatch.setattr(train_automl, "DATA_SOURCE", str(source_path))
    monkeypatch.setattr(train_automl, "MODEL_FILE", str(model_path))
    monkeypatch.setattr(train_automl, "AutoML", FakeAutoML)

    train_automl.main()

    assert model_path.exists()

    with open(model_path, "rb") as file_obj:
        artifacts = pickle.load(file_obj)

    assert set(artifacts.keys()) == {"model", "features", "column_metadata", "metrics", "config"}
    assert "Total" not in artifacts["features"]
    for leak_col in ["Invoice ID", "Tax 5%", "cogs", "gross income", "gross margin percentage", "Date", "Time"]:
        assert leak_col not in artifacts["features"], f"leakage column {leak_col!r} still in features"
    assert set(artifacts["metrics"].keys()) == {"r2", "mae", "mse"}
    assert artifacts["config"]["target"] == "Total"


def test_training_pipeline_missing_target_does_not_write_model(tmp_path, monkeypatch):
    source_path = tmp_path / "train_missing_target.csv"
    model_path = tmp_path / "automl_model.pkl"

    df = _sample_training_dataframe(rows=20).drop(columns=["Total"])
    df.to_csv(source_path, index=False)

    monkeypatch.setattr(train_automl, "DATA_SOURCE", str(source_path))
    monkeypatch.setattr(train_automl, "MODEL_FILE", str(model_path))
    monkeypatch.setattr(train_automl, "AutoML", FakeAutoML)

    train_automl.main()

    assert not model_path.exists()


def test_training_pipeline_missing_source_does_not_write_model(tmp_path, monkeypatch):
    source_path = tmp_path / "missing.csv"
    model_path = tmp_path / "automl_model.pkl"

    monkeypatch.setattr(train_automl, "DATA_SOURCE", str(source_path))
    monkeypatch.setattr(train_automl, "MODEL_FILE", str(model_path))
    monkeypatch.setattr(train_automl, "AutoML", FakeAutoML)

    train_automl.main()

    assert not model_path.exists()


def test_training_pipeline_filters_nan_rows(tmp_path, monkeypatch):
    """Phase-5 fix: rows with NaN target/features are dropped, training still succeeds."""
    source_path = tmp_path / "train_nan.csv"
    model_path = tmp_path / "automl_model.pkl"

    df = _sample_training_dataframe(rows=20)
    # Inject NaN into target and feature columns
    df.loc[0, "Total"] = np.nan
    df.loc[1, "Unit price"] = np.nan
    df.to_csv(source_path, index=False)

    monkeypatch.setattr(train_automl, "DATA_SOURCE", str(source_path))
    monkeypatch.setattr(train_automl, "MODEL_FILE", str(model_path))
    monkeypatch.setattr(train_automl, "AutoML", FakeAutoML)

    train_automl.main()

    assert model_path.exists(), "Model should still be created after NaN rows are filtered"

    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
    # Pipeline should have completed with valid metrics
    assert all(np.isfinite(v) for v in artifacts["metrics"].values())


def test_training_pipeline_propagates_gpu_flag(tmp_path, monkeypatch):
    """Phase-5 fix: USE_GPU env var is forwarded to AutoML settings."""
    source_path = tmp_path / "train.csv"
    model_path = tmp_path / "automl_model.pkl"

    df = _sample_training_dataframe(rows=20)
    df.to_csv(source_path, index=False)

    monkeypatch.setattr(train_automl, "DATA_SOURCE", str(source_path))
    monkeypatch.setattr(train_automl, "MODEL_FILE", str(model_path))
    monkeypatch.setattr(train_automl, "AutoML", FakeAutoML)
    monkeypatch.setenv("USE_GPU", "1")

    train_automl.main()

    assert model_path.exists()
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
    # FakeAutoML stores the settings passed to fit()
    assert artifacts["model"]._fit_settings["use_gpu"] is True