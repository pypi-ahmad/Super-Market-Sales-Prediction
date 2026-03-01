import io
import pandas as pd
import pytest

import app


def test_get_column_metadata_detects_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "city": ["A", "B", "A"],
            "amount": [10.0, 20.0, 30.0],
            "count": [1, 2, 3],
        }
    )

    metadata = app.get_column_metadata(df)

    assert metadata["city"]["type"] == "categorical"
    assert metadata["city"]["options"] == ["A", "B"]
    assert metadata["amount"]["type"] == "numeric"
    assert metadata["amount"]["min"] == 10.0
    assert metadata["amount"]["max"] == 30.0
    assert metadata["amount"]["mean"] == 20.0
    assert metadata["count"]["type"] == "numeric"


def test_get_column_metadata_empty_dataframe_returns_empty_dict():
    df = pd.DataFrame()
    metadata = app.get_column_metadata(df)
    assert metadata == {}


def test_load_data_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    expected = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    expected.to_csv(csv_path, index=False)

    with open(csv_path, "rb") as file_obj:
        loaded = app.load_data(file_obj)

    pd.testing.assert_frame_equal(loaded, expected)


def test_load_data_excel(tmp_path):
    xlsx_path = tmp_path / "sample.xlsx"
    expected = pd.DataFrame({"x": [5, 6], "y": [7, 8]})
    expected.to_excel(xlsx_path, index=False)

    with open(xlsx_path, "rb") as file_obj:
        loaded = app.load_data(file_obj)

    pd.testing.assert_frame_equal(loaded, expected)


def test_save_and_load_pretrained_model_roundtrip(tmp_path, monkeypatch):
    model_path = tmp_path / "automl_model.pkl"
    monkeypatch.setattr(app, "MODEL_FILE", str(model_path))

    dummy_model = {"name": "model"}
    features = ["f1", "f2"]
    metadata = {"f1": {"type": "numeric", "min": 0.0, "max": 1.0, "mean": 0.5}}
    metrics = {"r2": 1.0, "mae": 0.0, "mse": 0.0}
    config = {"target": "Total", "app_title": "test"}

    app.save_model(dummy_model, features, metadata, metrics, config)
    loaded = app.load_pretrained_model()

    assert loaded["model"] == dummy_model
    assert loaded["features"] == features
    assert loaded["column_metadata"] == metadata
    assert loaded["metrics"] == metrics
    assert loaded["config"] == config


def test_load_pretrained_model_missing_file_returns_none(tmp_path, monkeypatch):
    model_path = tmp_path / "missing.pkl"
    monkeypatch.setattr(app, "MODEL_FILE", str(model_path))

    loaded = app.load_pretrained_model()
    assert loaded is None


def test_load_pretrained_model_corrupted_file_returns_none(tmp_path, monkeypatch):
    model_path = tmp_path / "broken.pkl"
    model_path.write_bytes(b"not-a-pickle")
    monkeypatch.setattr(app, "MODEL_FILE", str(model_path))

    loaded = app.load_pretrained_model()
    assert loaded is None


def test_load_data_unsupported_file_type_raises(tmp_path):
    """Phase-5 fix: unsupported extensions must raise ValueError."""
    bad_path = tmp_path / "photo.jpg"
    bad_path.write_bytes(b"\xff\xd8\xff")  # JPEG header bytes

    with open(bad_path, "rb") as fobj:
        with pytest.raises(ValueError, match="Unsupported file type"):
            app.load_data.__wrapped__(fobj)


def test_load_pretrained_model_rejects_tampered_hash(tmp_path, monkeypatch):
    """Phase-5 fix: model with mismatched SHA-256 sidecar returns None."""
    model_path = tmp_path / "automl_model.pkl"
    monkeypatch.setattr(app, "MODEL_FILE", str(model_path))

    # Save a valid model first (creates .pkl + .sha256)
    app.save_model(
        {"name": "m"}, ["f1"], {}, {"r2": 1, "mae": 0, "mse": 0},
        {"target": "T", "app_title": "t"},
    )
    assert app.load_pretrained_model() is not None  # sanity: valid load works

    # Tamper the hash file
    hash_path = tmp_path / "automl_model.pkl.sha256"
    hash_path.write_text("0" * 64, encoding="utf-8")

    loaded = app.load_pretrained_model()
    assert loaded is None
