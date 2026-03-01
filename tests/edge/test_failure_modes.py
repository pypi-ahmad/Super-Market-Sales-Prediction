import pandas as pd

import app
import train_automl
from tests.helpers import FakeAutoML


def test_missing_model_file_returns_none(tmp_path, monkeypatch):
    model_path = tmp_path / "does_not_exist.pkl"
    monkeypatch.setattr(app, "MODEL_FILE", str(model_path))

    assert app.load_pretrained_model() is None


def test_corrupted_model_file_returns_none_on_load(tmp_path, monkeypatch):
    model_path = tmp_path / "corrupted.pkl"
    model_path.write_bytes(b"corrupted-bytes")
    monkeypatch.setattr(app, "MODEL_FILE", str(model_path))

    loaded = app.load_pretrained_model()
    assert loaded is None


def test_empty_dataframe_metadata_is_empty():
    empty_df = pd.DataFrame()
    assert app.get_column_metadata(empty_df) == {}


def test_invalid_training_data_missing_target_stops_pipeline(tmp_path, monkeypatch):
    source_path = tmp_path / "invalid.csv"
    model_path = tmp_path / "automl_model.pkl"

    invalid_df = pd.DataFrame(
        {
            "Invoice ID": ["INV-1", "INV-2", "INV-3", "INV-4"],
            "Branch": ["A", "B", "C", "A"],
            "Unit price": [10.0, 20.0, 30.0, 40.0],
        }
    )
    invalid_df.to_csv(source_path, index=False)

    monkeypatch.setattr(train_automl, "DATA_SOURCE", str(source_path))
    monkeypatch.setattr(train_automl, "MODEL_FILE", str(model_path))
    monkeypatch.setattr(train_automl, "AutoML", FakeAutoML)

    train_automl.main()

    assert not model_path.exists()


def test_missing_training_source_stops_pipeline(tmp_path, monkeypatch):
    source_path = tmp_path / "missing.csv"
    model_path = tmp_path / "automl_model.pkl"

    monkeypatch.setattr(train_automl, "DATA_SOURCE", str(source_path))
    monkeypatch.setattr(train_automl, "MODEL_FILE", str(model_path))
    monkeypatch.setattr(train_automl, "AutoML", FakeAutoML)

    train_automl.main()

    assert not model_path.exists()
