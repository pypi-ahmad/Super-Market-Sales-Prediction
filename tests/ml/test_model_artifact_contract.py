import numpy as np
import pandas as pd

import app
from tests.helpers import FakeAutoML


def test_model_artifact_roundtrip_contract(tmp_path, monkeypatch):
    model_path = tmp_path / "automl_model.pkl"
    monkeypatch.setattr(app, "MODEL_FILE", str(model_path))

    model = FakeAutoML()
    train_X = pd.DataFrame(
        {
            "cogs": [100.0, 200.0, 300.0],
            "Tax 5%": [5.0, 10.0, 15.0],
            "Branch": ["A", "B", "C"],
        }
    )
    train_y = pd.Series([105.0, 210.0, 315.0])
    model.fit(train_X, train_y, metric="r2")

    features = ["cogs", "Tax 5%", "Branch"]
    metadata = {
        "cogs": {"type": "numeric", "min": 100.0, "max": 300.0, "mean": 200.0},
        "Tax 5%": {"type": "numeric", "min": 5.0, "max": 15.0, "mean": 10.0},
        "Branch": {"type": "categorical", "options": ["A", "B", "C"]},
    }
    metrics = {"r2": 1.0, "mae": 0.0, "mse": 0.0}
    config = {"target": "Total", "app_title": "contract"}

    app.save_model(model, features, metadata, metrics, config)
    loaded = app.load_pretrained_model()

    assert loaded is not None
    assert set(loaded.keys()) == {"model", "features", "column_metadata", "metrics", "config"}
    assert loaded["features"] == features
    assert loaded["config"]["target"] == "Total"


def test_loaded_model_prediction_shape_and_type(tmp_path, monkeypatch):
    model_path = tmp_path / "automl_model.pkl"
    monkeypatch.setattr(app, "MODEL_FILE", str(model_path))

    model = FakeAutoML()
    train_X = pd.DataFrame({"cogs": [10.0, 20.0], "Tax 5%": [0.5, 1.0]})
    train_y = pd.Series([10.5, 21.0])
    model.fit(train_X, train_y)

    app.save_model(
        model,
        ["cogs", "Tax 5%"],
        {
            "cogs": {"type": "numeric", "min": 10.0, "max": 20.0, "mean": 15.0},
            "Tax 5%": {"type": "numeric", "min": 0.5, "max": 1.0, "mean": 0.75},
        },
        {"r2": 1.0, "mae": 0.0, "mse": 0.0},
        {"target": "Total", "app_title": "contract"},
    )

    loaded = app.load_pretrained_model()
    inference_df = pd.DataFrame({"cogs": [50.0, 80.0], "Tax 5%": [2.5, 4.0]})
    prediction = loaded["model"].predict(inference_df)

    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (2,)
    assert np.issubdtype(prediction.dtype, np.number)
