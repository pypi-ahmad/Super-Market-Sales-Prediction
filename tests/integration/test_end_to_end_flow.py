import numpy as np
import pandas as pd

import app
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


def test_end_to_end_training_then_load_then_predict(tmp_path, monkeypatch):
    source_path = tmp_path / "train.csv"
    model_path = tmp_path / "automl_model.pkl"

    df = _sample_training_dataframe(rows=20)
    df.to_csv(source_path, index=False)

    monkeypatch.setattr(train_automl, "DATA_SOURCE", str(source_path))
    monkeypatch.setattr(train_automl, "MODEL_FILE", str(model_path))
    monkeypatch.setattr(train_automl, "AutoML", FakeAutoML)
    train_automl.main()

    monkeypatch.setattr(app, "MODEL_FILE", str(model_path))
    artifacts = app.load_pretrained_model()

    assert artifacts is not None
    assert set(artifacts.keys()) == {"model", "features", "column_metadata", "metrics", "config"}

    features = artifacts["features"]
    metadata = artifacts["column_metadata"]
    model = artifacts["model"]

    inference_row = {}
    for feature in features:
        info = metadata[feature]
        if info["type"] == "categorical":
            inference_row[feature] = info["options"][0]
        else:
            inference_row[feature] = info["mean"]

    inference_df = pd.DataFrame([inference_row])
    prediction = model.predict(inference_df)

    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert np.issubdtype(prediction.dtype, np.number)
