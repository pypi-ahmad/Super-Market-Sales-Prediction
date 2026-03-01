import numpy as np


class FakeAutoML:
    def __init__(self):
        self.best_estimator = "fake_estimator"
        self.best_config = {"model": "fake"}
        self._fallback_value = 0.0

    def fit(self, X_train, y_train, **settings):
        self._fallback_value = float(np.mean(y_train)) if len(y_train) > 0 else 0.0
        self._fit_settings = settings
        return self

    def predict(self, X):
        if "cogs" in X.columns and "Tax 5%" in X.columns:
            return (X["cogs"].astype(float) + X["Tax 5%"].astype(float)).to_numpy(dtype=float)
        return np.full(len(X), self._fallback_value, dtype=float)
