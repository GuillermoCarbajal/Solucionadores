import numpy as np

class CalibratedModel:

    def __init__(self, model, calibrator, prevalence, threshold=0.5):
        self.model = model
        self.calibrator = calibrator
        self.threshold = threshold
        self.prevalence = prevalence

    def predict_proba_raw(self, X):
        return self.model.predict_proba(X)

    def predict_proba(self, X):
        p_raw = self.predict_proba_raw(X)[:, 1]

        p_cal = self.calibrator.predict_proba(
            p_raw.reshape(-1, 1)
        )[:, 1]

        return np.column_stack([
            1 - p_cal,
            p_cal
        ])

    def predict(self, X):
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)