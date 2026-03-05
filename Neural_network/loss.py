import numpy as np

class LossCategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        n_samples = len(y_pred)
        correct_confidence = y_pred[range(n_samples), y_true]
        confidences = -np.log(correct_confidence)
        return np.mean(confidences)