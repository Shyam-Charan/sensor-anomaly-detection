"""Detectors. Each takes (train_features, train_labels) and exposes predict(features)."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ZScoreBaseline:
    """Flag a reading when it is more than `threshold` rolling standard deviations from its rolling mean."""
    name = "zscore_baseline"

    def __init__(self, column="zscore_72", threshold=3.0):
        self.column, self.threshold = column, threshold

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return (X[self.column].abs() > self.threshold).astype(int).to_numpy()


class IsolationForestDetector:
    """Unsupervised: learns what normal looks like from the training period; labels are not used."""
    name = "isolation_forest"

    def __init__(self, contamination=0.02, seed=0):
        self.model = make_pipeline(StandardScaler(),
                                   IsolationForest(contamination=contamination, random_state=seed, n_jobs=-1))

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def predict(self, X):
        return (self.model.predict(X) == -1).astype(int)


class LogisticRegressionDetector:
    """Supervised linear classifier. class_weight='balanced' because anomalies are rare."""
    name = "logistic_regression"

    def __init__(self, seed=0):
        self.model = make_pipeline(StandardScaler(),
                                   LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class SVMDetector:
    """Supervised RBF-kernel SVM (trained on a subsample for speed)."""
    name = "svm_rbf"

    def __init__(self, max_train=6000, seed=0):
        self.max_train, self.seed = max_train, seed
        self.model = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))

    def fit(self, X, y):
        if len(X) > self.max_train:
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_train, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def all_models():
    return [ZScoreBaseline(), IsolationForestDetector(), LogisticRegressionDetector(), SVMDetector()]
