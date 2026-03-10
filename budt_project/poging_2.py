import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier

import mrmr


# =========================
# Load dataset
# =========================

data = pd.read_csv("budt_project/Lipo_radiomicFeatures.csv")

data = data.drop(columns=["ID"])

y = data["label"]
X = data.drop(columns=["label"])

encoder = LabelEncoder()
y = encoder.fit_transform(y)


# =========================
# Correlation filter
# =========================

class CorrelationFilter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):

        X = pd.DataFrame(X)

        corr_matrix = X.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.threshold)
        ]

        return self

    def transform(self, X):

        X = pd.DataFrame(X)

        return X.drop(columns=self.to_drop, errors="ignore")


# =========================
# mRMR selector
# =========================

class MRMRSelector(BaseEstimator, TransformerMixin):

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):

        X_df = pd.DataFrame(X)

        self.selected_features = mrmr.mrmr_classif(
            X=X_df,
            y=pd.Series(y),
            K=self.k
        )

        return self

    def transform(self, X):

        X_df = pd.DataFrame(X)

        return X_df[self.selected_features]


# =========================
# Pipeline
# =========================

pipeline = Pipeline([

    ("variance", VarianceThreshold()),

    ("correlation", CorrelationFilter(threshold=0.9)),

    ("mrmr", MRMRSelector()),

    ("model", AdaBoostClassifier(
        random_state=42
    ))
])


# =========================
# Hyperparameter grid
# =========================

param_grid = {

    "mrmr__k": [5, 8, 10, 12, 15],

    "model__n_estimators": [100, 200, 300, 500],

    "model__learning_rate": [0.01, 0.03, 0.05, 0.1]
}


# =========================
# Cross validation
# =========================

cv = RepeatedStratifiedKFold(

    n_splits=5,
    n_repeats=10,
    random_state=42
)


# =========================
# Grid search
# =========================

search = GridSearchCV(

    pipeline,

    param_grid,

    cv=cv,

    scoring="roc_auc",

    n_jobs=-1
)


search.fit(X, y)


# =========================
# Results
# =========================

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validated ROC-AUC:")
print(search.best_score_)