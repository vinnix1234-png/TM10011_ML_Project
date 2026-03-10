import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

import mrmr


# =========================
# Load data
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

        selected = mrmr.mrmr_classif(
            X=X_df,
            y=pd.Series(y),
            K=self.k
        )

        self.selected_features = selected

        return self

    def transform(self, X):

        X_df = pd.DataFrame(X)

        return X_df[self.selected_features]


# =========================
# Cross validation
# =========================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)


# =========================
# AdaBoost pipeline
# =========================

ada_pipeline = Pipeline([

    ("variance", VarianceThreshold()),

    ("correlation", CorrelationFilter(threshold=0.9)),

    ("mrmr", MRMRSelector(k=10)),

    ("model", AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    ))
])


ada_scores = cross_val_score(
    ada_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("\nAdaBoost Repeated CV AUC")
print("Mean:", ada_scores.mean())
print("Std:", ada_scores.std())


# =========================
# XGBoost pipeline
# =========================

xgb_pipeline = Pipeline([

    ("variance", VarianceThreshold()),

    ("correlation", CorrelationFilter(threshold=0.9)),

    ("mrmr", MRMRSelector(k=10)),

    ("model", XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ))
])


xgb_scores = cross_val_score(
    xgb_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("\nXGBoost Repeated CV AUC")
print("Mean:", xgb_scores.mean())
print("Std:", xgb_scores.std())