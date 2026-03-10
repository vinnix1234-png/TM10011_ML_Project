import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier

import mrmr


# ======================
# Load data
# ======================

data = pd.read_csv("budt_project/Lipo_radiomicFeatures.csv")

data = data.drop(columns=["ID"])

y = data["label"]
X = data.drop(columns=["label"])

feature_names = X.columns

encoder = LabelEncoder()
y = encoder.fit_transform(y)


# ======================
# Correlation filter
# ======================

class CorrelationFilter(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):

        X = pd.DataFrame(X)

        corr = X.corr().abs()

        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )

        self.to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.threshold)
        ]

        return self

    def transform(self, X):

        X = pd.DataFrame(X)

        return X.drop(columns=self.to_drop, errors="ignore")


# ======================
# mRMR selector
# ======================

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


# ======================
# Final pipeline
# ======================

pipeline = Pipeline([

    ("variance", VarianceThreshold()),

    ("correlation", CorrelationFilter(threshold=0.9)),

    ("mrmr", MRMRSelector(k=10)),

    ("model", AdaBoostClassifier(
        n_estimators=500,
        learning_rate=0.01,
        random_state=42
    ))
])


# ======================
# Repeated cross validation
# ======================

cv = RepeatedStratifiedKFold(

    n_splits=5,
    n_repeats=10,
    random_state=42
)


scores = cross_val_score(

    pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc"
)

print("\nFinal AdaBoost ROC-AUC")
print("Mean:", scores.mean())
print("Std:", scores.std())


# ======================
# Train final model
# ======================

pipeline.fit(X, y)

selected_indices = pipeline.named_steps["mrmr"].selected_features
selected_features = X.columns[selected_indices]

print("\nSelected radiomics features:")
for f in selected_features:
    print(f)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# train model op volledige dataset
pipeline.fit(X, y)

# predict probabilities
y_prob = pipeline.predict_proba(X)[:, 1]


# compute ROC
fpr, tpr, thresholds = roc_curve(y, y_prob)

auc = roc_auc_score(y, y_prob)


plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label=f"AdaBoost (AUC = {auc:.2f})")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()