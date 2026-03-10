import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

# --- data ---
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
df.columns = df.columns.str.strip()

target_col = "label"
id_col = "ID"

y_raw = df[target_col].astype(str).str.strip().str.lower()
X = df.drop(columns=[target_col, id_col])

POS_LABEL = "liposarcoma"
y = (y_raw == POS_LABEL).astype(int)

# --- cv ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- model + grid ---
pipe_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    "clf__n_estimators": [200, 500, 800],
    "clf__max_depth": [None, 3, 5, 10],
    "clf__min_samples_leaf": [1, 2, 5],
    "clf__max_features": ["sqrt", 0.3, 0.5]
}

gs_rf = GridSearchCV(
    pipe_rf,
    param_grid=param_grid_rf,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True
)

gs_rf.fit(X, y)

print("=== BEST RandomForest ===")
print("Best CV AUC:", gs_rf.best_score_)
print("Best params:", gs_rf.best_params_)



# --- Out-of-fold predicted probabilities for ROC curve (no leakage) ---
y_proba_oof = cross_val_predict(
    gs_rf.best_estimator_,  # best RF pipeline
    X, y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

auc_oof = roc_auc_score(y, y_proba_oof)
fpr, tpr, _ = roc_curve(y, y_proba_oof)

print("\n=== OUT-OF-FOLD ROC AUC ===")
print("OOF AUC:", auc_oof)

plt.figure()
plt.plot(fpr, tpr, label=f"RandomForest (OOF AUC = {auc_oof:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Out-of-Fold)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()