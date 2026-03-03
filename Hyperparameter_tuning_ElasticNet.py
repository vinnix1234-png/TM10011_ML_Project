import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- data ---
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
df.columns = df.columns.str.strip()

target_col = "label"
id_col = "ID"

y_raw = df[target_col].astype(str).str.strip().str.lower()
X = df.drop(columns=[target_col, id_col])

POS_LABEL = "liposarcoma"
y = (y_raw == POS_LABEL).astype(int)

# --- grid search ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=5000
    ))
])

param_grid_lr = {
    "clf__C": np.logspace(-3, 2, 6),               # 0.001 .. 100
    "clf__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]   # 0=L2, 1=L1
}

gs_lr = GridSearchCV(
    pipe_lr,
    param_grid=param_grid_lr,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True
)

gs_lr.fit(X, y)

print("=== BEST ElasticNet LogisticRegression ===")
print("Best CV AUC:", gs_lr.best_score_)
print("Best params:", gs_lr.best_params_)

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

# --- ROC curve (out-of-fold) for the best tuned model ---
y_proba_oof = cross_val_predict(
    gs_lr.best_estimator_,   # best pipeline from GridSearchCV
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
plt.plot(fpr, tpr, label=f"ElasticNet LogReg (OOF AUC = {auc_oof:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Out-of-Fold)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()