import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# ---------- LOAD DATA ----------
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
df.columns = df.columns.str.strip()

target_col = "label"
id_col = "ID"

y_raw = df[target_col].astype(str).str.strip().str.lower()
X = df.drop(columns=[target_col, id_col])

# ---------- BINARIZE LABEL ----------
POS_LABEL = "liposarcoma"   # <-- kies jouw "ziek" class
y = (y_raw == POS_LABEL).astype(int)

print("Label counts (raw):")
print(y_raw.value_counts())
print("\nPositive label:", POS_LABEL)
print("Binary distribution (0/1):")
print(y.value_counts().sort_index())
print("X shape:", X.shape)


# ---------- CV SETUP ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"auc": "roc_auc", "bacc": "balanced_accuracy", "acc": "accuracy"}


# ---------- MODELS ----------
logreg_l2 = LogisticRegression(
    penalty="elasticnet", l1_ratio=0.0, solver="saga",
    C=1.0, max_iter=5000, class_weight="balanced"
)

logreg_l1 = LogisticRegression(
    penalty="elasticnet", l1_ratio=1.0, solver="saga",
    C=1.0, max_iter=5000, class_weight="balanced"
)

models = {
    "logreg_l2": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", logreg_l2)
    ]),
    "logreg_l1": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", logreg_l1)
    ]),
    "linear_svm": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", probability=True, class_weight="balanced"))
    ]),
    "sgd_logreg": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ]),
    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2
        ))
    ]),
    "extra_trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=800,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2
        ))
    ]),
    "naive_bayes": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GaussianNB())
    ]),
    "decision_tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=5,
            random_state=42
        ))
    ]),
    "knn": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),
}

# ---------- RUN CV ----------
print("\n5-fold Stratified CV results (mean ± std):")
for name, pipe in models.items():
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(
        f"{name:12s} "
        f"AUC {res['test_auc'].mean():.3f} ± {res['test_auc'].std():.3f} | "
        f"BAcc {res['test_bacc'].mean():.3f} ± {res['test_bacc'].std():.3f} | "
        f"Acc {res['test_acc'].mean():.3f} ± {res['test_acc'].std():.3f}"
    )