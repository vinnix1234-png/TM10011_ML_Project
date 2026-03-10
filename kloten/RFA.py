import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import feature_selection
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load dataset
# -----------------------------
file_path = "worclipo/Lipo_radiomicFeatures_scaled_powertransform.csv"
df = pd.read_csv(file_path)

# -----------------------------
# Prepare data
# -----------------------------
X = df.drop(columns=["ID", "label"])
y = df["label"]

# encode labels
le = LabelEncoder()
y = le.fit_transform(y)

print("Original number of features:", X.shape[1])

# -----------------------------
# Remove highly correlated features
# -----------------------------
corr_matrix = X.corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

X = X.drop(columns=to_drop)

print("Removed correlated features:", len(to_drop))
print("Remaining features:", X.shape[1])

# -----------------------------
# Convert to numpy
# -----------------------------
X2 = X.values
y2 = y

# -----------------------------
# Random Forest model
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# RFECV feature selection
# -----------------------------
rfecv = feature_selection.RFECV(
    estimator=rf,
    step=3,
    cv=model_selection.StratifiedKFold(4),
    scoring="roc_auc",
    n_jobs=-1
)

rfecv.fit(X2, y2)

# -----------------------------
# Results
# -----------------------------
print("Optimal number of features:", rfecv.n_features_)

selected_features = X.columns[rfecv.support_]

print("\nSelected features:")
for f in selected_features:
    print(f)

# -----------------------------
# Plot performance
# -----------------------------
plt.figure(figsize=(8,6))

plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation ROC AUC")

plt.plot(
    range(1, len(rfecv.cv_results_["mean_test_score"]) + 1),
    rfecv.cv_results_["mean_test_score"]
)

plt.title("RFECV Feature Selection (Random Forest)")
plt.show()