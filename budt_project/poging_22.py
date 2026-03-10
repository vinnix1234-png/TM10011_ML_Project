import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline



# Load data


data = pd.read_csv("budt_project/Lipo_radiomicFeatures.csv")

data = data.drop(columns=["ID"])

y = data["label"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)



# Selected radiomics features


selected_features = [

"PREDICT_original_sf_volume_2.5D",
"PREDICT_original_tf_GLCMMS_homogeneityd1.0A0.0std",
"PREDICT_original_tf_LBP_peak_R8_P24",
"PREDICT_original_tf_Gabor_peak_position_F0.05_A2.36",
"PREDICT_original_tf_Gabor_min_F0.2_A1.57",
"PREDICT_original_sf_compactness_std_2.5D",
"PREDICT_original_sf_area_avg_2.5D",
"PREDICT_original_tf_Gabor_median_F0.2_A0.0",
"PREDICT_original_tf_Gabor_range_F0.2_A0.79",
"PREDICT_original_logf_peak_sigma10"

]

X = data[selected_features]



# Final AdaBoost model


model = AdaBoostClassifier(

    n_estimators=500,
    learning_rate=0.01,
    random_state=42

)


pipeline = Pipeline([
    ("model", model)
])



# Cross validation


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


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# train model op volledige dataset
pipeline.fit(X, y)

# predict probabilities
y_prob = pipeline.predict_proba(X)[:, 1]

# compute ROC
fpr, tpr, thresholds = roc_curve(y, y_prob)
auc = roc_auc_score(y, y_prob)

print("\nFinal AdaBoost ROC-AUC (fixed features)")
print("Mean:", scores.mean())
print("Std:", scores.std())

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AdaBoost (AUC = {auc:.2f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()