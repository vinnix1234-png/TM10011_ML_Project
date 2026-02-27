# Mijn eigen code - joey

#%% Data Loading as a DataFrame
# Data Loading as a DataFrame
from worclipo.load_data import load_data

data = load_data()

print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')
print(data.columns)
# Opbouw Columns: [PREDICT]_[original]_[featuregroep]_[maat]_[statistiek]_[dimensie/filter]

#%% Import packages
# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Import extra functies 
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

#%% Import Basic Classifiers
# Import Basic Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#%% Basic Classifier test
# Dictionary met alle basic classifiers
basic_classifiers = {
    "Naive Bayes": GaussianNB(),
    "Linear Discriminant": LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'),
    "Quadratic Discriminant": QuadraticDiscriminantAnalysis(),
    "Logistic Regression": LogisticRegression(),
    "Stochastic Gradient Descent": SGDClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbour": KNeighborsClassifier()
}

results = {}

# OPLETTEN: begint vanaf label kolom, ID ziet hij niet als kolom ofzo. 
# X is de feature matrix,
X = data.iloc[:,1:]
# y zijn de labels
y = data.iloc[:,0]
# Zet label om in getal, dus lipoma=0, liposarcoma=1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
#print(y_encoded)

for name, classifier in basic_classifiers.items():
    # Pipeline voor eerst schalen en dan classifier toepassen
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=20)),  # reduceer naar 20 features
        ("classifier", classifier)
    ])

    scores = cross_val_score(pipe, X, y_encoded, cv=5, scoring="roc_auc")

    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f}")

#%% # Import extra classifiers
# Import extra classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Gradient boosting (XGBoost, LightGBM)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Support Vector Machines (SVM)
from sklearn.svm import SVC
# Multi-layer Perceptron (MLP)
#from sklearn.neural_network import MLPClassifier
# Extra lineaire modellen
from sklearn.linear_model import RidgeClassifier

#%% Extra Classifier test
# Dictionary met extra classifiers
extra_classifiers = {
    "Random Forrest": RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42, class_weight='balanced'),
    "Support Vector Machines": SVC(kernel='linear', probability=True, class_weight='balanced'),
    "XGBoost": XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(max_depth=5, n_estimators=200, learning_rate=0.05, random_state=42, class_weight='balanced', num_leaves=20, min_child_samples=5),
    "Logistic Regression - L1": LogisticRegression(l1_ratio=1, solver='liblinear', class_weight='balanced', max_iter=1000),
    "Logistic Regression - L2": LogisticRegression(l1_ratio=0, C=1, class_weight='balanced', max_iter=1000),
    "Logistic Regression - Elasticnet": LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', C=1, class_weight='balanced', max_iter=1000, random_state=42),
    "Ridge Classifier": RidgeClassifier(class_weight='balanced'),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
}


for name, classifier in extra_classifiers.items():
    # Pipeline voor eerst schalen en dan classifier toepassen
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("feature selection", SelectKBest(f_classif, k=20)),  # PCA(n_components=20), reduceer naar 20 features
        ("classifier", classifier)
    ])

    scores = cross_val_score(pipe, X, y_encoded, cv=5, scoring="roc_auc")

    results[name] = scores.mean()
    print(f"{name}: {scores.mean():.4f}")

#%%