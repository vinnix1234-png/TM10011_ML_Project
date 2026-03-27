#%% Data Loading as a Pandas DataFrame
# Data Loading as a Pandas DataFrame
from worclipo.load_data import load_data

data = load_data()

print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')
#print(data.columns)

# Opbouw Columns: [PREDICT]_[original]_[featuregroep]_[maat]_[statistiek]_[dimensie/filter]

#%% Import packages
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, LearningCurveDisplay, ShuffleSplit
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve, auc

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Feature selection methodes
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.base import clone      # Tijdens SelectFromModel en RFE wordt model al iets getraint, maar je wil wel met schone classifier elke keer beginnen natuurlijk

#%% Parameters defineren
# Parameters defineren
# Data opslitsen in labels en features
y = data.iloc[:, 0]     # Labels
X = data.iloc[:, 1:]    # Features

# Labels omzetten om aan te geven wat positief label is voor roc_curve functie
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)  # 'lipoma' → 0, 'liposarcoma' → 1

# Dictionary met classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=20000, random_state=42),        
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(tree_method='hist', random_state=42, eval_metric='logloss', verbosity=0)
}

# Hyperparameters voor aantal features
hyperparameter_grids_feature_selection = {
    "SelectKBest": {"feature_selection__k": [5, 10, 15, 20]},
    "RFE": {"feature_selection__n_features_to_select": [5, 10, 15, 20]},
    "SelectFromModel": {"feature_selection__max_features": [5, 10, 15, 20]},
    "PCA": {"feature_selection__n_components": [5, 10, 15, 20]},
    "None": {}
}

# Dictionary met hyperparameters van classifiers
# Logistic Regression hyperparameters
hyperparameter_grid_logistic_regression = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['saga'],         # Saga ondersteunt alle drie
    'l1_ratio': [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]   # 0.0 = L2, 1.0 = L1, tussenin = elasticnet, penalty hoeft niet.
}
# Random Forest hyperparameters
hyperparameter_grid_random_forest = {
    'n_estimators': [200, 300, 500, 800],         # Aantal bomen
    'max_depth': [3, 4, 5],         # Diepte van de bomen
    'min_samples_split': [2, 5, 10, 15, 20],        # Minimum samples om een node te splitsen
    'min_samples_leaf': [2, 4, 6, 8, 10],          # Minimum samples in een leaf node
    'max_features': ['sqrt', 0.2, 0.3],      # Aantal features om te overwegen voor splitsing
    'max_samples': [0.7, 0.8]
}
# XGBoost hyperparameters
hyperparameter_grid_xgboost = {
    'n_estimators': [50, 100, 200, 300],         # nrounds
    'learning_rate': [0.01, 0.05, 0.1, 0.5],     # eta
    'gamma': [0, 0.1, 0.3, 0.5],            # minimum loss reduction
    'max_depth': [3, 4, 5],              # klein vanwege kleine dataset
    'subsample': [0.6, 0.7, 0.8],
    'min_child_weight': [2, 3, 5, 10],
    'colsample_bytree': [0.3, 0.5, 0.7, 0.9]
}
# Koppelen van hyperparameters aan classifiers
hyperparameter_grids = {
    "Logistic Regression": hyperparameter_grid_logistic_regression,
    "Random Forest": hyperparameter_grid_random_forest,
    "XGBoost": hyperparameter_grid_xgboost
}

#%% Extra functies 
# Extra functies
# Functie voor pakken feature selection methodes per classifier 
def get_feature_selection_methods(classifier, clf_name):
    methods = {
        "None": None,
        "SelectKBest": SelectKBest(f_classif, k=10),
        "SelectFromModel": SelectFromModel(clone(classifier), max_features=10),      
    }
    
    # PCA en RFE alleen voor Logistic Regression
    if clf_name == "Logistic Regression":
        methods["PCA"] = PCA(n_components=10)
        methods["RFE"] = RFE(clone(classifier), n_features_to_select=10)        # step=10

    return methods

# Outliers verwijderen 
class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
    
    def fit(self, X, y=None):
        self.lower_bounds = np.quantile(X, self.lower, axis=0)
        self.upper_bounds = np.quantile(X, self.upper, axis=0)
        return self
    
    def transform(self, X):
        return np.clip(X, self.lower_bounds, self.upper_bounds)

# Verwijderen van sterk gecorreleerde features, met Spearman
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.to_drop = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        correlation_matrix = df.corr(method='spearman').abs()         # Bereken van absolute correlatiematrix tussen alle features                   
        upper_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))       # Houd alleen de bovenste helft van matrix over, om duplicaten te vermijden
        self.to_drop = [col for col in upper_matrix.columns if any(upper_matrix[col] > self.threshold)]       # Verzamel de features die sterk gecorreleerd zijn met een andere
        return self
    
    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.to_drop).values      # Verwijderen van de features

# Functie voor het plotten van de ROC-curve
def plot_mean_roc_with_variability(roc_dict, title):
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []

    plt.figure(figsize=(7, 7))

    # Plot ROC-curves van individuele folds
    for i, (fpr, tpr, fold_auc) in enumerate(zip(roc_dict["fprs"], roc_dict["tprs"], roc_dict["aucs"])):
        plt.plot(
            fpr, tpr,
            linestyle="--",
            lw=1,
            alpha=0.4,
            label=f"ROC fold {i+1} (AUC = {fold_auc:.2f})"
        )

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    # Gemiddelde ROC
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_dict["aucs"])

    plt.plot(
        mean_fpr, mean_tpr,
        color="blue",
        lw=2.5,
        label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})"
    )

    # Variabiliteit
    std_tpr = np.std(interp_tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label="± 1 std. dev."
    )

    # Chance line
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", lw=2, label="Chance level (AUC = 0.5)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC curve with variability\n{title}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

#%% Nested CV
# Nested CV
# Cross-validaties
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # shuffle=True moet, anders error op random_state
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Resultaten opslaan
results = []

# Dictionary voor ROC-data
roc_data = {}

# Nested CV
for clf_name, clf in classifiers.items():
    feature_selection_methods = get_feature_selection_methods(clf, clf_name)

    for fs_name, fs in feature_selection_methods.items():      
        print(f"Begonnen met de Nested CV van {clf_name} en {fs_name}")

        try:
            # Outer loop per classifier
            for outer_fold, (outer_train_rijen, outer_test_rijen) in enumerate(outer_cv.split(X, y), start=1):

                # De rijen staan voor de samples
                X_outer_train = X.iloc[outer_train_rijen]
                X_outer_test = X.iloc[outer_test_rijen]     
                y_outer_train = y[outer_train_rijen]        # Geen .iloc meer nodig, door LabelEncoder is het nu een array geworden.
                y_outer_test = y[outer_test_rijen]

                # Pipeline 
                pipeline = Pipeline([
                    ("scaler", RobustScaler()),
                    ("outliers", Clipper(0.01, 0.99)),
                    ("variance", VarianceThreshold(threshold=0.1)),
                    ("correlation", CorrelationFilter(threshold=0.8)),
                    ("feature_selection", clone(fs) if fs is not None else "passthrough"),
                    ("classifier", clone(clf))      # Frisse kopie bij elke iteratie
                ])

                # Dit is prefix voor hyperparameters en pipeline, komt door dat dictionary in dictionary is soort van
                # k = parameternaam, v = lijst met waarden
                hyperparameter_grid = {f"classifier__{k}": v for k, v in hyperparameter_grids[clf_name].items()}
                # Feature selection parameters samenvoegen
                hyperparameter_grid.update(hyperparameter_grids_feature_selection[fs_name])

                # Inner loop voor hyperparametertuning met Gridsearch op outer trainset
                inner_loop_hyperparametertuning = RandomizedSearchCV(
                    pipeline, 
                    param_distributions=hyperparameter_grid,
                    n_iter=1000,
                    cv=inner_cv,
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=42,
                    error_score='raise'
                )
                
                inner_loop_hyperparametertuning.fit(X_outer_train, y_outer_train)

                # Bekijken van overgebleven features per fold
                best_pipeline = inner_loop_hyperparametertuning.best_estimator_
                n_features_after_variance = best_pipeline.named_steps["variance"].get_support().sum()       # Telt hoeveel features overblijven na de VarianceThreshold
                X_after_variance = best_pipeline.named_steps["variance"].transform(X_outer_train)
                n_features_after_correlation = best_pipeline.named_steps["correlation"].transform(X_after_variance).shape[1]    # Telt hoeveel featues na CorrelationFilter overblijven

                # Evalueer beste model op de outer testset              
                # Beste model uit de inner loop
                best_model = inner_loop_hyperparametertuning.best_estimator_

                # Predict probabilities op outer testset
                y_outer_proba = best_model.predict_proba(X_outer_test)[:, 1]

                # ROC-curve voor deze outer fold
                fpr, tpr, _ = roc_curve(y_outer_test, y_outer_proba)
                outer_auc = auc(fpr, tpr)

                # ROC-data opslaan per classifier + feature selection
                key = f"{clf_name} + {fs_name}"
                if key not in roc_data:
                    roc_data[key] = {
                        "fprs": [],
                        "tprs": [],
                        "aucs": []
                    }

                roc_data[key]["fprs"].append(fpr)
                roc_data[key]["tprs"].append(tpr)
                roc_data[key]["aucs"].append(outer_auc)

                # Resultaten opslaan
                results.append({
                    "Classifier": clf_name,
                    "Feature Selection": fs_name,
                    "Features after Variance": n_features_after_variance,
                    "Features after Correlation": n_features_after_correlation,
                    "Outer Fold": outer_fold,
                    "Outer AUC": outer_auc,
                    "Inner Mean AUC": inner_loop_hyperparametertuning.best_score_,
                    "Inner Std AUC": inner_loop_hyperparametertuning.cv_results_['std_test_score'][inner_loop_hyperparametertuning.best_index_],
                    "Best Hyperparameters": inner_loop_hyperparametertuning.best_params_
                })
                print(f"    {clf_name} + {fs_name} ==> fold {outer_fold} afgerond")

                # opruimen van tijdelijke variabelen
                del best_model, inner_loop_hyperparametertuning
                del X_outer_train, X_outer_test, y_outer_train, y_outer_test
                del best_pipeline,n_features_after_variance, n_features_after_correlation, X_after_variance
                gc.collect()

        except Exception as e:
            results.append({
                "Classifier": clf_name,
                "Feature Selection": fs_name,
                "Features after Variance": None,
                "Features after Correlation": None,
                "Outer Fold": None,
                "Outer AUC": float('nan'),
                "Inner Mean AUC": float('nan'),
                "Inner Std AUC": float('nan'),
                "Best Hyperparameters": None
            })
            print(f"    {clf_name} + {fs_name} ==> fold {outer_fold} gefaald: {e}")

# Resultaten
results_df = pd.DataFrame(results)

# Overzicht per combinatie (gemiddeld over outer folds)
summary_df = results_df.groupby(["Classifier", "Feature Selection"]).agg(
    Mean_Outer_AUC=("Outer AUC", "mean"),
    Std_Outer_AUC=("Outer AUC", "std"),
    Mean_Inner_AUC=("Inner Mean AUC", "mean"),
).reset_index().sort_values("Mean_Outer_AUC", ascending=False)

print("\nGedetailleerde resultaten per outer fold:")
print(results_df.to_string())

print("\nSamenvatting gesorteerd op classifier + feature selection:")
print(summary_df.to_string())

for key, roc_dict in roc_data.items():
    if len(roc_dict["fprs"]) > 0:
        plot_mean_roc_with_variability(roc_dict, key)

#%% Voting Classifier
# Voting Classifier met beste configuratie per classifier

# Beste configuratie per classifier (handmatig invullen na resultaten)
best_configuration = {
    "Logistic Regression": {
        "feature_selection": RFE(LogisticRegression(max_iter=20000, random_state=42, C=1, solver='saga', l1_ratio=0.1), n_features_to_select=10),     # Waarden ook invullen vanwege de RFE
        "hyperparameters": {
            'C': 1,
            'solver': 'saga' ,         
            'l1_ratio': 0.1
        }
    },
    "Random Forest": {
        "feature_selection": None,         
        "hyperparameters": {
            'n_estimators': 200,         
            'max_depth': 4,      
            'min_samples_split': 15,        
            'min_samples_leaf': 6,  
            'max_features': 0.2,      
            'max_samples': 0.8
        }                        
    },
    "XGBoost": {
        "feature_selection": None,          
        "hyperparameters": {
            'n_estimators': 100,        
            'learning_rate': 0.1,    
            'gamma': 0.3,           
            'max_depth': 3,             
            'subsample': 0.7,
            'min_child_weight': 3,
            'colsample_bytree': 0.9
        }                        
    }
}

# Pipelines bouwen per classifier met beste configuratie
logistic_regression_pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("outliers", Clipper(0.01, 0.99)),
    ("variance", VarianceThreshold(threshold=0.1)),
    ("correlation", CorrelationFilter(threshold=0.8)),
    ("feature_selection", best_configuration["Logistic Regression"]["feature_selection"] or "passthrough"),
    ("classifier", clone(LogisticRegression(max_iter=20000, random_state=42, 
                                      **best_configuration["Logistic Regression"]["hyperparameters"])))      # ** is een Python "unpacking" operator. Het pakt een dictionary uit en zet alle key-value paren om naar losse argumenten.
])

random_forest_pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("outliers", Clipper(0.01, 0.99)),
    ("variance", VarianceThreshold(threshold=0.1)),
    ("correlation", CorrelationFilter(threshold=0.8)),
    ("feature_selection", best_configuration["Random Forest"]["feature_selection"] or "passthrough"),
    ("classifier", clone(RandomForestClassifier(random_state=42, 
                                          **best_configuration["Random Forest"]["hyperparameters"])))
])

xgboost_pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("outliers", Clipper(0.01, 0.99)),
    ("variance", VarianceThreshold(threshold=0.1)),
    ("correlation", CorrelationFilter(threshold=0.8)),
    ("feature_selection", best_configuration["XGBoost"]["feature_selection"] or "passthrough"),
    ("classifier", clone(XGBClassifier(tree_method='hist', random_state=42, 
                                 eval_metric='logloss', verbosity=0,
                                 **best_configuration["XGBoost"]["hyperparameters"])))
])

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ("Logistic Regression", logistic_regression_pipeline),
        ("Random Forest", random_forest_pipeline),
        ("XGBoost", xgboost_pipeline)
    ],
    voting='soft'       # Op basis van kans     
)

# Resultaten opslaan van voting
roc_data_voting = {"fprs": [], "tprs": [], "aucs": []}

# Evalueren met outer CV
print("Begonnen met de Nested CV van Voting Classifier")
for outer_fold, (outer_train_rijen, outer_test_rijen) in enumerate(outer_cv.split(X, y), start=1):
    
    X_outer_train = X.iloc[outer_train_rijen]
    X_outer_test = X.iloc[outer_test_rijen]
    y_outer_train = y[outer_train_rijen]
    y_outer_test = y[outer_test_rijen]
    
    voting_clf.fit(X_outer_train, y_outer_train)
    
    y_proba = voting_clf.predict_proba(X_outer_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_outer_test, y_proba)
    fold_auc = auc(fpr, tpr)
    
    roc_data_voting["fprs"].append(fpr)
    roc_data_voting["tprs"].append(tpr)
    roc_data_voting["aucs"].append(fold_auc)
    
    print(f"    Voting Classifier ==> fold {outer_fold} afgerond")
    
    del X_outer_train, X_outer_test, y_outer_train, y_outer_test
    gc.collect()

# Resultaten printen van voting classifier
print(f"\nVoting Classifier - Mean AUC: {np.mean(roc_data_voting['aucs']):.3f} ± {np.std(roc_data_voting['aucs']):.3f}")

# ROC-curve plotten van voting classifier
plot_mean_roc_with_variability(roc_data_voting, "Voting Classifier")

#%% Learning curves
# Learning curves
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)

common_params = {
    "X": X,
    "y": y,
    "train_sizes": np.linspace(0.1, 1.0, 10),
    "cv": outer_cv,         # Gebruik de outer loop van nested CV
    "score_type": "both",
    "n_jobs": -1,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "AUC",
    "scoring": "roc_auc",       
}

estimators = [
    ("Logistic Regression", logistic_regression_pipeline),
    ("Random Forest", random_forest_pipeline),
    ("XGBoost", xgboost_pipeline)
]

for ax, (name, estimator) in zip(axes, estimators):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Test Score"])
    ax.set_title(f"Learning Curve - {name}")

plt.tight_layout()
plt.show()

#%%