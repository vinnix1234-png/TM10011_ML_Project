#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
#clf = RandomizedSearchCV(RandomForestClassifier(), param_distributions, cv=5, n_iter=20, random_state=42)

# random_state=42, zorgt ervoor dat elke keer hetzelfde verdeelt wordt, kan bij splitten, classifier, RandomizedSearchCV
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

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

# Feature selection methodes
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.base import clone      # Tijdens SelectFromModel en RFE wordt model al iets getraint, maar je wil wel met schone classifier elke keer beginnen natuurlijk

#%% Train-Test Split
# Train-Test Split
# Data opslitsen in labels en features
y = data.iloc[:, 0]     # Labels
X = data.iloc[:, 1:]    # Features

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 80% train en 20% test
    stratify=y,         # Houdt verhouding lipoma en liposarcoma gelijk
    random_state=42     
)

print(f"Trainset: {X_train.shape}")
print(f"Testset: {X_test.shape}")

#%% Scaling
# Scaling
scaler = PowerTransformer()

# Fit alleen op trainset, maar transform op beide
# Bij fit op testset krijg je data leakage
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)    # Alleen transform, niet fitten!

#%% Preprocessing
# Preprocessing
print(f"Aantal features voor preprocessing: {X_train_scaled.shape[1]}")
# Variance Threshold, voor features met variantie van 0 eruit halen
preprocessing_variance = VarianceThreshold(threshold=0)    
X_train_prepro_var = preprocessing_variance.fit_transform(X_train_scaled)
X_test_prepro_var = preprocessing_variance.transform(X_test_scaled)     # Alleen transform, niet fitten!

print(f"Aantal features na variance threshold: {X_train_prepro_var.shape[1]}")

# Verwijderen van sterk gecorreleerde features
def remove_correlated_features(X, threshold=0.85):
    dataframe = pd.DataFrame(X)
    correlation_matrix = dataframe.corr().abs()         # Bereken van absolute correlatiematrix tussen alle features                   
    upper_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))       # Houd alleen de bovenste helft van matrix over, om duplicaten te vermijden
    to_drop = [col for col in upper_matrix.columns if any(upper_matrix[col] > threshold)]       # Verzamel de features die sterk gecorreleerd zijn met een andere
    return dataframe.drop(columns=to_drop).values, to_drop      # Verwijderen van de features en geeft verwijderde features terug

X_train_prepro_corr, dropped_features = remove_correlated_features(X_train_prepro_var)          # Functie verwijderen van gecorreleerde features toepassen op trainingsdata
X_test_prepro_corr = pd.DataFrame(X_test_prepro_var).drop(columns=dropped_features).values      # Verwijderde features van trainingsdata ook bij testdata weghalen, niet functie toepassen anders data leakage

print(f"Aantal features na verwijderen gecorreleerde features: {X_train_prepro_corr.shape[1]}")
print(f"    Aantal verwijderde gecorreleerde features: {len(dropped_features)}")

# Outliers???

#%% Inner-loop
# Inner-loop
# Dictionary met classifiers
classifiers = {
    "Linear Discriminant": LinearDiscriminantAnalysis(n_components=1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),        
    "Random Forest": RandomForestClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
}

# Functie voor pakken feature selection methodes per classifier 
def get_feature_selection_methods(classifier):
    return {
    "SelectKBest": SelectKBest(f_classif, k=10),
    "PCA": PCA(n_components=10),
    "SelectFromModel": SelectFromModel(clone(classifier), max_features=10),
    "RFE": RFE(clone(classifier), n_features_to_select=10)
    }

# Dictionary met hyperparameters van classifiers
# Linear Discriminant hyperparameters
hyperparameter_grid_linear_discriminant = {
    'solver': ['svd', 'lsqr', 'eigen'],         # svd is standaard en robuust
    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]          # Regularisatie om overfitting te voorkomen
}
# Logistic Regression hyperparameters
hyperparameter_grid_logistic_regression = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['saga'],         # Saga ondersteunt alle drie
    'l1_ratio': [0.1, 0.5, 0.9]        # Alleen relevant voor elasticnet
}
# Random Forest hyperparameters
hyperparameter_grid_random_forest = {
    'n_estimators': [50, 100, 200],         # Aantal bomen
    'max_depth': [5, 10, 15, None],         # Diepte van de bomen
    'min_samples_split': [2, 5, 10],        # Minimum samples om een node te splitsen
    'min_samples_leaf': [1, 2, 4],          # Minimum samples in een leaf node
    'max_features': ['sqrt', 'log2', 0.5]       # Aantal features om te overwegen voor splitsing
}
# CatBoost hyperparameters
hyperparameter_grid_catboost = {
    'iterations': [50, 100, 200],       # Aantal bomen (vergelijkbaar met n_estimators)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],        # Leersnelheid
    'depth': [2, 4, 6],         # Diepte van de bomen
    'l2_leaf_reg': [1, 3, 5, 10],           # L2 regularisatie om overfitting te voorkomen
    'border_count': [32, 64, 128]       # Aantal bins voor numerieke features
}
# Koppelen van hyperparameters aan classifiers
hyperparameter_grids = {
    "Linear Discriminant": hyperparameter_grid_linear_discriminant,
    "Logistic Regression": hyperparameter_grid_logistic_regression,
    "Random Forest": hyperparameter_grid_random_forest,
    "CatBoost": hyperparameter_grid_catboost
}

# Cross-validatie strategie
cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Resultaten opslaan
results = []

# Inner-loop
for clf_name, clf in classifiers.items():
    feature_selection_methods = get_feature_selection_methods(clf)

    for fs_name, fs in feature_selection_methods.items():
        print(f"Begonnen met {clf_name} en {fs_name}")

        try:
            # Pipeline 
            pipeline = Pipeline([
                ("feature_selection", fs),
                ("classifier", clone(clf))      # Frisse kopie bij elke iteratie
            ])

            # Dit is prefix voor hyperparameters en pipeline, komt door dat dictionary in dictionary is soort van
            # k = parameternaam, v = lijst met waarden
            hyperparameter_grid = {f"classifier__{k}": v for k, v in hyperparameter_grids[clf_name].items()}

            # Hyperparametertuning
            hyperparametertuning = RandomizedSearchCV(
                pipeline,
                param_distributions=hyperparameter_grid,
                n_iter=20,
                cv=cross_validation,
                scoring='roc_auc',
                random_state=42,
                n_jobs=-1
            )
            hyperparametertuning.fit(X_train_prepro_corr, y_train)

            # Resultaten opslaan
            results.append({
                "Classifier": clf_name,
                "Feature Selection": fs_name,
                "Mean AUC": hyperparametertuning.best_score_,
                "Best Hyperparameters": hyperparametertuning.best_params_
            })
            print(f"    {clf_name} en {fs_name} afgerond")

        except Exception as e:
            results.append({
                "Classifier": clf_name,
                "Feature Selection": fs_name,
                "Mean AUC": float('nan'),
                "Best Hyperparameters": None
            })
            print(f"    {clf_name} en {fs_name} gefaald: {e}")

results_df = pd.DataFrame(results)
print(results_df.sort_values("Mean AUC", ascending=False))

# %%
