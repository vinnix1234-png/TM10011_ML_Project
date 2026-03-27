#%% Data Loading as a Pandas DataFrame
# Data Loading as a Pandas DataFrame
# Dit is een basis functie die we van de docenten zelf hebben gekregen
from worclipo.load_data import load_data

data = load_data()

print(f'The number of samples: {len(data.index)}')      # Number of samples = 115
print(f'The number of columns: {len(data.columns)}')    # Number of columns = 494, hiervan is 1 kolom de labels en de rest is features
#print(data.columns)

# Eerste kolom zijn de labels, dus lipoma of liposarcoma
# Opbouw naam features per feature kolom: [PREDICT]_[original]_[featuregroep]_[maat]_[statistiek]_[dimensie/filter]

#%% Import packages
# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc       # Garbage Collector: handmatig geheugen vrijmaken na elke fold, hierbij wordt geheugen minder belast

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
                                    # Clone maakt een frisse kopie van elke estimator, zodat elke fold met een ongetraind model begint

#%% Parameters defineren
# Parameters defineren, waaronder de classifiers, feature selection methodes en hyperparameters

# Data opslitsen in labels (y) en features (X)
y = data.iloc[:, 0]     # Kolom 0, ook wel eerste kolom = Labels (lipoma of liposarcoma)
X = data.iloc[:, 1:]    # Kolom 1 t/m 493, ook wel overige kolommen = Features (radiomics)

# Labels omzetten naar 0/1, om aan te geven wat positief label voor de ROC_curve functie.
# heeft namelijk een positive en negative nodig voor True Positive, True Negative, etc. 
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)  # Resulteert in: 'lipoma' → 0, 'liposarcoma' → 1

# Dictionary met classifiers
# Gekozen voor 3 verschillende classifiers:
    # Logistic Regression: lineair simpel model
        # - max_iter=20000: het maximale aantal iteraties dat het model mag uitvoeren om te convergeren. Default is 100 en dat is heel weinig.
        # - random_state: zorgt voor reproduceerbaarheid, bij hetzelfde getal krijg je altijd hetzelfde resultaat
    # Random Forest: ensemble van beslisbomen
        # - random_state: reproduceerbaarheid, de willekeurige boomsplitsingen zijn reproduceerbaar bijv.
    # XGBoost: gradient booster
        # - tree_method='hist': gebruikt histogram-gebaseerd algoritme om splitsingen te vinden. Dit is sneller dan dan de default exacte methode('exaxt')
        # - random_state: reproduceerbaarheid
        # - eval_metric='logloss': is de evaluatiemetric tijdens intern trainen. De logloss is goed voor binaire classificatie.
        # - verbosity=0: onderdrukt alle outputberichten van XGBoost tijdens het trainen, anders loopt interactive/terminal helemaal vol.
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=20000, random_state=42),        
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(tree_method='hist', random_state=42, eval_metric='logloss', verbosity=0)
}

# Hyperparameters voor bij feature selectie methodes
# Per feature selectie methode worden het aantal te selecteren features gevarieerd, hierdoor wordt aantal features ook een hyperparameter.
# Deze worden samengevoegd met de classifier-hyperparameters voor uiteindelijk in de RandomizedSearchCV.
hyperparameter_grids_feature_selection = {
    "SelectKBest": {"feature_selection__k": [5, 10, 15, 20]},
    "RFE": {"feature_selection__n_features_to_select": [5, 10, 15, 20]},
    "SelectFromModel": {"feature_selection__max_features": [5, 10, 15, 20]},
    "PCA": {"feature_selection__n_components": [5, 10, 15, 20]},
    "None": {}      # Geen feature selectie: geen extra hyperparameters hierbij
}

# Dictionary met hyperparameters van verschillende classifiers
# Logistic Regression hyperparameters:
    # - C: regularisatiesterkte (kleiner = sterkere regularisatie)
    # - solver: 'saga' ondersteunt voor alle 3, dus L1, L2 en ElasticNet
    # - l1_ratio: 0.0=L2 penalty en 0.1=L1(lasso) penalty. mixverhouding L1/L2 is voor ElasticNet (rest)
hyperparameter_grid_logistic_regression = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['saga'],         
    'l1_ratio': [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]   # 0.0 = L2, 1.0 = L1, tussenin = elasticnet. Penalty hoeft niet meer bij sklearn versie .18 in l1_ratio.
}
# Random Forest hyperparameters:
    # - n_estimators: aantal bomen in het ensemble
    # - max_depth: maximale diepte per boom (beperkt om overfitting te voorkomen)
    # - min_samples_split: minimum samples per node voor splitsen (regularisatie)
    # - min_samples_leaf: minimum samples per blad/leaf (regularisatie)
    # - max_features: hoeveel features per splisting overwogen worden
    # - max_samples: fractie van de data die per boom gebruikt wordt (bootstrap)
hyperparameter_grid_random_forest = {
    'n_estimators': [200, 300, 500, 800],    
    'max_depth': [3, 4, 5],      
    'min_samples_split': [2, 5, 10, 15, 20],  
    'min_samples_leaf': [2, 4, 6, 8, 10],          
    'max_features': ['sqrt', 0.2, 0.3],      
    'max_samples': [0.7, 0.8]
}
# XGBoost hyperparameters:
    # - n_estimators: aantal boosting rondes, is nrounds.
    # - learning_rate (eta): stapgrootte per ronde, hierbij is kleiner voorzichtiger leren/trainen. Dit zorgt ervoor dat model minder snel overfit, maar wel trager traint.
    # - gamma: minimale verlies reductie voor een splitsing (pruning, minimum loss reduction)(hogere waarde verlaagt kans op overfitting, maar maakt model wel minder accuraat)
    # - max_depth: klein gehouden vanwege de kleine dataset voor voorkomen van overfitting en model complexiteit.
    # - subsample: fractie van trainsamples per boom (hogere waarde maakt model meer robuust tegen overfitting)
    # - min_child_weight: minimale gewichtsom/samples in een leaf (regularisatie)
    # - colsample_bytree: fractie van features per boom (hogere waarde maakt model meer robuust tegen overfitting, maar wel minder accuraat)
hyperparameter_grid_xgboost = {
    'n_estimators': [50, 100, 200, 300],       
    'learning_rate': [0.01, 0.05, 0.1, 0.5],  
    'gamma': [0, 0.1, 0.3, 0.5],           
    'max_depth': [3, 4, 5],              
    'subsample': [0.6, 0.7, 0.8],
    'min_child_weight': [2, 3, 5, 10],
    'colsample_bytree': [0.3, 0.5, 0.7, 0.9]
}
# Koppelen van hyperparameter grids aan de classifiers
hyperparameter_grids = {
    "Logistic Regression": hyperparameter_grid_logistic_regression,
    "Random Forest": hyperparameter_grid_random_forest,
    "XGBoost": hyperparameter_grid_xgboost
}

#%% Extra functies 
# Extra functies

# Functie voor pakken van de bijbehorende feature selection methodes per classifier 
def get_feature_selection_methods(classifier, clf_name):
    # Parameters voor functie:
        # - classifier: is sklearn estimator die als basis dient voor de model-gebaseerde feature selectie bij SelectFromModel en RFE.
        # - clf_name: naam van classifier, voor bepalen van PCA en RFE extra bij Logistic Regression
    methods = {
        "None": None,       # Geen features selectie. pipeline gebruikt 'passthrough'
        "SelectKBest": SelectKBest(f_classif, k=10),    # Selecteert beste (k) aantal features op basis van ANOVA F-waarde
        "SelectFromModel": SelectFromModel(clone(classifier), max_features=10),      # Selecteert features op basis van feature importances van het model zelf
    }
    
    # PCA en RFE alleen voor Logistic Regression, omdat RFE te zwaar is en niet effectief voor random forest en XGBoost. 
    # RFE kijkt op zijn eigen manier naar feature importance en dat doen random forest en XGBoost zelf ook al, dus zou dubbelop zijn.
    # PCA is ook voor tree-based modellen zoals random forest en XGBoost minder effectief, wat niet relevant maakt.
        # - PCA: reduceert dimensionaliteit via samenvoegen van features tot 1 feature
        # - RFE: traint het model herhaaldelijk en verwijdert telkens de zwakste feature, dit is op basis van geselecteerde model.
    if clf_name == "Logistic Regression":
        methods["PCA"] = PCA(n_components=10)
        methods["RFE"] = RFE(clone(classifier), n_features_to_select=10)        # step=10

    return methods  # retourneert een dictionary met {clf_name: feature selectie methode of None}


# Outliers verwijderen: 
    # Extreme waarden (bijv. meetfouten) kunnen modellen destabiliseren. Deze
    # transformer berekent per feature de onder- en bovengrens op de trainset
    # en knipt (clippen) waarden daarbuiten af tijdens uitvoeren van de Clipper.
# Uitleg BaseEstimator en TransformerMixin:
    # Zijn 2 basis classes die ervoor zorgen dat de Clipper en CorrelationFilter class zich gedragen
    # als volwaardige scikit-learn transformers. Hierdoor kunnen ze gewoon in de Pipeline gestopt worden
    # of gebruikt worden voor RandomizedSearchCV zonder dat ze tot error of andere problemen leiden.
class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        # lower: onderste kwantielgrens (standaard op 0.01 = 1e percentiel)
        # upper: bovenste kwantielgrens (standaard op 0.99 = 99e percentiel)
        self.lower = lower
        self.upper = upper
    
    def fit(self, X, y=None):
        # Sla de kwantielgrenzen op per feature op de trainset, omdat clipper pas wordt geintroduceerd in de inner loop pipeline.
        self.lower_bounds = np.quantile(X, self.lower, axis=0)
        self.upper_bounds = np.quantile(X, self.upper, axis=0)
        return self
    
    def transform(self, X):
        # Clip(knip) alle waarden naar het bereik [lower_bound, upper_bound] per feature
        return np.clip(X, self.lower_bounds, self.upper_bounds)


# Verwijderen van sterk gecorreleerde features, met Spearman: 
    # Hoge correlatie tussen features kan modellen instabiel maken en leidt tot redundante informatie.
    # Redundante informatie betekent overbodige of herhalende informatie die niets toevoegt voor het model.
    # Deze transformer gebruikt Spearman-correlatie, omdat robuust is voor niet-lineare relaties en uitbijters bij de features.
    # De transformers verwijdert features die met een andere feature correleren boven de opgegeven drempel.
    # Voorbeeld: feature B en C correleren met features A, dan worden feature B en C verwijdert en A blijft in trainingsset. 
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        # Correlatiegrens waarboven een feature wordt verwijderd (standaard = 0.8)
        self.threshold = threshold
        self.to_drop = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        correlation_matrix = df.corr(method='spearman').abs()         # Bereken van absolute Spearman-correlatiematrix tussen alle features                   
        upper_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))       # Houd alleen de bovendriehoek van matrix over, om duplicaten te vermijden (zodat elk paar slechts één keer vergeleken wordt)
        self.to_drop = [col for col in upper_matrix.columns if any(upper_matrix[col] > self.threshold)]       # Verzamel de features die sterk gecorreleerd zijn met minstens één andere feature
        return self
    
    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.to_drop).values      # Verwijderen van de geselecteerde sterk gecorreleerde features


# Functie voor het plotten van de gemiddelde ROC-curve met variabiliteitsband over de meerdere outer CV-folds:
    # Per fold wordt de TPR(True Positive Rate) geïnterpoleerd op een gemeenschappelijke FPR(False Positive Rate)-raster
    # (0-1, 100 punten), zodat gemiddelde en standaarddeviatie (grijze vlak) berekend kunnen worden. Individuele fold-curves
    # worden als gestippelde lijnen weergegeven voor transparantie over de spreiding van de prestaties. 
def plot_mean_roc_with_variability(roc_dict, title):
    # parameters voor functie:
        # - roc_dict: dictionary met 'fprs', 'tprs' en 'aucs' (per fold).
        # - title: is titel voor de grafiek (Key: bevat classifier- en fs-naam)
    mean_fpr = np.linspace(0, 1, 100)   # gemeenschappelijk FPR-raster (0-1, 100 punten) voor interpolatie
    interp_tprs = []

    plt.figure(figsize=(7, 7))

    # Plot de individuele ROC-curve *per outer fold
    for i, (fpr, tpr, fold_auc) in enumerate(zip(roc_dict["fprs"], roc_dict["tprs"], roc_dict["aucs"])):
        plt.plot(
            fpr, tpr,
            linestyle="--",
            lw=1,
            alpha=0.4,
            label=f"ROC fold {i+1} (AUC = {fold_auc:.2f})"
        )

        # Interpoleer TPR op het gemeenschappelijke FPR-raster
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0     # zorg dat de curve bij (0,0) begint
        interp_tprs.append(interp_tpr)

    # Bereken en plot de gemiddelde ROC-cuve
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0  # zorg dat de curve bij (1,1) eindigt
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(roc_dict["aucs"])

    plt.plot(
        mean_fpr, mean_tpr,
        color="blue",
        lw=2.5,
        label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})"
    )

    # Maken van variabiliteitsband van +/- 1 standaarddeviatie rondom de gemiddelde ROC-curve
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

    # Diagonale kanslijn (AUC = 0.5 = willekeurige classificatie)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", lw=2, label="Chance level (AUC = 0.5)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC curve with variability\n{title}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

#%% Nested CV
# Nested CV
# Idee Nested CV is het scheiden van de hyperparametertuning (inner loop) en de prestatie-evaluatie (outer loop),
# hierdoor krijg je een onvertekende schatting van de generalisatieprestatie van je model. 
# Zonder nesting zou hyperparametertuning op de testset kunnen lekken, wat een te optimistische AUC score zou geven.
# Dit geeft dan een vertekent beeld de kwaliteit van je model.

# Structuur Nested CV:
    # Outer loop bestaande uit 5 folds: splitst de data in train- en testset voor evaluatie
        # - Inner loop ook bestaande uit 5 folds: voert RandomizedSearchCV uit op de outer trainset
                                                # om de beste hyperparameters te vinden
        # - Beste model (beste hyperparameters) uit inner loop wordt geëvalueerd op de outer testset

# Cross-validaties
# Outer en Inner CV:  
    # - 5-fold: voor voldoende testset samples per fold. Helpt voor verlagen standaarddeviatie
    # - Stratified, voor goede verdeling van labels binnen de folds --> 'lipoma' → 0, 'liposarcoma' → 1
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # shuffle=True moet, anders error op random_state=42
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Resultaten opslaan
results = []

# Dictionary voor ROC-data opslaan
roc_data = {}

# Itereer over alle classifiers en hun bijbehorende features selection methodes
for clf_name, clf in classifiers.items():
    feature_selection_methods = get_feature_selection_methods(clf, clf_name)

    for fs_name, fs in feature_selection_methods.items():      
        print(f"Begonnen met de Nested CV van {clf_name} en {fs_name}")

        try:
            # Outer loop per classifier en features selection methode
            for outer_fold, (outer_train_rijen, outer_test_rijen) in enumerate(outer_cv.split(X, y), start=1):

                # Splits data in outer train- en testset voor deze fold
                # De rijen staan voor de samples
                X_outer_train = X.iloc[outer_train_rijen]
                X_outer_test = X.iloc[outer_test_rijen]     
                y_outer_train = y[outer_train_rijen]        # Geen .iloc meer nodig, door LabelEncoder is het nu een array geworden y.
                y_outer_test = y[outer_test_rijen]

                # Pipeline voor deze fold. Stappen:
                    # 1. RobustScaler: schaalt de features op basis van mediaan en interkwartielafstand (robuust voor uitbijters)
                    # 2. Clipper: zie transformer, kapt extreme waarden af op 1e en 99e percentiel
                    # 3. VarianceThreshold: verwijdert features met (bijna) geen variantie, deze worden namelijk gezien als informatieloos.
                    # 4. CorrelationFilter: zie transformer: verwijdert sterk gecorreleerde features (Spearman > 0.8)
                    # 5. Feature selectie: optionele verdere feature reductie
                    # 6. Classifier: het uiteindelijke machine learning model
                pipeline = Pipeline([
                    ("scaler", RobustScaler()),
                    ("outliers", Clipper(0.01, 0.99)),
                    ("variance", VarianceThreshold(threshold=0.1)),
                    ("correlation", CorrelationFilter(threshold=0.8)),
                    ("feature_selection", clone(fs) if fs is not None else "passthrough"),
                    ("classifier", clone(clf))      # Frisse kopie bij elke fold: voorkomt datalekkage tussen folds
                ])

                # Dit is prefix voor hyperparameters en pipeline, komt door dat dictionary in dictionary is soort van en
                # Pipeline stappen hebben "classifier__" of "feature_selection__" nodig
                # k = parameternaam, v = lijst met waarden
                hyperparameter_grid = {f"classifier__{k}": v for k, v in hyperparameter_grids[clf_name].items()}
                # Feature selection parameters toevoegen
                hyperparameter_grid.update(hyperparameter_grids_feature_selection[fs_name])

                # Inner loop voor hyperparametertuning met RandomSearchCV op outer trainset:
                    # - n_iter=1000: 1000 willekeurige combinaties worden uitgevoerd 
                    # - scoring='roc_auc': optimaliseer op AUC 
                    # - n_jobs=-1: gebruik alle beschikbare CPU-cores parallel
                    # - random_state=42: reproduceerbaarheid, dus telkens de zelfde resultaten
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
                
                # Hyperparametertuning op de outer trainset voor beste hyperparameters
                inner_loop_hyperparametertuning.fit(X_outer_train, y_outer_train)

                # Bekijken van overgebleven features per fold na preprocessing in de pipeline
                best_pipeline = inner_loop_hyperparametertuning.best_estimator_
                n_features_after_variance = best_pipeline.named_steps["variance"].get_support().sum()       # Telt hoeveel features overblijven na de VarianceThreshold
                X_after_variance = best_pipeline.named_steps["variance"].transform(X_outer_train)           # Transformeer de outer trainset met de variance stap om het startpunt voor CorrelationFilter te krijgen
                n_features_after_correlation = best_pipeline.named_steps["correlation"].transform(X_after_variance).shape[1]    # Telt hoeveel featues na CorrelationFilter overblijven

                # Evalueer beste model op de outer testset              
                # Beste model uit de inner loop, gekozen op basis van AUC
                best_model = inner_loop_hyperparametertuning.best_estimator_

                # Voorspel de kansen op outer testset voor de positieve klasse (liposarcoma = 1)
                y_outer_proba = best_model.predict_proba(X_outer_test)[:, 1]

                # Bereken ROC-curve en AUC voor deze outer fold
                fpr, tpr, _ = roc_curve(y_outer_test, y_outer_proba)
                outer_auc = auc(fpr, tpr)

                # ROC-data opslaan per combinatie van classifier + feature selection
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

                # Resultaten opslaan van deze fold
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

                # opruimen van tijdelijke variabelen na elke fold, om geheugenproblemen te voorkomen.
                del best_model, inner_loop_hyperparametertuning
                del X_outer_train, X_outer_test, y_outer_train, y_outer_test
                del best_pipeline,n_features_after_variance, n_features_after_correlation, X_after_variance
                gc.collect()

        except Exception as e:
            # Bij een fout: sla een lege rij op met NaN-waarden en sla de fout op
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

# Resultaten verwerken naar pandas DataFrame
results_df = pd.DataFrame(results)

# Overzicht per combinatie (gemiddeld over outer folds)
# Aggregeer(samenvoegen) per combinatie van classifier + feature selectie 
summary_df = results_df.groupby(["Classifier", "Feature Selection"]).agg(
    Mean_Outer_AUC=("Outer AUC", "mean"),
    Std_Outer_AUC=("Outer AUC", "std"),
    Mean_Inner_AUC=("Inner Mean AUC", "mean"),
).reset_index().sort_values("Mean_Outer_AUC", ascending=False)

print("\nGedetailleerde resultaten per outer fold:")
print(results_df.to_string())

print("\nSamenvatting gesorteerd op classifier + feature selection:")
print(summary_df.to_string())

# Plot de ROC-curves per fold en gemiddelde ROC-curve met variabiliteitsband voor elke combinatie van de key(classifier + feature selectie methode)
for key, roc_dict in roc_data.items():
    if len(roc_dict["fprs"]) > 0:
        plot_mean_roc_with_variability(roc_dict, key)

#%% Voting Classifier
# Voting Classifier met beste configuratie per classifier

# Een Voting Classifier combineert de voorspellingen van meerdere modellen. 
# Met 'soft' voting wordt het gemiddelde van de voorspelde kansen genomen, 
# Waardoor het ensemble robuuster is dan een enkel model. 

# Met 'soft' kijk je dus naar de kansen en met 'hard' kijk je naar predictie uitkomst en vergelijk je die alleen tussen classifiers.
# Je houdt bij 'hard' geen rekening met de kansen, daarom is gekozen voor 'soft'.


# Beste configuratie per classifier (handmatig invullen na resultaten)
# De hyperparameters zijn op basis van de beste configuraties die in de Nested CV naar voren zijn gekomen.
best_configuration = {
    "Logistic Regression": {
        "feature_selection": RFE(LogisticRegression(max_iter=20000, random_state=42, C=1, solver='saga', l1_ratio=0.1), n_features_to_select=10),     # Waarden ook invullen vanwege de RFE
        # RFE traint het model intern om feature importances te bepalen, hierom invullen binnen RFE. 
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
# ** is een Python "unpacking" operator. Het pakt een dictionary uit en zet alle key-value paren om naar losse argumenten.
logistic_regression_pipeline = Pipeline([
    ("scaler", RobustScaler()),
    ("outliers", Clipper(0.01, 0.99)),
    ("variance", VarianceThreshold(threshold=0.1)),
    ("correlation", CorrelationFilter(threshold=0.8)),
    ("feature_selection", best_configuration["Logistic Regression"]["feature_selection"] or "passthrough"),
    ("classifier", clone(LogisticRegression(max_iter=20000, random_state=42, 
                                      **best_configuration["Logistic Regression"]["hyperparameters"])))      
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

# Voting Classifier maken
# voting='soft': combineert de kansen (predict_proba) van alle drie de modellen
voting_clf = VotingClassifier(
    estimators=[
        ("Logistic Regression", logistic_regression_pipeline),
        ("Random Forest", random_forest_pipeline),
        ("XGBoost", xgboost_pipeline)
    ],
    voting='soft'       # Op basis van kans     
)

# ROC-data opslaan van voting classifier
roc_data_voting = {"fprs": [], "tprs": [], "aucs": []}

# Evalueren van voting classifier met outer CV
print("Begonnen met de Nested CV van Voting Classifier")
for outer_fold, (outer_train_rijen, outer_test_rijen) in enumerate(outer_cv.split(X, y), start=1):
    
    X_outer_train = X.iloc[outer_train_rijen]
    X_outer_test = X.iloc[outer_test_rijen]
    y_outer_train = y[outer_train_rijen]
    y_outer_test = y[outer_test_rijen]
    
    # Trainen van voting classifier op outer trainset
    voting_clf.fit(X_outer_train, y_outer_train)
    
    # Voorspel de kansen voor de positieve klasse op de outer testset
    y_proba = voting_clf.predict_proba(X_outer_test)[:, 1]
    
    # Bereken ROC-curve en AUC voor deze fold
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

# Plot de ROC-curve van voting classifier
plot_mean_roc_with_variability(roc_data_voting, "Voting Classifier")

#%% Learning curves
# Learning curves
# Learning curves tonen hoe de train- en testscore veranderen als de hoeveelheid traindata toeneemt. 

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)

common_params = {
    "X": X,
    "y": y,
    "train_sizes": np.linspace(0.1, 1.0, 10),   # 10 gelijke stappen van 10% tot 100% van de trainset
    "cv": outer_cv,         # Gebruik de outer loop van nested CV, voor consistentie
    "score_type": "both",   # Toon zowel train- als testscore
    "n_jobs": -1,           # Parallel uitvoeren op alle CPU cores
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",    # Variabiliteitsband rondom de gemiddelde score
    "score_name": "AUC",
    "scoring": "roc_auc",       
}

# De drie pipelines met de vastgestelde hyperparameters
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