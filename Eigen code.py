import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from worclipo.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')

print(f'The number of columns: {len(data.columns)}')

# ID verwijderen 
#if "ID" in data.columns:
   # data = data.drop(columns=["ID"])

#X = data.drop(columns=["label"])

#variances = X.var()
#print(variances)

#plt.hist(variances, bins=50)
#plt.xlabel("Variance")
#plt.ylabel("Aantal features")
#plt.title("Verdeling van feature variantie")
#plt.show()

#print(variances.sort_values(ascending=False).head(10))

#low_var = variances[variances < 1e-10]
#print(len(low_var))

#Outliers
# X en y maken
#ID verwijderen 
if "ID" in data.columns:
    data = data.drop(columns=["ID"])

X = data.drop(columns=["label"])
y = data["label"]

print("Shape X:", X.shape)

def outlier_summary_iqr(X):
    summary = []

    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        n_outliers = ((X[col] < lower) | (X[col] > upper)).sum()

        summary.append({
            "feature": col,
            "n_outliers": n_outliers,
            "percentage": 100 * n_outliers / len(X)
        })

    return pd.DataFrame(summary).sort_values("n_outliers", ascending=False)

outlier_df = outlier_summary_iqr(X)

print(outlier_df.head(10))

import matplotlib.pyplot as plt


# Alleen features met minstens 1 outlier
outlier_df_nonzero = outlier_df[outlier_df["n_outliers"] > 0]
# features nummeren
outlier_df_nonzero["feature_id"] = range(1, len(outlier_df_nonzero) +1)

# Eventueel sorteren van hoog naar laag
outlier_df_nonzero = outlier_df_nonzero.sort_values("n_outliers", ascending=False)
print(len(outlier_df_nonzero))

# Grafiek maken
plt.figure(figsize=(12, 6))
plt.bar(outlier_df_nonzero["feature_id"], outlier_df_nonzero["n_outliers"])
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Aantal outliers")
plt.title("Aantal outliers per feature")
plt.tight_layout()
plt.show()

