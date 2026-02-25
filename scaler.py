import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer


# Data laden
df = pd.read_csv("worclipo/Lipo_radiomicFeatures.csv")
df.columns = df.columns.str.strip()

y_raw = df["label"].astype(str).str.strip().str.lower()
X = df.drop(columns=["label", "ID"])

ziek = "liposarcoma"  
y = (y_raw == ziek).astype(int)


# hier eigelijk data splitten 


# scalen > zorg dat labels uit je X zijn gehaald (alleen parameters)

print("sick patients", (y == 1).sum())

#scaler = StandardScaler()
#scaler = RobustScaler 
scaler =PowerTransformer


#X_scaled = scaler.fit_transform(X)  # deze is het beste van de keuze uit de college's

pt = PowerTransformer(method="yeo-johnson", standardize=True)   # deze is blijkbaar het beste voor onze data ) 
X_scaled = pt.fit_transform(X)


# tadaa dit is scalen .... er zijn wel een paar verschillende maar vgm is de standaard het beste




# Hier een plot om te kijken of het bij het klopt bij een feature 
col = X.columns[0]
i = 0  # index van die kolom

orig = X[col].to_numpy()
scaled = X_scaled[:, i]

print("Feature:", col)
print("Orig mean/std:", np.mean(orig), np.std(orig))
print("Scaled mean/std:", np.mean(scaled), np.std(scaled))

# 1 plot met 2 boxplots: origineel vs geschaald
plt.figure(figsize=(6, 4))
plt.boxplot([orig, scaled], tick_labels=["orig", "scaled"])
plt.ylabel(col)
plt.title("Orig vs scaled (StandardScaler)")
plt.tight_layout()
plt.show()