# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!


#%% Data loading functions. Uncomment the one you want to use
#from worcgist.load_data import load_data
from worclipo.load_data import load_data
#from worcliver.load_data import load_data
#from hn.load_data import load_data
#from ecg.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')

print(f'The number of columns: {len(data.columns)}')

## Feature selection
# Er zijn 3 keer zo veel features als samples. dat is niet handig. dus er moeten er een paar uit
# stap 1: variantie kijken

variances = data.select_dtypes(include="number").var()

# in sommeige kolommen is de variantie 0, die extraheren
zero_var_cols = variances[variances == 0].index
data_drop = data.drop(columns=zero_var_cols)
variances_drop = variances.drop(columns=zero_var_cols)
#print(data_drop)
percentage = (variances_drop / variances_drop.sum()) * 100
#print(percentage)
#print(percentage[percentage >= 5])
#print(percentage.sort_values(ascending=False))

variances = data.select_dtypes(include="number").var()

# kolommen met variantie 0 extraheren
zero_var_cols = variances[variances == 0].index

data_drop = data.drop(columns=zero_var_cols)
variances_drop = variances.drop(zero_var_cols)

percentage = (variances / variances.sum()) * 100

print(variances.sort_values(ascending=False))
print(percentage.sort_values(ascending=False))