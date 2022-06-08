import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
test.describe()
#train.head()
train['SalePrice_Log'] = np.log(train['SalePrice'])
data = np.round(train['SalePrice_Log'],1).value_counts()
data = data.sort_index()
print(data.head(10))
numerical_feats = train.dtypes[train.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))
categorical_feats = train.dtypes[train.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))
