import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#读入数据
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
sample_submission = pd.read_csv("sample_submission1.csv")
#train.head()
train.head()
test.head()
#原本以及测试id
target = train.SalePrice.values
test_ids = test.Id.values
#整理数据，去除要算的
train.drop(['SalePrice','Id'], axis=1 , inplace=True)
test.drop(["Id"],axis = 1,inplace = True)
num_train = len(train)
data = pd.concat([train,test],axis=0).reset_index(drop=True) #按照列拼接
data.dtypes.value_counts()#看是不是数值
data.drop(data.isnull().mean()[data.isnull().mean() > 0.8].index.tolist(), axis=1, inplace=True)
data.loc[:, data.dtypes[data.dtypes != "object"].index] = \
data.loc[:, data.dtypes[data.dtypes != "object"].index].fillna(
data.loc[:, data.dtypes[data.dtypes != "object"].index].median()) #数值填充
data.loc[:, data.dtypes[data.dtypes == "object"].index] = \
data.loc[:, data.dtypes[data.dtypes == "object"].index].fillna("NULL")#非数值填充
data.shape
data.head()


#sns.displot(target)
#sns.displot(np.log1p(target))

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error as mse
cat_feature = OneHotEncoder().fit_transform(data[data.columns[data.dtypes == "object"]])  #归一化
cat_feature.shape
num_feature = StandardScaler().fit_transform(data[data.columns[data.dtypes != "object"]]) #数值数据 平均值=0 标准差=1
feature = np.concatenate([cat_feature.todense(), num_feature], axis=1)
#区分测试集以及训练集
test_feature = feature[num_train:] 
train_feature = feature[:num_train]

#归一化
target_log1p = np.log1p(target)
scaler = StandardScaler()
scaler.fit(target_log1p.reshape(-1, 1))

target_log1p_scale = scaler.transform(target_log1p.reshape(-1, 1))  #任意行 1列
target_log1p_scale.mean()


model = Ridge()

model.fit(train_feature, target_log1p_scale)

def predict (model,feature):
    pred = model.predict(feature)
    pred = scaler.inverse_transform(pred) #还原归一化
    pred = np.expm1(pred)
    return pred

train_pred = predict(model, train_feature)
train_pred

np.sqrt(mse(target, train_pred))
test_pred = predict(model, test_feature)

sample_submission["SalePrice"] = test_pred.reshape(-1)
sample_submission.to_csv("submission.csv1", index=False)
