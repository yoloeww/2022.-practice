import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer 
from sklearn.model_selection import cross_val_score
%matplotlib inline
train = pd.read_csv("train4.csv")
test = pd.read_csv("test4.csv")
submission = pd.read_csv("gender.csv")
train.groupby(pd.cut(train.Age, bins=10))["Survived"].mean().sort_index().plot(kind="bar")
train.groupby("Sex")["Survived"].mean()
train.groupby("Embarked")["Survived"].mean()
train.groupby("Embarked")["Fare"].mean()
def extract_name(x):
    x = x.split("(")[0]
    name1, name2 = x.split(",")
    name2, name3 = name2.split(".")
    return [name1.strip().lower(), name2.strip().lower(), name3.strip().lower()]
train_names = train.Name.apply(extract_name)
test_names = test.Name.apply(extract_name)
train["name1"] = [n[0] for n in train_names]
train["name2"] = [n[1] for n in train_names]
train["name3"] = [n[2] for n in train_names]
test["name1"] = [n[0] for n in test_names]
test["name2"] = [n[1] for n in test_names]
test["name3"] = [n[2] for n in test_names]
train.groupby("name2")["Survived"].agg(["mean", "count"])
drop_name2 = train.name2.value_counts().to_frame().query("name2<5").index.tolist()
train.loc[train.name2.isin(drop_name2), "name2"] = "other"
test.loc[test.name2.isin(drop_name2), "name2"] = "other"
train.groupby("name2")["Survived"].agg(["mean", "count"])
train.drop(["Name", "name1", "name3"], axis=1, inplace=True)
test.drop(["Name", "name1", "name3"], axis=1, inplace=True)
def extract_ticket(x):
     x = x.split(" ")
     if len(x) > 1:
        prefix = x[0]
        ticket = " ".join(x[1:])
     else:
        prefix = "NULL"
        ticket = x
     return prefix.replace(".", "").lower()
train_ticket = train.Ticket.apply(extract_ticket)
test_ticket = test.Ticket.apply(extract_ticket)
train["Ticket"] = train_ticket
test["Ticket"] = test_ticket
#统计缺失值个数
train.isnull().sum()
test.isnull().sum()
num_train = len(train)
train_target = train["Survived"].values
train.drop(["PassengerId", "Survived"], axis=1, inplace=True)
test.drop(["PassengerId"], axis=1, inplace=True)
data = pd.concat([train, test], axis=0)
sns.displot(data.Age, bins=30)
data["Age"] = pd.cut(data.Age, bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 100],labels=[f"age_{i}" for i in range(1, 12)]).astype(str)
data.isnull().sum()

data["Fare"] = data.Fare.fillna(data.Fare.median())
data["Embarked"] = data.Embarked.fillna("S")
data["Cabin"].unique()
data["Cabin"] = data["Cabin"].apply(lambda x:x[0] if isinstance(x, str) else "null")
 #判断x：x【0】（索引值）是不是字符串
data["Fare"] = pd.cut(np.log1p(data.Fare), bins=[0, 2, 3, 4, 5, 8], labels=[f"fare_{i}" for i in range(1, 6)])
                                                                            
data.loc[data.Parch >= 3, "Parch"] = 3
data = data.rename({"name2": "Name"}, axis=1)
data["Fare"] = data.Fare.astype("str")
one_hot = OneHotEncoder(drop="first").fit(data)
data_onehot = one_hot.transform(data)
one_hot.categories_
train = data_onehot[:num_train, :]
test = data_onehot[num_train:, :]   
model = DecisionTreeClassifier(random_state=25
                                ,max_depth=3
                                ,criterion="entropy"
                               min_samples_split=3)  # criterion默认为'gini'系数，也可选择信息增益熵'entropy'
 # 调用fit()方法进行训练,()内为训练集的特征值与目标值

model.fit(train, train_target)
model.predict(test) 
submission["Survived"] = model.predict(test)
submission.to_csv("submission.csv", index=False)
