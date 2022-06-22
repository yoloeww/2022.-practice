import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
%matplotlib inline
train = pd.read_csv("train4.csv")
test = pd.read_csv("test4.csv")
submission = pd.read_csv("gender_submission.csv")
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
train.isnull().sum()
num_train = len(train)
train_target = train["Survived"].values
train.drop(["PassengerId", "Survived"], axis=1, inplace=True)
test.drop(["PassengerId"], axis=1, inplace=True)
data = pd.concat([train, test], axis=0)
sns.displot(data.Age, bins=30)
data["Age"] = pd.cut(data.Age, bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 100],
labels=[f"age_{i}" for i in range(1, 12)]).astype(str)
data.isnull().sum()
data["Fare"] = data.Fare.fillna(data.Fare.median())
data["Embarked"] = data.Embarked.fillna("S")
data["Cabin"].unique()
data["Cabin"] = data["Cabin"].apply(lambda x:x[0] if isinstance(x, str) else "null")
data["Fare"] = pd.cut(np.log1p(data.Fare), bins=[0, 2, 3, 4, 5, 8], labels=[f"fare_{i}" for i in range(1, 6)])
                                                                            
data.loc[data.Parch >= 3, "Parch"] = 3
data = data.rename({"name2": "Name"}, axis=1)
data["Fare"] = data.Fare.astype("str")
one_hot = OneHotEncoder(drop="first").fit(data)
data_onehot = one_hot.transform(data)
one_hot.categories_
train = data_onehot[:num_train, :]
test = data_onehot[num_train:, :]    
model = LogisticRegression()
model.fit(train, train_target)
model.predict(test) 
