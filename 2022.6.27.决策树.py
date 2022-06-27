import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

data = pd.read_csv("train4.csv")
#删除冗余字段
data_new = data.drop(["Name","Ticket","Cabin"],axis=1)
#axis=1,轴向=1即对列进行操作
#使用年龄字段的均值填补缺失值
data_new["Age"] = data_new["Age"].fillna(data_new["Age"].mean())
#将embarked字段中含有缺失值的行删除
data_new.dropna(axis=0)
data_new.info()
#将sex、embarked字段转换为字段属性，可有两种不同的方法
labels = data_new["Embarked"].unique().tolist()
data_new["Embarked"] = data_new["Embarked"].apply(lambda x:labels.index(x))
data_new["Sex"] = (data_new["Sex"]=="male").astype("int")
#至此数据的基本处理已经结束
x = data_new.iloc[:,data_new.columns!="Survived"]
y = data_new.iloc[:,data_new.columns=="Survived"]
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)
for i in [xtrain,xtest,ytrain,ytest]:
    i.index = range(i.shape[0])
clf = DecisionTreeClassifier(random_state=25)
clf = clf.fit(xtrain,ytrain)
score = clf.score(xtest,ytest)
score
clf = DecisionTreeClassifier(random_state=25)
clf = clf.fit(xtrain,ytrain)
score_mean = cross_val_score(clf,x,y,cv=10).mean()
score_mean
