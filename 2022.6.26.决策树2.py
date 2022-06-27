#交叉验证的结果比单个的结果更低，因此要来调整参数，首先想到的是max_depth,因此绘制超参数曲线
score_test=[]
score_train=[]
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25
                                ,max_depth=i+1)
    clf = clf.fit(xtrain,ytrain)
    score_tr = clf.score(xtrain,ytrain)
    score_te = cross_val_score(clf,x,y,cv=10).mean()
    score_train.append(score_tr)
    score_test.append(score_te)
print(max(score_test))
#绘制超参数图像
plt.plot(range(1,11),score_train,color="red",label="train")
plt.plot(range(1,11),score_test,color="blue",label="test")
plt.legend()
plt.xticks(range(1,11))
plt.show()
