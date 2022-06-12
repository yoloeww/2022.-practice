fig,ax=plt.subplots(1,2,figsize=(20,8))#分成两个
sns.countplot(data=train, x="Parch", ax=ax[0]).set_title("Count plot for Parch")
# SibSp
sns.countplot(data=train, x="SibSp", ax=ax[1]).set_title("Count plot for SibSp")

plt.show()
