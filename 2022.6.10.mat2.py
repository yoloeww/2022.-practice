# sns
sns.scatterplot(x = train['TotalBsmtSF'],y = train['SalePrice_Log'])
# 
plt.figure(figsize = (4, 3))
sns.jointplot(x = train.TotalBsmtSF, y = train.SalePrice_Log)
plt.xlabel('GrLvArea')
plt.ylabel('SalePrice')
plt.title('Basis')
plt.show()
