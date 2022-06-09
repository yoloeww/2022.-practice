plt.figure(figsize = (6,4))
for val in range(10):
     indeX = train.OverallQual == val
plt.scatter(x = train.GrLivArea.loc[indeX], y = train.SalePrice.loc[indeX]) # 
plt.legend(bbox_to_anchor = [1.1, 1],labels=range(10))#显示图例 图像对应的类别
