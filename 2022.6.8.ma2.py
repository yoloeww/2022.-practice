plt.title("House-SalePrice", fontsize=16) # 
plt.xlabel("SalePrice_Log", fontsize=15) # 
plt.ylabel("count", fontsize=15) # 
plt.plot(data.index, data.values, linestyle=':',marker=".",color='r') #":" ,"." ,"r"
plt.show()
