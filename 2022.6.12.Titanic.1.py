fig, ax = plt.subplots(figsize=(20, 8))

sns.scatterplot(data=train, x="Age", y="Fare", s=70, hue="Survived", alpha=0.5, ax=ax).set_title("Scatter plot of Age vs Fare")
plt.show()
