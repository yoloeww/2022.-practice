probabilities = np.arange(0, 1, 0.01) I = -np.log(probabilities)
plt.plot(probabilities, I)
plt.xlabel("probabilities")
plt.ylabel("Information Content")
