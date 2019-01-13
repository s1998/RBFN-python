import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot([i for i in range(10)])
plt.savefig(myfig)
