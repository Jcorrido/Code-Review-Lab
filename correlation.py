'''
correlation analysis
using seaborn
'''

# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reading the file to be analyzed
df=pd.read_csv('NFL_Census.csv')

# creating the correlation matrix
corr = df.corr()

# creating the graph with the correlation matrix using seaborn
sns_plot = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
	cmap='PuBu', annot=True, vmin=-1, vmax=1,)

# displaying the matrix
plt.show()

# saving the file
sns_plot.figure.savefig("Correlation_Matrix.png")
