"""
Sports Betting Project: Feature Analysis
Author: Trevor Cross
Last Updated: 01/20/22

Analyze features using Pearson Correlation, p-value, and F-score methods.
"""

# ----------------------
# ---Import Libraries---
# ----------------------

# import standard libraries
import numpy as np
import pandas as pd

# import ML libraries
from sklearn.feature_selection import f_classif

# import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# ---Analyze Features---
# ----------------------

# define data path & read
data_path = ""
feature_data = pd.read_csv(data_path)

# get feature correlations
corr_mat = feature_data.corr()

# get F-scores and p-values
f_scores, p_values = f_classif(feature_data[feature_data.columns[:-1]], 
                               feature_data[feature_data.columns[-1]])

# -----------------------
# ---Visualize Results---
# -----------------------

# plot correlation matrix
mask = np.zeros_like(corr_mat)
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(corr_mat, mask=mask, cmap='Greens',annot=True)

# plot p-values
plt.figure()
plt.title("p-values")
plt.xticks(rotation=25)
plt.bar(feature_data.columns[:-1], p_values)

# plot F-scores
plt.figure()
plt.title('F-scores')
plt.xticks(rotation=25)
plt.bar(feature_data.columns[:-1], f_scores)