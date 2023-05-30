# %%
# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# %%
# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# %%
# Fit regression model
"""
!!! To be done
I. Apply DecisionTreeRegressor to fit data
II. Apply plot_tree to show the result of DecisionTreeRegressor
"""

tree_clf = DecisionTreeRegressor(max_depth=3).fit(X, y)
plt.figure(figsize=(15,10))
plt.style.use('seaborn-darkgrid')
plot_tree(tree_clf,filled=True)

# %%
