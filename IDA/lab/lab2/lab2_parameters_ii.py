#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

#%%
# Load data from file: 4-D data with 3 classes (0, 1, 2)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

#%%
'''
Construct 2-d feature and 2 classes.
'''
from sklearn.preprocessing import StandardScaler

# Select two dimensions.
X_2d = X[:, :2]
# Keep class 1 and 2.
X_2d = X_2d[y > 0]
y_2d = y[y > 0]
y_2d -= 1

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_2d = scaler.fit_transform(X_2d)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r)


#%%
from sklearn.svm import SVC
'''
https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
https://scikit-learn.org/stable/modules/svm.html#svm-kernels
https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
'''
# Construct parameter combinations
C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
classifiers = []
        
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_2d, y_2d)
        classifiers.append((C, gamma, clf))


#%%
# Visualization
#
# Visualize the of performance of different parameter combinations.

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
for (k, (C, gamma, clf)) in enumerate(classifiers):
    # evaluate decision function in a grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')
    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')


#%%
'''
TUNE HYPER-PARAMETERS OF SVM on (X, y)
1. Set the range of the hyperparameters
2. Split data into training set and test set (split ratio=0.8)
3. Train the model on training set and test the model on test set
   with different hyper-parameter combinations 
4. Take the hyper-parameter combination with the best performance 
   as the final model parameter.
'''
C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
train_ratio = 0.8
train_size = int(X.shape[0]*train_ratio)
X_train = X[:train_size]
X_test = X[train_size:]
Y_train = y[:train_size]
Y_test = y[train_size:]

for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_train, Y_train)
        accuracy_train = clf.score(X_train, Y_train)
        accuracy_test = clf.score(X_test, Y_test)
        print('{0:^5}\t{1:^5}\t{2:^5}\t{3:^5}'.format(C, gamma, np.around(accuracy_train, 2), \
            np.around(accuracy_test, 2)))


#%%
'''
TUNE HYPER-PARAMETERS OF SVM on (X, y)
Apply better splitting methods and larger hyperparameter range
'''

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
# for train_index, test_index in cv.split(X,y):
#     print("TRAIN:", train_index, "TEST:", test_index)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, y)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
# %%
'''
Draw heatmap of the validation accuracy as a function of gamma and C.

The score are encoded as colors with the hot colormap which varies from dark
red to bright yellow. As the most interesting scores are all located in the
0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
as to make it easier to visualize the small variations of score values in the
interesting range while not brutally collapsing all the low score values to
the same color.
'''
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
#%%

'''
Draw heat map.
'''

scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()
# %%
