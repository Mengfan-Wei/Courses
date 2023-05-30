#%%
from collections import OrderedDict
from sklearn import manifold, datasets
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

import warnings
warnings.filterwarnings('ignore')
plt.style.use("seaborn-darkgrid")

#%%
n_points = 1000
X, color = datasets.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2

#%%
# Set-up manifold methods

methods = OrderedDict()
methods['LLE'] = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,method='standard')
methods['LTSA'] = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,method='ltsa')
methods['Hessian LLE'] = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,method='hessian')
methods['Modified LLE'] = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,method='modified')
#以上四个算法的method参数不同，method参数表示LLE的具体算法。LocallyLinearEmbedding支持4种LLE算法，分别是
# ‘standard’对应我们标准的LLE算法，默认是’standard’
# ‘ltsa’对应LTSA算法，
# ‘hessian’对应HLLE算法，
# ‘modified’对应MLLE算法。
methods['Isomap'] = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
methods['SE'] = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca', random_state=0)



#%%
# Create figure

fig = plt.figure(figsize=(15, 8))
fig.suptitle("Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), fontsize=14)

# Add 3d scatter plot
ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

# Plot results
for i, (label, method) in enumerate(methods.items()):
    t0 = time()
    Y = method.fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (label, t1 - t0))
    ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')

plt.show()
# %%
