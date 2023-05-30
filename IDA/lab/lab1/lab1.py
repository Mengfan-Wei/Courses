#%%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import numpy as np


# %%
train = np.loadtxt('train.data', delimiter=' ', dtype = 'object', usecols=(1,2))
test = np.loadtxt('test.data', delimiter=' ', dtype = 'object', usecols=(1,2))

# %%
X_train = train[:,0]
y_train = train[:,1]

#%%
X_train = [list(instance) for instance in X_train]
X_train = np.array(X_train)
print(X_train[:2]) 

#%%#%%
X_test = test[:,0]
y_test = test[:,1]
X_test = [list(instance) for instance in X_test]
X_test = np.array(X_test)

#%% 
enc_ord = OrdinalEncoder()
X_train_ord = enc_ord.fit_transform(X_train)
X_test_ord = enc_ord.transform(X_test)

#%%
model = LogisticRegression()
model.fit(X_train_ord,y_train)

#%%
accuracy_train = model.score(X_train_ord, y_train)
accuracy_test = model.score(X_test_ord, y_test)
print(accuracy_train)
print(accuracy_test)

#%% 
enc_onehot = OneHotEncoder()
X_train_onehot = enc_onehot.fit_transform(X_train)
X_test_onehot = enc_onehot.transform(X_test)

#%%
model = LogisticRegression()
model.fit(X_train_onehot,y_train)

#%%
accuracy_train = model.score(X_train_onehot, y_train)
accuracy_test = model.score(X_test_onehot, y_test)
print(accuracy_train)
print(accuracy_test)

# %%
