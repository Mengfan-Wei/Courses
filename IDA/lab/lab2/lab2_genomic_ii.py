#%%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import numpy as np

# %%
# Load data
train = np.loadtxt('train.data', delimiter=' ', dtype = 'object', usecols=(1,2))
test = np.loadtxt('test.data', delimiter=' ', dtype = 'object', usecols=(1,2))

# %%
# convert string data to string array for training data
X_train = train[:,0]
y_train = train[:,1]
X_train = [list(instance) for instance in X_train]
X_train = np.array(X_train)

#%%
# convert string data to string array for test data
X_test = test[:,0]
y_test = test[:,1]
X_test = [list(instance) for instance in X_test]
X_test = np.array(X_test)

#%% 
# Try ordinal encoder

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
# Try one-hot encoder

enc_onehot = OneHotEncoder()
X_train_onehot = enc_onehot.fit_transform(X_train)
X_test_onehot = enc_onehot.transform(X_test)

#%%
'''
LR on one-hot feature.
'''
import time

lr_start = time.perf_counter()
model = LogisticRegression()
model.fit(X_train_onehot,y_train)
lr_end = time.perf_counter()
print(f'Solve the lr problem use {lr_end - lr_start} seconds.')

#%%
accuracy_train = model.score(X_train_onehot, y_train)
accuracy_test = model.score(X_test_onehot, y_test)
print(accuracy_train)
print(accuracy_test)

#%%
'''
Dual problem.
'''
from sklearn.svm import LinearSVC
import time

dual_start = time.perf_counter()
model_dual = LinearSVC()
model_dual.fit(X_train_onehot,y_train)
dual_end = time.perf_counter()

print(f'Solve the dual problem use {dual_end - dual_start} seconds.')

#%%
accuracy_train = model_dual.score(X_train_onehot, y_train)
accuracy_test = model_dual.score(X_test_onehot, y_test)
print(accuracy_train)
print(accuracy_test)

#%%
'''
Original problem.
'''
from sklearn.svm import LinearSVC
import time

original_start = time.perf_counter()
model_original = LinearSVC(dual=False)
model_original.fit(X_train_onehot,y_train)
original_end = time.perf_counter()

print(f'Solve the original problem use {original_end - original_start} seconds.')

#%%
accuracy_train = model_original.score(X_train_onehot, y_train)
accuracy_test = model_original.score(X_test_onehot, y_test)
print(accuracy_train)
print(accuracy_test)

#%%
'''
Use linear kernel.

From LinearSVC:
Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.

From SVC:
The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples.
'''
from sklearn.svm import SVC
import time

linear_start = time.perf_counter()
model_linear = SVC(kernel='linear')
model_linear.fit(X_train_onehot,y_train)
linear_end = time.perf_counter()

print(f'Solve the linear problem use {linear_end - linear_start} seconds.')

#%%
accuracy_train = model_linear.score(X_train_onehot, y_train)
accuracy_test = model_linear.score(X_test_onehot, y_test)
print(accuracy_train)
print(accuracy_test)
