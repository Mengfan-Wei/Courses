#%%
import numpy as np

# Define data
X=np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
y= np.array([
  [1], # Alice
  [0], # Bob
  [0], # Charlie
  [1], # Diana
])


#%%

# Define neural network
class NeuralNetwork(object):
    def __init__(self):
        # Parameters
        self.inputSize = 2
        self.hiddenSize = 3
        self.outputSize = 1

        np.random.seed(42)
        self.W1=np.random.randn(self.inputSize, self.hiddenSize)# (2,3) weight matrix from input to hidden layer
        self.b1=np.random.randn(self.hiddenSize) # (3,)
        self.W2=np.random.randn(self.hiddenSize, self.outputSize)# (3,1) weight matrix from hidden to output layer
        self.b2=np.random.randn(self.outputSize) # (1,)

    def forward(self, X):
        # forward propagation
        # X (n,2)
        self.z1 = np.dot(X, self.W1) + self.b1 # (n,3)
        self.a1 = self.sigmoid(self.z1) # (n,3)
        self.z2 = np.dot(self.a1, self.W2) + self.b2 # (n,1)
        self.a2 = self.sigmoid(self.z2) # (n,1), the output

        return self.a2

    ################################
    """
    !!!To be done
    """

    def sigmoid(self, s):
        # activition function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, a): 
        # The gradient of sigmoid; a is the output of the sigmoid function
        return a*(1-a)

    def backward(self, X, y_true, y_pred, lr=0.1):
        # backpropagation
        '''
        X.shape: (n,2)
        y_true.shape: (n,1)
        y_pred.shape: (n,1)
        '''
        
        # Calculate the gradient of the second layer
        dL_da2 = - (y_true - y_pred) # (n,1), from the MSE, a2 is y_pred
        da2_dz2 = self.sigmoidPrime(self.a2) # (n,1)
        dz2_dw2 = self.a1 # (n,3)
        temp1 = np.expand_dims((dL_da2 * da2_dz2), axis=1).repeat(self.hiddenSize, axis=1) # (n,1) => (n,3,1) 
        temp2 = np.expand_dims(dz2_dw2, axis=2).repeat(self.outputSize, axis=2) # (n,3) => (n,3,1) 
        dL_dw2 = temp1*temp2 # (n,3,1)*(n,3,1) => (n,3,1)
        dL_db2 = dL_da2 * da2_dz2 # (n,1)

        # Calculate the gradient of the first layer
        dz2_da1 = self.W2 # (3,1)
        dL_da1 = np.dot((dL_da2 * da2_dz2) , dz2_da1.T) # (n,3)
        da1_dz1 = self.sigmoidPrime(self.a1) # (n,3)
        dz1_dw1 = X # (n,2)
        temp1 = np.expand_dims((dL_da1 * da1_dz1), axis=1).repeat(self.inputSize, axis=1) # (n,3) => (n,2,3) 
        temp2 = np.expand_dims(dz1_dw1, axis=2).repeat(self.hiddenSize, axis=2) #  (n2) => (n,2,3) 
        dL_dw1 = temp1*temp2 # (n,2,3)*(n,2,3) => (n,2,3)
        dL_db1 = dL_da1 * da1_dz1 # (n,3)

        # Update by the calculated gradient.
        self.W1 -= lr* np.mean(dL_dw1, axis = 0)
        self.b1 -= lr* np.mean(dL_db1, axis = 0)
        self.W2 -= lr* np.mean(dL_dw2, axis = 0)
        self.b2 -= lr* np.mean(dL_db2, axis = 0)

    #####################################

    def train_epoch(self, X, y):
        y_pred = self.forward(X)
        self.backward(X, y, y_pred)
# %%

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return 1/2 * ((y_true - y_pred) ** 2).mean()

# %%

NN = NeuralNetwork()

for i in np.arange(100):#trains the NN 1,000 times
    # print(f"#{str(i)}\n")
    # print(f"Input (scaled):\n{str(X)} \n")
    # print(f"Actual Output:\n{str(y)} \n")
    # print(f"Predicted Output:\n{str(NN.forward(X))}\n")
    print(f"MSE Loss:{mse_loss(y, NN.forward(X))}")
    NN.train_epoch(X,y)
# %%
