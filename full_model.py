import numpy as np
import pickle
from sklearn.model_selection import train_test_split

import pickle
with open('dataset.p', 'rb') as file:
  X, y = pickle.load(file)
X = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

class Dense:
  def __init__(self, input_neurons, output_neurons):
    self.weights = 0.1*np.random.randn(input_neurons, output_neurons)
    self.biases = np.zeros((1, output_neurons))
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases
  def backprop(self, dvalues):
    self.dinputs = np.dot(dvalues, self.weights.T)
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

class ReLu:
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(0, inputs)
  def backprop(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0

class Softmax:
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss:
  def calculate(self, y_pred, y_true):
    samples = len(y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    correct_confidences = y_pred[range(samples), y_true]
    return -np.mean(np.log(correct_confidences))

class Softmax_Loss:
  def __init__(self):
    self.activation = Softmax()
    self.loss = Loss()
  def forward(self, inputs, y_true):
    self.activation.forward(inputs)
    self.output = self.activation.output
    return self.loss.calculate(self.output, y_true)
  def backprop(self, dvalues, y_true):
    samples = len(dvalues)
    self.dinputs = dvalues.copy()
    self.dinputs[range(samples), y_true] -= 1
    self.dinputs /= samples

class Optimizer_GD:
  def __init__(self, learning_rate=0.001, decay=0., momentum=0.):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.momentum = momentum
    self.iterations = 0
  def update_params(self, layer):
    if self.decay:
      self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)
    if self.momentum:
      if not hasattr(layer, 'weight_momentums'):
        layer.weight_momentums = np.zeros_like(layer.weights)
        layer.bias_momentums = np.zeros_like(layer.biases)
      weight_updates = layer.weight_momentums * self.momentum - self.current_learning_rate * layer.dweights
      layer.weight_momentums = weight_updates
      bias_updates = layer.bias_momentums * self.momentum - self.current_learning_rate * layer.dbiases
      layer.bias_momentums = bias_updates
    else:
      weight_updates = -self.current_learning_rate * layer.dweights
      bias_updates = -self.current_learning_rate * layer.dbiases
    layer.weights += weight_updates
    layer.biases += bias_updates
    self.iterations += 1

class Optimizer_RMSProp:
  def __init__(self, learning_rate=0.001, epsilon=1e-7, rho=0.9):
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.rho = rho
  def update_params(self, layer):
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_cache = np.zeros_like(layer.biases)
    layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
    layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
    layer.weights -= self.learning_rate * layer.dweights / np.sqrt(layer.weight_cache + self.epsilon)
    layer.biases -= self.learning_rate * layer.dbiases / np.sqrt(layer.bias_cache + self.epsilon)

class Optimizer_Adam:
  def __init__(self, epsilon=1e-7, learning_rate=0.001, beta_1=0.9, beta_2=0.999):
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.iterations = 0
  def update_params(self, layer):
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_cache = np.zeros_like(layer.biases)
      layer.weight_momentums = np.zeros_like(layer.weights)
      layer.bias_momentums = np.zeros_like(layer.biases)
    layer.weight_momentums = self.beta_1 * layer.weight_momentums  + (1 - self.beta_1) * layer.dweights
    layer.bias_momentums = self.beta_2 * layer.bias_momentums  + (1 - self.beta_2) * layer.dbiases
    weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
    bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_2 ** (self.iterations + 1))
    layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
    layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
    weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
    bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
    layer.weights -= self.learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
    layer.biases -= self.learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

layer_one = Dense(784, 128)
relu_one = ReLu()
layer_two = Dense(128, 64)
relu_two = ReLu()
layer_three = Dense(64, 10)
softmax_loss = Softmax_Loss()
optimizer = Optimizer_Adam()
epochs = 10
batch_size = 32

for epoch in range(epochs):
  indices = np.arange(len(X_train))
  np.random.shuffle(indices)
  X_train, y_train = X_train[indices], y_train[indices]
  
  for i in range(0, len(X_train), batch_size):
    X_batch, y_batch = X_train[i:i+batch_size], y_train[i:i+batch_size]
     
    layer_one.forward(X_batch)
    relu_one.forward(layer_one.output)
    layer_two.forward(relu_one.output)
    relu_two.forward(layer_two.output)
    layer_three.forward(relu_two.output)
    loss = softmax_loss.forward(layer_three.output, y_batch)

    predictions = np.argmax(softmax_loss.output, axis=1)
    accuracy = np.mean(predictions == y_batch)

    softmax_loss.backprop(softmax_loss.output, y_batch)
    layer_three.backprop(softmax_loss.dinputs)
    relu_two.backprop(layer_three.dinputs)
    layer_two.backprop(relu_two.dinputs)
    relu_one.backprop(layer_two.dinputs)
    layer_one.backprop(relu_one.dinputs)

    optimizer.update_params(layer_one)
    optimizer.update_params(layer_two)
    optimizer.update_params(layer_three)

  print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.5f}, Accuracy: {accuracy * 100:.2f}%')

layer_one.forward(X_test)
relu_one.forward(layer_one.output)
layer_two.forward(relu_one.output)
relu_two.forward(layer_two.output)
layer_three.forward(relu_two.output)

test_loss = softmax_loss.forward(layer_three.output, y_test)
print(f'Test Loss: {test_loss:.5f}') 

predictions = np.argmax(softmax_loss.output, axis=1) 
accuracy = np.mean(predictions == y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')