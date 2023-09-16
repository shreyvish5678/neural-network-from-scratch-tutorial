import numpy as np
from tqdm import tqdm
import pickle

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

class Sigmoid:
  def forward(self, inputs):
    self.inputs = inputs
    self.output = 1 / (1 + np.exp(-inputs))
  def backprop(self, dvalues):
    self.dinputs = dvalues * self.inputs * (1 - self.inputs)

class Loss_Binary:
  def forward(self, y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    sample_losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    sample_losses = np.mean(sample_losses, axis=-1)
    return -np.mean(np.log(sample_losses))
  
  def backprop(self, dvalues, y_true):
    samples = len(dvalues)
    outputs = len(dvalues[0])
    dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
    self.dinputs = -(y_true / dvalues - (1 - y_true) / (1 - dvalues)) / outputs
    self.dinputs /= samples

class Sigmoid_Loss:
  def __init__(self):
    self.activation = Sigmoid()
    self.loss = Loss_Binary()
  def forward(self, inputs):
    self.activation.forward(inputs)
    self.output = self.activation.output
  def backward(self, dvalues, y_true):
    self.loss.backprop(dvalues, y_true)
    self.dinputs = self.loss.dinputs
    self.activation.backprop(self.dinputs)

class Softmax:
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss_CC:
  def calculate(self, y_pred, y_true):
    samples = len(y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    correct_confidences = y_pred[range(samples), y_true]
    return -np.mean(np.log(correct_confidences))
  
class Softmax_Loss:
  def __init__(self):
    self.activation = Softmax()
    self.loss = Loss_CC()
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
  def __init__(self, learning_rate=0.001, epsilon=1e-7, rho=0.9, decay=0.):
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.rho = rho
    self.decay = decay
    self.current_learning_rate = learning_rate

  def update_params(self, layer):
    if self.decay:
      self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_cache = np.zeros_like(layer.biases)
    layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
    layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
    layer.weights -= self.current_learning_rate * layer.dweights / np.sqrt(layer.weight_cache + self.epsilon)
    layer.biases -= self.current_learning_rate * layer.dbiases / np.sqrt(layer.bias_cache + self.epsilon)
    self.iterations += 1

class Optimizer_Adam:
  def __init__(self, epsilon=1e-7, learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.):
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.iterations = 0
    self.current_learning_rate = learning_rate
    self.decay = decay

  def update_params(self, layer):
    if self.decay:
      self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

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
    layer.weights -= self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
    layer.biases -= self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    self.iterations += 1

def train_model(input_model, X_train, y_train, epochs, batch_size, optimizer):
  model = input_model[0]
  activation = input_model[1]
  if activation == "Softmax":
    loss_function = Softmax_Loss()
  elif activation == "Sigmoid":
    loss_function = Sigmoid_Loss()
  if optimizer == 'SGD':
    optimizer_function = Optimizer_GD()
  elif optimizer == 'RMS':
    optimizer_function = Optimizer_RMSProp()
  elif optimizer == 'Adam':
    optimizer_function = Optimizer_Adam()
  else:
    optimizer_function = optimizer

  for epoch in range(epochs):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]

    for i in tqdm(range(0, len(X_train), batch_size), desc=f'Epoch {epoch+1}', ncols=100):
      X_batch, y_batch = X_train[i:i+batch_size], y_train[i:i+batch_size]
      for num in range(len(model)):
        if num == 0:
          model[num].forward(X_batch)
        else:
          model[num].forward(model[num - 1].output)

      loss = loss_function.forward(model[-1].output, y_batch)
      predictions = np.argmax(loss_function.output, axis=1)
      accuracy = np.mean(predictions == y_batch)
      loss_function.backprop(loss_function.output, y_batch)
      backprop_model = model[::-1]

      for num in range(len(backprop_model)):
        if num == 0:
          backprop_model[num].backprop(loss_function.dinputs)
        else:
          backprop_model[num].backprop(backprop_model[num - 1].dinputs)
      for layer in model:
        if isinstance(layer, Dense):
          optimizer_function.update_params(layer)
    print(f'Loss: {loss:.5f}, Accuracy: {accuracy * 100:.2f}%')

def eval_model(input_model, X_test, y_test):
  model = input_model[0]
  activation = input_model[1]
  if activation == "Softmax":
    loss_function = Softmax_Loss()
  elif activation == "Sigmoid":
    loss_function = Sigmoid_Loss()
  for num in range(len(model)):
    if num == 0:
      model[num].forward(X_test)
    else:
      model[num].forward(model[num - 1].output)
  test_loss = loss_function.forward(model[-1].output, y_test)
  print(f'Test Loss: {test_loss:.5f}') 
  predictions = np.argmax(loss_function.output, axis=1) 
  accuracy = np.mean(predictions == y_test)
  print(f'Test Accuracy: {accuracy * 100:.2f}%')

def predict_model(data, input_model):
  model = input_model[0]
  activation = input_model[1]
  if activation == "Softmax":
    activation_function = Softmax()
  elif activation == "Sigmoid":
    activation_function = Sigmoid()
  for num in range(len(model)):
    if num == 0:
      model[num].forward(data)
    else:
      model[num].forward(model[num - 1].output)
  predictions = activation_function.forward(model[-1].output)
  return predictions

def save_model(file, input_model):
  model = input_model[0]
  model_list = []
  for layer in model:
    if isinstance(layer, Dense):
      model_list.append([layer.weights, layer.biases])
    elif isinstance(layer, ReLu):
      model_list.append([0])
  with open(file, 'wb') as f:
    pickle.dump(model_list, f)
    
def load_model(file):
  model = []
  with open(file, 'rb') as f:
    model_list = pickle.load(f)
  for layer_list in model_list:
    if layer_list[0] == 0:
      model.append(ReLu())
    elif len(layer_list) == 2:
      layer_weights = np.array(layer_list[0])
      layer_biases = np.array(layer_list[1])
      Dense_layer = Dense(layer_weights.shape[0], layer_weights.shape[1])
      Dense_layer.weights = layer_weights
      Dense_layer.biases = layer_biases
      model.append(Dense_layer)
  return model
