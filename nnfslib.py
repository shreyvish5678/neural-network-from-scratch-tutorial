import numpy as np
from tqdm import tqdm
import pickle

class Dense:
  def __init__(self, output_neurons):
    self.output_size = output_neurons
    self.weights = []
    self.biases = np.zeros((1, output_neurons))
  def forward(self, inputs):
    self.inputs = inputs
    _, input_size = inputs.shape
    if len(self.weights) == 0:
      self.weights = 0.1*np.random.randn(input_size, self.output_size)
    self.output = np.dot(inputs, self.weights) + self.biases
  def backprop(self, dvalues):
    self.dinputs = np.dot(dvalues, self.weights.T)
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

class Conv2D:
  def __init__(self, num_kernels, kernel_size):
    self.num_kernels = num_kernels
    self.kernel_size = kernel_size
    self.biases = np.zeros((num_kernels,))
  def forward(self, inputs):
    self.inputs = inputs
    self.filters = 0.1*np.random.randn(self.num_kernels, self.kernel_size, self.kernel_size, self.inputs.shape[3])
    self.output = np.zeros((self.inputs.shape[0], self.inputs.shape[1] - self.filters.shape[1] + 1, self.inputs.shape[2] - self.filters.shape[2] + 1, self.filters.shape[0]))
    for b in range(self.inputs.shape[0]):
      for f in range(self.num_kernels):
        for i in range(self.inputs.shape[1] - self.filters.shape[1] + 1):
          for j in range(self.inputs.shape[2] - self.filters.shape[2] + 1):
            for c in range(self.inputs.shape[3]):
              self.output[b, i, j, f] += np.sum(self.inputs[b, i:i + self.filters.shape[1], j:j + self.filters.shape[2], c] * self.filters[f, :, :, c])     
        self.output[b, :, :, f] += self.biases[f]   
  def backprop(self, dvalues):
    self.dweights = np.zeros_like(self.filters)
    self.dbiases = np.zeros_like(self.biases)
    self.dinputs = np.zeros_like(self.inputs)
    for b in range(self.inputs.shape[0]):
      for f in range(self.num_kernels):
        for i in range(self.inputs.shape[1] - self.filters.shape[1] + 1):
          for j in range(self.inputs.shape[2] - self.filters.shape[2] + 1):
            for c in range(self.inputs.shape[3]):
              self.dweights[f, :, :, c] += dvalues[b, i, j, f] * self.inputs[b, i:i + self.filters.shape[1], j:j + self.filters.shape[2], c]
              self.dinputs[b, i:i + self.filters.shape[1], j:j + self.filters.shape[2], c] += dvalues[b, i, j, f] * self.filters[f, :, :, c]
              self.dbiases[f] += dvalues[b, i, j, f]

class MaxPooling2D:
  def __init__(self, pool_size):
    self.pool_size = pool_size
  def forward(self, inputs):
    self.inputs = inputs
    batch_size, input_height, input_width, input_channels = inputs.shape
    output_height = input_height // self.pool_size
    output_width = input_width // self.pool_size
    self.output = np.zeros((self.inputs.shape[0], self.inputs.shape[1] // self.pool_size, self.inputs.shape[2] // self.pool_size, self.inputs.shape[3]))
    for b in range(self.inputs.shape[0]):
      for c in range(self.inputs.shape[3]):
        for i in range(self.inputs.shape[1] // self.pool_size):
          for j in range(self.inputs.shape[2] // self.pool_size):
            start_i, start_j = i * self.pool_size, j * self.pool_size
            end_i, end_j = start_i + self.pool_size, start_j + self.pool_size
            patch = inputs[b, start_i:end_i, start_j:end_j, c]
            self.output[b, i, j, c] = np.max(patch)
  def backprop(self, dvalues):
    self.dinputs = np.zeros_like(self.inputs)
    for b in range(dvalues.shape[0]):
      for c in range(dvalues.shape[3]):
        for i in range(dvalues.shape[1]):
          for j in range(dvalues.shape[2]):
            start_i, start_j = i * self.pool_size, j * self.pool_size
            end_i, end_j = start_i + self.pool_size, start_j + self.pool_size
            patch = self.inputs[b, start_i:end_i, start_j:end_j, c]
            max_index = np.unravel_index(np.argmax(patch), patch.shape)
            self.dinputs[b, start_i + max_index[0], start_j + max_index[1], c] = dvalues[b, i, j, c]
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
  def calculate(self, y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    sample_losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    sample_losses = np.mean(sample_losses, axis=1)
    return np.mean(sample_losses)
  
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
  def forward(self, inputs, y_true):
    self.activation.forward(inputs)
    self.output = self.activation.output
    return self.loss.calculate(self.output, y_true)
  def backprop(self, dvalues, y_true):
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

class Loss_RMSE:
  def calculate(self, inputs, y_true):
    self.output = inputs
    sample_losses = np.mean((y_true - inputs)**2, axis=0)
    return np.mean(sample_losses)
  
  def backprop(self, dvalues, y_true):
    samples = len(dvalues)
    outputs = len(dvalues[0])
    self.dinputs = -2 * (y_true - dvalues) / outputs
    self.dinputs /= samples

class Linear:
  def forward(self, inputs):
    self.output = inputs
  def backprop(self, dvalues):
    self.dinputs = dvalues

class Regression_Loss:
  def __init__(self):
    self.activation = Linear()
    self.loss = Loss_RMSE()
  def forward(self, inputs, y_true):
    self.activation.forward(inputs)
    self.output = self.activation.output
    return self.loss.calculate(self.output, y_true)
  def backprop(self, dvalues, y_true):
    self.loss.backprop(dvalues, y_true)
    self.dinputs = self.loss.dinputs
    self.activation.backprop(self.dinputs)
  def find_r2score(self, y_true, y_pred):
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2)
    rss = np.sum((y_true - y_pred) ** 2)
    return 1 - (rss / tss)

class Dropout:
  def __init__(self, rate):
    self.rate = 1 - rate
  def forward(self, inputs):
    self.inputs = inputs
    self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
    self.output = inputs * self.binary_mask
  def backprop(self, dvalues):
    self.dinputs = dvalues * self.binary_mask

class Flatten:
    def forward(self, inputs):
        self.original_shape = inputs.shape
        self.output = inputs.reshape((inputs.shape[0], -1))
        return self.output
    def backprop(self, dvalues):
        self.dinputs = dvalues.reshape(self.original_shape)

class GD:
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

class RMSProp:
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

class Adam:
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
  elif activation == "None":
    loss_function = Regression_Loss()
  if optimizer == 'SGD':
    optimizer_function = GD()
  elif optimizer == 'RMS':
    optimizer_function = RMSProp()
  elif optimizer == 'Adam':
    optimizer_function = Adam()
  else:
    optimizer_function = optimizer

  for epoch in range(epochs):
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train, y_train = X_train[indices], y_train[indices]
    for i in tqdm(range(0, len(X_train), batch_size), desc=f'Epoch: {epoch+1}'):
      X_batch, y_batch = X_train[i:i+batch_size], y_train[i:i+batch_size]
      for num in range(len(model)):
        if num == 0:
          model[num].forward(X_batch)
        else:
          model[num].forward(model[num - 1].output)

      loss = loss_function.forward(model[-1].output, y_batch)
      if isinstance(loss_function, Regression_Loss):
        y_batch_flatten = y_batch.flatten()
        y_pred_flatten = model[-1].output.flatten()
        accuracy = max(0, loss_function.find_r2score(y_batch_flatten, y_pred_flatten))
        user = f'Loss: {loss:.5f}, Accuracy: {accuracy * 100:.2f}%'
      else:
        predictions = np.argmax(loss_function.output, axis=1)
        accuracy = np.mean(predictions == y_batch)
        user = f'Loss: {loss:.5f}, Accuracy: {accuracy * 100:.2f}%'
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
      print(user)

def eval_model(input_model, X_test, y_test):
  model = input_model[0]
  activation = input_model[1]
  if activation == "Softmax":
    loss_function = Softmax_Loss()
  elif activation == "Sigmoid":
    loss_function = Sigmoid_Loss()
  elif activation == "None":
    loss_function = Regression_Loss()
  for num in range(len(model)):
    if num == 0:
      model[num].forward(X_test)
    else:
      model[num].forward(model[num - 1].output)
  test_loss = loss_function.forward(model[-1].output, y_test)
  print(f'Test Loss: {test_loss:.5f}') 
  if isinstance(loss_function, Regression_Loss):
    y_test_flatten = y_test.flatten()
    y_pred_flatten = model[-1].output.flatten()
    accuracy = loss_function.find_r2score(y_test_flatten, y_pred_flatten)
  else:
    predictions = np.argmax(loss_function.output, axis=1) 
    accuracy = max(0, np.mean(predictions == y_test))
  print(f'Test Accuracy: {accuracy * 100:.2f}%')

def predict_model(data, input_model):
  model = input_model[0]
  activation = input_model[1]
  if activation == "Softmax":
    activation_function = Softmax()
  elif activation == "Sigmoid":
    activation_function = Sigmoid()
  elif activation == "None":
    activation_function = Linear()
  for num in range(len(model)):
    if num == 0:
      model[num].forward(data)
    else:
      model[num].forward(model[num - 1].output)
  activation_function.forward(model[-1].output)
  predictions = activation_function.output
  return predictions

def save_model(file, input_model):
  model = input_model[0]
  model_list = []
  for layer in model:
    if isinstance(layer, Dense):
      model_list.append([layer.weights, layer.biases])
    elif isinstance(layer, ReLu):
      model_list.append([])
    elif isinstance(layer, Dropout):
      model_list.append([1 - layer.rate])
  with open(file, 'wb') as f:
    pickle.dump(model_list, f)
    
def load_model(file):
  model = []
  with open(file, 'rb') as f:
    model_list = pickle.load(f)
  for layer_list in model_list:
    if len(layer_list) == 0:
      model.append(ReLu())
    elif len(layer_list) == 1:
      layer_rate = layer_list[0]
      model.append(Dropout(layer_rate))
    elif len(layer_list) == 2:
      layer_weights = np.array(layer_list[0])
      layer_biases = np.array(layer_list[1])
      Dense_layer = Dense(layer_weights.shape[1])
      Dense_layer.weights = layer_weights
      Dense_layer.biases = layer_biases
      model.append(Dense_layer)
  return model
