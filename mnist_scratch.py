from tensorflow.keras import datasets 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

class layer_Dense:
  def __init__(self, input_size, neurons):
    self.weights = 0.10*np.random.randn(input_size, neurons)
    self.biases = np.zeros((1, neurons))
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases
  def backprop(self, dvalues):
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dinputs = np.dot(dvalues, self.weights.T) 
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    
class activation_ReLu:
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(0, inputs)
  def backprop(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0 

class activation_Softmax: 
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss: 
  def calculate(self, output, y):
    sample_losses = self.forward(output, y)
    data_mean = np.mean(sample_losses)
    return data_mean

class loss_catergorical_crossentropy(Loss):
  def forward(self, y_pred, y_true):
    samples = len(y_pred)
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    if len(y_true.shape) == 1: 
      correct_confidences = y_pred[range(samples), y_true] 
    elif len(y_true.shape) == 2: 
      correct_confidences = np.sum(y_pred*y_true, axis=1) 
    negative_log = -np.log(correct_confidences) 
    return negative_log

class activation_softmax_loss_catergorical_crossentropy():
    def __init__(self):
        self.activation = activation_Softmax()
        self.loss = loss_catergorical_crossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backprop(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

dense1 = layer_Dense(784, 128)
relu1 = activation_ReLu()
dense2 = layer_Dense(128, 64)
relu2 = activation_ReLu()
dense3 = layer_Dense(64, 10)

softmax_loss = activation_softmax_loss_catergorical_crossentropy() 

epochs = 10 
batch_size = 32 
learning_rate = 0.001 

for epoch in range(epochs): 
    indices = np.arange(len(X_train)) 
    np.random.shuffle(indices) 
    X_train = X_train[indices] 
    y_train = y_train[indices]
    
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size] 
        y_batch = y_train[i:i+batch_size] 
        
        dense1.forward(X_batch)
        relu1.forward(dense1.output)
        dense2.forward(relu1.output)
        relu2.forward(dense2.output)
        dense3.forward(relu2.output)
        loss = softmax_loss.forward(dense3.output, y_batch)
        
        softmax_loss.backprop(softmax_loss.output, y_batch)
        dense3.backprop(softmax_loss.dinputs)
        relu2.backprop(dense3.dinputs)
        dense2.backprop(relu2.dinputs)
        relu1.backprop(dense2.dinputs)
        dense1.backprop(relu1.dinputs)

        dense1.weights -= learning_rate * dense1.dweights
        dense1.biases -= learning_rate * dense1.dbiases
        dense2.weights -= learning_rate * dense2.dweights
        dense2.biases -= learning_rate * dense2.dbiases
        dense3.weights -= learning_rate * dense3.dweights
        dense3.biases -= learning_rate * dense3.dbiases
        
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

dense1.forward(X_test)
relu1.forward(dense1.output)
dense2.forward(relu1.output)
relu2.forward(dense2.output)
dense3.forward(relu2.output)
test_loss = softmax_loss.forward(dense3.output, y_test)

print(f'Test Loss: {test_loss:.4f}') 

predictions = np.argmax(softmax_loss.output, axis=1) 
accuracy = np.mean(predictions == y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

def predict_image(img_path):
  img_path = img_path
  img = Image.open(img_path)
  plt.imshow(img) 
  plt.show()
  img = img.resize((28, 28))
  img = img.convert('L')
  img = np.array(img) 
  img = img.reshape(1, -1) 
  softmax = activation_Softmax()

  dense1.forward(img)
  relu1.forward(dense1.output)
  dense2.forward(relu1.output)
  relu2.forward(dense2.output)
  dense3.forward(relu2.output)
  softmax.forward(dense3.output)
  print(f'Number is {np.argmax(softmax.output)}')
