from tensorflow.keras import datasets #only using tensorflow to load dataset
import numpy as np #easy for computations
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data() #get dataset
#convert an image array into vector
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

class layer_Dense: #dense class
  def __init__(self, input_size, neurons):
    #create weights based on number of input and output neurons
    self.weights = 0.10*np.random.randn(input_size, neurons)
    #create bias for each neuron
    self.biases = np.zeros((1, neurons))
  def forward(self, inputs):
    #store inputs for backprop
    self.inputs = inputs
    #calculate output using dot product
    self.output = np.dot(inputs, self.weights) + self.biases
  def backprop(self, dvalues):
    #use the law of partial derivative of multiplication to calculate gradients
    self.dweights = np.dot(self.inputs.T, dvalues) #transpose so dot product can be computed
    self.dinputs = np.dot(dvalues, self.weights.T) 
    #use the law of partial derivative of addition to calculate bias gradient
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    
class activation_ReLu: #relu class
  def forward(self, inputs):
    #store inputs for backprop
    self.inputs = inputs
    #apply relu function
    self.output = np.maximum(0, inputs)
  def backprop(self, dvalues):
    #copy so no tampering of original gradient from previous layer
    self.dinputs = dvalues.copy()
    #apply relu derivative function, which is 0 if value is 0 or less, but otherwise is the same as values
    self.dinputs[self.inputs <= 0] = 0 

class activation_Softmax: 
  def forward(self, inputs):
    #calculate e**x values for each output, to increase differentiation
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    #normalize the values, to create distribution
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Loss: 
  def calculate(self, output, y):
    #get loss for each sample
    sample_losses = self.forward(output, y)
    #calculate and return loss for entire batch
    data_mean = np.mean(sample_losses)
    return data_mean

class loss_catergorical_crossentropy(Loss):
  def forward(self, y_pred, y_true):
    #number of samples in batch
    samples = len(y_pred)
    #clip to avoid errors
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    if len(y_true.shape) == 1: 
      correct_confidences = y_pred[range(samples), y_true] 
    elif len(y_true.shape) == 2: 
      #if labels one hot-encoded then just multiply the arrays, and then compute the sum of confidences, for each sample
      correct_confidences = np.sum(y_pred*y_true, axis=1) 
    #now use loss formula, to calculate negative log of correct confidences, and get loss for each sample
    negative_log = -np.log(correct_confidences) 
    return negative_log

#read online, to combine these together, makes it easier to calculate gradient, which is true
class activation_softmax_loss_catergorical_crossentropy():
  #code till backprop, just does combined function of softmax and loss 
    def __init__(self):
        self.activation = activation_Softmax()
        self.loss = loss_catergorical_crossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backprop(self, dvalues, y_true):
        samples = len(dvalues)
        #if one-hot encoded, use argmax to turn them into label-encoded
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        #copy so no tampering of original gradient
        self.dinputs = dvalues.copy()
        #calculate gradient for loss function, with respect to softmax, so just subtract one from confidence of correct class
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

#define dense layer with 784 input neurons, for each pixel in picture, and 128 output neurons
dense1 = layer_Dense(784, 128)
#add relu activation function
relu1 = activation_ReLu()
dense2 = layer_Dense(128, 64)
relu2 = activation_ReLu()
dense3 = layer_Dense(64, 10)
#architecture:
#784 --> 128 --> relu --> 64 --> relu --> softmax and loss

softmax_loss = activation_softmax_loss_catergorical_crossentropy() #define softmax_loss instance

#hyperparameters
epochs = 10 #how many times will model see training data
batch_size = 32 #how many samples per batch
learning_rate = 0.001 #how fast to move towards calculated minimum loss function

for epoch in range(epochs):  #define loop to train 
    indices = np.arange(len(X_train)) #get an ordered list of the vectors in X_train
    np.random.shuffle(indices) #shuffle them around
    #apply shuffled indices to shuffle data
    X_train = X_train[indices] 
    y_train = y_train[indices]
    
    for i in range(0, len(X_train), batch_size): #now go through the data, stepping by batch_size
      #extract batch
        X_batch = X_train[i:i+batch_size] 
        y_batch = y_train[i:i+batch_size] 
        
        #forward pass
        dense1.forward(X_batch)
        relu1.forward(dense1.output)
        dense2.forward(relu1.output)
        relu2.forward(dense2.output)
        dense3.forward(relu2.output)
        loss = softmax_loss.forward(dense3.output, y_batch)
        
        #backward pass
        softmax_loss.backprop(softmax_loss.output, y_batch)
        dense3.backprop(softmax_loss.dinputs)
        relu2.backprop(dense3.dinputs)
        dense2.backprop(relu2.dinputs)
        relu1.backprop(dense2.dinputs)
        dense1.backprop(relu1.dinputs)

        #update weights and biases according to gradients
        dense1.weights -= learning_rate * dense1.dweights
        dense1.biases -= learning_rate * dense1.dbiases
        dense2.weights -= learning_rate * dense2.dweights
        dense2.biases -= learning_rate * dense2.dbiases
        dense3.weights -= learning_rate * dense3.dweights
        dense3.biases -= learning_rate * dense3.dbiases
        
    #display current loss after epoch
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

#forward pass on testing data
dense1.forward(X_test)
relu1.forward(dense1.output)
dense2.forward(relu1.output)
relu2.forward(dense2.output)
dense3.forward(relu2.output)
test_loss = softmax_loss.forward(dense3.output, y_test)

print(f'Test Loss: {test_loss:.4f}') #print test loss

predictions = np.argmax(softmax_loss.output, axis=1) #get predictions from softmax
accuracy = np.mean(predictions == y_test) #compare to actual labels to get accuracy
print(f'Test Accuracy: {accuracy * 100:.2f}%') #print accuracy

from PIL import Image #for image manipulation
import matplotlib.pyplot as plt #to display image
#define image path and open image
img_path = 'ENTER YOUR IMAGE PATH HERE'
img = Image.open(img_path)
#plot image
plt.imshow(img) 
plt.show()
img = img.resize((28, 28)) #resize to 28x28, since we trained our model on that
img = img.convert('L') #grayscale the image
img = np.array(img) #now convert to array
img = img.reshape(1, -1) #flatten the image array into a vector
softmax = activation_Softmax() #define softmax layer for predictions

#forward pass on image
dense1.forward(img)
relu1.forward(dense1.output)
dense2.forward(relu1.output)
relu2.forward(dense2.output)
dense3.forward(relu2.output)
softmax.forward(dense3.output)
print(f'Number is {np.argmax(softmax.output)}')
