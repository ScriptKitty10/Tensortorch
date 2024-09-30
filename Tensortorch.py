import numpy as np
import time
import random
import matplotlib.pyplot as plt

import ghhgj

import tensorflow as tf
import keras
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Reshape and normalize the data
X_train = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0


# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)

import numpy as np

# Assuming 'images' is your dataset with shape (num_samples, height, width, channels)
# For example, if you have 1000 images in your dataset:
# images.shape would be (1000, 28, 28, 1)

# Concatenate all images along the first axis to create a single array

# Flatten the concatenated array
X_train = X_train.reshape(X_train.shape[0], -1)



# X, y = ghhgj.create_dataset("_classes.csv",'C:\\Users\\tacoc\\Desktop\\quizzzzz\\Fruits by YOLO\\train\\')


# Assuming X is your training data
# Calculate mean and standard deviation for each feature
# mean_values = np.mean(X, axis=0)
# std_dev_values = np.std(X, axis=0)

# # Normalize the data
# normalized_X = (X - mean_values) / std_dev_values


# fruits = list(zip(normalized_X, y))


# class InputLayer():

#     def __init__(self, neurons, activation, input_x, input_y):
        

class DenseLayer():

    def __init__(self, neurons, activation, inputs):
        self.weights = np.random.randn(neurons, inputs) * np.sqrt(1 / inputs)
        self.biases = np.zeros((neurons, 1))
        self.activation = activation
    
    def forward(self, inputs):
        output = np.dot(self.weights, inputs) + self.biases
        if self.activation == "relu":
            z = self.ReLU(output)
        if self.activation == "sigmoid":
            z = self.sigmoid(output)
        if self.activation == "tanh":
            z = self.tanh(output)
        if self.activation == "softmax":
            z = self.softmax(output)
        
        print("Dense Layer forward: z", z)
        print("Z shape", z.shape)
        return output, z
    
    def ReLU(self, inputs):
        return np.maximum(0, inputs)

    def sigmoid(self, inputs):
        return 1/(1+np.exp(-inputs))
    
    def tanh(self, inputs):
        return np.tanh(inputs)

    def softmax(self, inputs):
        inputs = np.squeeze(inputs)
        print(inputs)
        nums = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        probabilities = nums / np.sum(nums, axis=0, keepdims=True)
        return probabilities

class ActivationFunction():

    def __init__(self, type):
        self.type = type
    
    def ReLU(self, inputs):
        return np.maximum(0, inputs)

    def sigmoid(self, inputs):
        return 1/(1+np.exp(-inputs))
    
    def tanh(self, inputs):
        return np.tanh(inputs)
    
    def softmax(self, inputs):
        nums = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
        probabilities = nums / np.sum(nums, axis=0, keepdims=True)
        return probabilities
    

class LossFunction():

    def __init__(self, type):
        self.type = type

    def calculate(self, output, y):
        if self.type == "cce":
            losses = self.categorical_crossentropy(output, y)
        
        elif self.type == "mse":
            losses = self.mean_squared_error(output, y)

        data_loss = np.mean(losses)
        return data_loss
    
    def categorical_crossentropy(self, y_predicted, y_true):
        samples = len(y_predicted)
        y_predicted_clipped = np.clip(y_predicted, 1e-7, 1-1e-7)
        correct_confidences = 0

        if len(y_true.shape) == 1:
            correct_confidences = y_predicted_clipped[range(samples), y_true]
        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_predicted_clipped*y_true, axis=1)

            
            
        epsilon = 1e-8
        
        negative_log_likelihoods = -np.log(correct_confidences + epsilon)
        return negative_log_likelihoods
    
    def mean_squared_error(self, y_predicted, y_true):
        num = np.mean((y_true - y_predicted) ** 2)
        return num
    


class FeedForwardNeuralNet():

    def __init__(self, layers):
        self.layers = layers
        self.size = len(layers)
        self.optimizer = None
        self.loss = None
        self.metrics = None
    
    def feedforward(self, inputs):
       
       z = inputs
       for layer in self.layers:
           output, z = layer.forward(z)
       return output, z
    
    def feedforwardlist(self, inputs):

        z = inputs
        print("Inputs for neural net: ", inputs)
        print("Inputs shape", inputs.shape)
        output_list = []
        zlist = []
        zlist.append(inputs)
        for layer in self.layers:
            output, z = layer.forward(z)
            output_list.append(output)
            zlist.append(z)
        
        print("Zlist", zlist)
        print("Zlist Shape", len(zlist))
        return output_list, zlist
    
    def mse_derivative(self, input, correct):
        n = len(correct)
        return (2 / n) * (input - correct)
    
    def cce_derivative(self, input, correct):
        epsilon = 1e-8
        return -correct / (input + epsilon)
    
    def cce_derivative_with_softmax(self, input, correct):
        correct = np.reshape(correct, (-1,1))
        return input - correct
    
    def sigmoid_derivative(self, input):
        return self.layers[0].sigmoid(input) * (1-self.layers[0].sigmoid(input))
    
    def relu_derivative(self, input):
        return 1.0 * (input > 0)
    
    def tanh_derivative(self, input):
        return 1 - np.tanh(input)**2
    
    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
    
    def fit(self, training_data, learning_rate, epochs, mini_batch_size, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        optimizer = Optimizer(self.optimizer)
        loss = LossFunction(self.loss)

        metrics = Metrics(self.metrics)

        for i in range(epochs):

            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            time.sleep(5)
            for mini_batch in mini_batches:

                new_weights = [np.zeros_like(layer.weights) for layer in self.layers]
                new_biases = [np.zeros_like(layer.biases) for layer in self.layers]
                for x, y in mini_batch:

                    new_w, new_b = model.backprop(x, y)
                    new_weights = [nw+nwg for nw, nwg in zip(new_weights, new_w)]
                    new_biases = [nb+nbg for nb, nbg in zip(new_biases, new_b)]
                optimizer.optimize(model, mini_batch, learning_rate, new_weights, new_biases)

            train_inputs = [input_data for input_data, _ in training_data]
            train_inputs = np.array(train_inputs)
            
            train_predictions = [self.predict(train_inputs)]

            train_predictions = np.array(train_predictions)
            
            train_labels = [label for _, label in training_data]

            train_labels = np.array(train_labels)

            train_loss = loss.calculate(train_predictions, train_labels)

            train_metrics = {'loss': train_loss}

            for metric_name in metrics.metrics:
                if metric_name != 'loss':
                    metric_value = metrics.calculate(metric_name, train_predictions, train_labels)
                    train_metrics[metric_name] = metric_value
            
            metrics.evaluation(**train_metrics)

            train_metrics_str = f"Epoch {i + 1}: Training "
            for metric_name, metric_value in train_metrics.items():
                train_metrics_str += f"{metric_name.capitalize()}: {metric_value}, "
            print(train_metrics_str[:-2])
            if test_data:
                print("Epoch {0}: {1} / {2}".format(i + 1, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(i + 1))

    def evaluate(self, test_data):
        metrics = Metrics(self.metrics)

        for x, y in test_data:
            _, activated_output = self.feedforward(x)
            for metric_name in metrics:
                metric_value = metrics.calculate(metric_name, activated_output, y)
                metrics.evaluation(metric_name, metric_value)
        return metrics.metric_values

    
    def predict(self, input_data):
        predictions = []
        for x in input_data:
            _, output = self.feedforward(x)
            predicted_label = np.argmax(output)
            predictions.append(predicted_label)
        return predictions
    
    def summary(self):
        loss_function = "Loss Function: " + str(self.loss)
        optimizer = "optimizer: " + str(self.optimizer)
                                        
        metrics = "Metrics: " + str(self.metrics)
        layer_info = "\nLayers:\n"
        for i, layer in enumerate(self.layers):
            layer_info += f"Layer {i + 1}: + {layer.activation} \n"
            layer_info += f"Weights: {layer.weights}\n"
            layer_info += f"Biases: {layer.biases}\n"
        return loss_function + "\n" + optimizer + "\n" + metrics + "\n" + layer_info


    def backprop(self, inputs, correct):

        new_weights = [np.zeros_like(layer.weights) for layer in self.layers]
        new_biases = [np.zeros_like(layer.biases) for layer in self.layers]
        output, z = self.feedforwardlist(inputs)


        if self.loss == "mse":
            if self.layers[-1].activation == "sigmoid":
                error = self.mse_derivative(z[-1], correct) * self.sigmoid_derivative(output[-1])
            if self.layers[-1].activation == "relu":
                error = self.mse_derivative(z[-1], correct) * self.relu_derivative(output[-1])
            if self.layers[-1].activation == "tanh":
                error = self.mse_derivative(z[-1], correct) * self.tanh_derivative(output[-1])
        if self.loss == "cce":
            if self.layers[-1].activation == "sigmoid":
                error = self.cce_derivative(z[-1], correct) * self.sigmoid_derivative(output[-1])
            if self.layers[-1].activation == "relu":
                error = self.cce_derivative(z[-1], correct) * self.relu_derivative(output[-1])
            if self.layers[-1].activation == "tanh":
                error = self.cce_derivative(z[-1], correct) * self.tanh_derivative(output[-1])
            if self.layers[-1].activation == "softmax":
                error = self.cce_derivative_with_softmax(z[-1], correct)

        

        

        bias_error = error.mean(axis=1, keepdims=True)

        
        new_weights[-1] = np.dot(error, z[-2].T)

        new_biases[-1] =bias_error


        for l in range(2, self.size + 1):
            zw = output[-l]

            if self.layers[-l].activation == "sigmoid":
                ad = self.sigmoid_derivative(zw)
            if self.layers[-l].activation == "relu":
                ad = self.relu_derivative(zw)
                
            if self.layers[-l].activation == "tanh":
                ad = self.tanh_derivative(zw)
            

            error = np.dot(self.layers[-l+1].weights.T, error) * ad
            bias_error = error.mean(axis=1, keepdims=True)
            new_weights[-l] = np.dot(error, z[-l-1].T)
            new_biases[-l] = bias_error
        
        return (new_weights, new_biases)




        

class Metrics():

    def __init__(self, metrics):
        self.metrics = metrics
        self.metric_values = {metric: [] for metric in self.metrics}
    
    
    
    def calculate(self, metric_name, input_data, labels):
        if metric_name == "accuracy":
            return self.accuracy(input_data, labels)
        if metric_name == "precision":
            return self.precision(input_data, labels)
        if metric_name == "recall":
            return self.recall(input_data, labels)
        if metric_name == "f1_score":
            return self.f1_score(input_data, labels)
        else:
            raise ValueError(f"Metric '{metric_name}' not supported.")

    @staticmethod
    def accuracy(input_data, labels):
        print("Input Data", input_data)
        time.sleep(5)
        print("Labels", labels)
        time.sleep(5)
        correct = sum(1 for predict, label in zip(input_data, labels) if np.all(predict == label))
        return correct / len(labels) if len(labels) > 0 else 0.0
    
    @staticmethod
    def precision(input_data, labels):
        true_positives = sum(1 for predicted_data, label in zip(input_data, labels) if np.all(predicted_data == 1
                                                                                              ) and np.all(label == 1))
        predicted_positives = sum(1 for predicted_data in input_data if np.all(predicted_data == 1))
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    @staticmethod
    def recall(input_data, labels):
        true_positives = sum(1 for predicted_data, label in zip(input_data, labels) if predicted_data == 1 and label == 1)
        actual_positives = sum(1 for label in labels if label == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0.0
    
    @staticmethod
    def f1_score(input_data, labels):
        precision = Metrics.precision(input_data, labels)
        recall = Metrics.recall(input_data, labels)
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def evaluation(self, **kwargs):
        for metric_name in self.metrics:
            if metric_name in kwargs:
                self.metric_values[metric_name].append(kwargs[metric_name]) 
        

class Optimizer():

    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def optimize(self, model, mini_batch, learning_rate, new_weights, new_biases):
        if self.optimizer == "sgd":
            self.SGD(model, learning_rate, mini_batch, new_weights, new_biases)
        
    def SGD(self, model, learning_rate, mini_batch, new_weights, new_biases):
        for layer, nw, nb in zip(model.layers, new_weights, new_biases):
           
            layer.weights -= (learning_rate/len(mini_batch))*nw
         
            layer.biases -= (learning_rate/len(mini_batch))*nb
       
            

        
model = FeedForwardNeuralNet([
    DenseLayer(50, "relu", 784),
    DenseLayer(50, "relu", 50),
    DenseLayer(10, "softmax", 50),
])

model.compile("sgd", "cce", ["accuracy"])


x = list(zip(X_train, y_train))
model.fit(x, 0.03, 10, 8, None)



