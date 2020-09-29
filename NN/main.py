import math
import random

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class Input:
    def __init__ (self, value):
        self.a = value
        self.delta = 0
    
    def calculate_a(self):
        return

    def get_delta_weights_next_layer(self):
        return 0

class Neuron:

    def __init__(self, previous_layer: list, next_layer: list, last_layer: bool):
        self.bias = get_random()
        self.weights = []
        self.a = 0
        self.delta = 0
        self.last = last_layer
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        for i in range(len(self.previous_layer)):
            self.weights.append(get_random())

    def calculate_a(self):
        self.a = 0
        for i, input_i in enumerate(self.previous_layer):
            self.a += self.weights[i] * input_i.a + self.bias
        self.a = sigmoid(self.a)

    def calculate_last(self, y_value):
        self.delta = self.a * (y_value - self.a)

    def get_delta_weights_next_layer(self):
        value = 0
        index = self.next_layer[0].previous_layer.index(self)
        for output_neuron in self.next_layer:
            value += output_neuron.weights[index] * output_neuron.delta
        return value

    def back_propegation(self):
        for i, input_i in enumerate(self.previous_layer):
            input_i.calculate_a()
            input_i.delta = input_i.a * input_i.get_delta_weights_next_layer()

    def update_weights(self, learning_constant: float):
        self.back_propegation()
        for i, input_i in enumerate(self.previous_layer):
            self.weights[i] += learning_constant * self.delta * input_i.a
        self.bias += self.delta * learning_constant


def get_random():
    return random.randrange(-100, 100, 1) /100


def feed_forward_layer(layer: list):
    for neuron in layer:
        neuron.calculate_a()

def update_weights_layer(layer: list, learning_constant: float):
    for neuron in layer:
        neuron.update_weights(learning_constant)

def update_last_layer(last_layer: list, y_list: list):
        for i, neuron in enumerate(last_layer):
            neuron.calculate_last(y_list[i])

def train_network(input_layer, hidden_layers, last_layer, learning_constant, correct_output, itterations):
    
    for i in range(itterations):
        for hidden_layer in hidden_layers:
            feed_forward_layer(hidden_layer)

        feed_forward_layer(last_layer)
        update_last_layer(last_layer, correct_output)
        update_weights_layer(last_layer, learning_constant)

        for hidden_layer in hidden_layers:
            update_weights_layer(hidden_layer, learning_constant)

        for i in last_layer:
            print(i.a)
        print("=======")

if __name__ == "__main__":
    
    first_layer = []
    hidden_layer_1 = []
    hidden_layer_2 = []
    last_layer = []

    #add input layer neurons
    first_layer.append(Input(0.0))
    first_layer.append(Input(0.0))
    first_layer.append(Input(0.0))
    first_layer.append(Input(0.0))

    #add hidden layer neurons
    hidden_layer_1.append(Neuron(first_layer, hidden_layer_2, False))
    hidden_layer_1.append(Neuron(first_layer, hidden_layer_2, False))
    hidden_layer_1.append(Neuron(first_layer, hidden_layer_2, False))

    #add hidden layer neurons
    hidden_layer_2.append(Neuron(hidden_layer_1, last_layer, False)) 
    hidden_layer_2.append(Neuron(hidden_layer_1, last_layer, False)) 
    hidden_layer_2.append(Neuron(hidden_layer_1, last_layer, False)) 
    hidden_layer_2.append(Neuron(hidden_layer_1, last_layer, False))
    
    #add output layer neurons
    last_layer.append(Neuron(hidden_layer_2, None, True))
    last_layer.append(Neuron(hidden_layer_2, None, True))
    last_layer.append(Neuron(hidden_layer_2, None, True))

    correct_output = [1, 0, 1]
    train_network(first_layer, [hidden_layer_1, hidden_layer_2], last_layer, 0.1, correct_output, 100)

