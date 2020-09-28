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
        self.weights = weights
        self.bias = 0
        self.a = 0
        self.wij = []
        self.delta = 0
        self.last = last_layer
        self.outputs = []
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        for i in len(self.previous_layer):
            self.weights.append(get_random())

    def calculate_a(self):
        self.a = 0
        for i, input_i in enumerate(self.previous_layer):
            self.a += self.weights[i] * input_i.a + self.bias

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
        if self.last:
            self.calculate_last()
        self.back_propegation()
        for i, input_i in enumerate(self.previous_layer):
            self.weights[i] += learning_constant * self.delta * input_i.a
        self.bias += self.delta_j * learning_constant


def get_random():
    return random.random(0, 100) /100


if __name__ == "__main__":
    first_layer = []
    hidden_layer_1 = []
    hidden_layer_2 = []
    last_layer = []

    first_layer = [Input(0.1), Input(0.5)]
    hidden_layer_1 = [Neuron(first_layer, hidden_layer_2, False), Neuron(first_layer, hidden_layer_2, False), Neuron(first_layer, hidden_layer_2, False)]
    hidden_layer_2 = [Neuron(hidden_layer_1, last_layer, False), Neuron(hidden_layer_1, last_layer, False), Neuron(hidden_layer_1, last_layer, False), Neuron(hidden_layer_1, last_layer, False)]
    last_layer = [Neuron(hidden_layer_2, None, True), Neuron(hidden_layer_2, None, True), Neuron(hidden_layer_2, None, True)]

