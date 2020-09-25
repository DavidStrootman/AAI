import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


class Neuron:

    def __init__(self, inputs: list, outputs: list, weights: list, last_layer: bool):
        super().__init__()
        self.x = inputs
        self.weights = weights
        self.bias = 0
        self.a = 0
        self.delta = 0
        self.last = last_layer
        self.outputs = outputs

    def calculate_a(self):
        self.a = 0
        for i, input in enumerate(self.x):
            self.a += self.weights[i] * input.a + self.bias

    def calculate_last(self):
        y_value = sum(self.outputs) / len(self.outputs)
        self.delta = self.a * (y_value - self.a)

    def back_propegation(self):
        for i, input_i in enumerate(self.x):
            input_i.calculate_a()
            sigma_value = 0
            for output_i in input_i.outputs:
                sigma_value += output_i.delta_j * output_i.weights[output_i.inputs.index(input_i)]
            input_i.delta = input_i.a * sigma_value
            print(input_i.delta)

    def update_weights(self, learning_constant: float):
        if self.last:
            self.calculate_last()
        self.back_propegation()
        for i, input in enumerate(self.x):
            self.weights[i] += learning_constant * self.delta_j * input.a
        self.bias += self.delta_j * learning_constant







