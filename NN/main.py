import math
import random
import csv

def sigmoid(x):
  return 1 / (1 + math.e **-x)

class Input:
    def __init__ (self, value):
        self.a = value
        self.delta = 0
    
    def calculate_a(self):
        return

    def get_delta_weights_next_layer(self):
        return 0

class Neuron:

    def __init__(self, previous_layer: list, next_layer: list):
        self.bias = get_random()
        self.weights = []
        self.a = 0
        self.delta = 0
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
        self.delta = 0.5*(y_value - self.a)**2
        # print("========")
        # print(0.5 * self.delta**2)

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
    itteration = 0
    for i in range(itterations):
        for hidden_layer in hidden_layers:
            feed_forward_layer(hidden_layer)

        feed_forward_layer(last_layer)
        update_last_layer(last_layer, correct_output)
        update_weights_layer(last_layer, learning_constant)

        for hidden_layer in hidden_layers:
            update_weights_layer(hidden_layer, learning_constant)

    # for i in last_layer:
    #     print(i.a)


def set_input_layer(first_layer: list, input_data: list):
    for i, neuron in enumerate(first_layer):
        neuron.a = input_data[i]


class DataSet:
    def __init__(self, data_points: list, classifications: list):
        self.data_points = data_points
        self.classifications = classifications

class DataPoint:
    def __init__(self, attributes: list, classification_index: int):
        self.attributes = attributes
        self.classification_index = classification_index
        if type(attributes[0]) == str:
            for i, attribute in enumerate(attributes):
                self.attributes[i] = float(attribute)


def import_data_set():
    data_set = []
    classifications = []
    with open("dataset.csv", newline='') as csvfile:
        csv_file = csv.reader(csvfile, delimiter=' ')
        for row in csv_file:
            l = row[0].split(',')
            if l[4] not in classifications:
                classifications.append(l[4])
            data_set.append(DataPoint(l[:4], classifications.index(l[4])))
    return DataSet(data_set, classifications)

def normalize_data(data_set: DataSet):
    max_features = []
    first_run = True
    #get highest valiue for every feature
    for data_point in data_set.data_points:
        if first_run:
            for feature in data_point.attributes:
                max_features.append(feature)
            first_run = False
        else:
            for index, feature in enumerate(data_point.attributes):
                if feature >= max_features[index]:
                    max_features[index] = feature
    for data_point in data_set.data_points:
        for index, feature in enumerate(data_point.attributes):
            data_point.attributes[index] = feature / max_features[index]

    return data_set



def convert_to_output(index: int, list_size: int):
    output = []
    for i in range(list_size):
        if i == index:
            output.append(1)
        else:
            output.append(0)
    return output
            
def feed_forward_result(attributes, first_layer, hidden_layers: list, last_layer):
    set_input_layer(first_layer, attributes)
    for hidden_layer in hidden_layers:
            feed_forward_layer(hidden_layer)
    feed_forward_layer(last_layer)
    results = []
    for neuron in last_layer:
        results.append(int(neuron.a >= 0.5))
    return results
    
if __name__ == "__main__":


    first_layer = []
    hidden_layer_1 = []
    hidden_layer_2 = []
    last_layer = []

    data_set = import_data_set()
    data_set = normalize_data(data_set)
    #add input layer neurons for every attribute
    first_layer.append(Input(0.0))
    first_layer.append(Input(0.0))
    first_layer.append(Input(0.0))
    first_layer.append(Input(0.0))

    #add hidden layer neurons
    for nr_hidden_layers in range(8):
        hidden_layer_1.append(Neuron(first_layer, hidden_layer_2))

    #add hidden layer neurons
    for nr_hidden_layers in range(8):
        hidden_layer_2.append(Neuron(hidden_layer_1, last_layer))

    
    #add output layer neurons fo every classification
    last_layer.append(Neuron(hidden_layer_2, None))
    last_layer.append(Neuron(hidden_layer_2, None))
    last_layer.append(Neuron(hidden_layer_2, None))

    correct_output = [1, 0, 1, 0]

    itteration = 0
    samples = []
    #training the network
    for i in range(10000):
        for data_point in data_set.data_points:
            sample_output = convert_to_output(data_point.classification_index, len(data_set.classifications))
            samples.append(sample_output)
            set_input_layer(first_layer, data_point.attributes)
            train_network(first_layer, [hidden_layer_1, hidden_layer_2], last_layer, 0.1, sample_output, 1)
            itteration += 1
        print("==========" + str(i))
    hits = 0

    #getting validation results
    for i in range(len(data_set.data_points)):
        result = feed_forward_result(data_set.data_points[i].attributes, first_layer, [hidden_layer_1, hidden_layer_2], last_layer)
        
        if samples[i] == result:
            hits += 1
    print(hits / len(data_set.data_points) * 100)
        



