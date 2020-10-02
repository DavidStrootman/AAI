import math
import random
import csv



#afgeldien is a(1-a)
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
        self.bias = 0.5
        self.weights = []
        self.a = 0
        self.z = 0
        self.delta = 0
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        for i in range(len(self.previous_layer)):
            self.weights.append(get_random())

    def update_a(self):
        self.a = 0
        self.z = 0
        for i, input_i in enumerate(self.previous_layer):
            self.z += self.weights[i] * input_i.a + self.bias
        self.a = sigmoid(self.z)

    def calculate_last(self, y_value):
        self.update_a() 
        self.delta = (sigmoid(self.z) * (1 - sigmoid(self.z))) * (y_value - self.a)
        return (y_value - self.a)

    def get_delta_weights_next_layer(self):
        value = 0
        index = self.next_layer[0].previous_layer.index(self)
        for output_neuron in self.next_layer:
            value += output_neuron.weights[index] * output_neuron.delta
        return value

    def back_propegation(self):
        if self.next_layer is not None:
            self.delta = (sigmoid(self.z) * (1 - sigmoid(self.z))) * self.get_delta_weights_next_layer()

    def update_weights(self, learning_constant: float):
        self.back_propegation()
        for i, input_i in enumerate(self.previous_layer):
            self.weights[i] += learning_constant * self.delta * input_i.a
        self.bias += self.delta * learning_constant


def get_random():
    return random.randrange(-100, 100, 1) /100


def feed_forward_layer(layer: list):
    for neuron in layer:
        neuron.update_a()

def update_weights_layer(layer: list, learning_constant: float):
    for neuron in layer:
        neuron.update_weights(learning_constant)

def update_last_layer(last_layer: list, y_list: list):
        c = 0
        for i, neuron in enumerate(last_layer):
            c += neuron.calculate_last(y_list[i])
        return c**2

def train_network(input_layer, hidden_layers, last_layer, learning_constant, correct_output, itterations):
    itteration = 0
    for i in range(itterations):
        for hidden_layer in hidden_layers:
            feed_forward_layer(hidden_layer)

        feed_forward_layer(last_layer)
        c = update_last_layer(last_layer, correct_output)
        update_weights_layer(last_layer, learning_constant)

        for hidden_layer in hidden_layers:
            update_weights_layer(hidden_layer, learning_constant)
    return c 
    for i in last_layer:
        print(i.a)


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

def print_weights(layers):
    for layer in layers:
        print("======================")
        for node in layer:
            print(node.weights)

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
    for nr_first_layers in range(4):
        first_layer.append(Input(0.0))

    #add hidden layer neurons
    for nr_hidden_layers in range(4):
        hidden_layer_1.append(Neuron(first_layer, hidden_layer_2))

    #add hidden layer neurons
    for nr_hidden_layers in range(4):
        hidden_layer_2.append(Neuron(hidden_layer_1, last_layer))

    #add output layer neurons fo every classification
    for nr_output_layers in range(3):
        last_layer.append(Neuron(hidden_layer_2, None))

    correct_output = []

    itteration = 0
    samples = []
    #training the network
    c = 1
    learning_constant = 0.1
    i = 0
    while i < 1000:
        i += 1
        last_c = c
        c= 0

        for data_point in data_set.data_points:
            sample_output = convert_to_output(data_point.classification_index, len(data_set.classifications))
            samples.append(sample_output)
            set_input_layer(first_layer, data_point.attributes)
            c += train_network(first_layer, [hidden_layer_1, hidden_layer_2], last_layer, learning_constant, sample_output, 1)
        itteration += 1
        print(c / len(data_set.data_points))
        print("==========" + str(i))
        
        #change learning constant based on error
        if c / len(data_set.data_points) > 0.001:
            if not learning_constant > 5.0:
                learning_constant += 0.01
        else:
            if learning_constant > 0.04:
                learning_constant /= 4
        print(learning_constant)

       

    hits = 0

    #getting validation results
    print_weights([hidden_layer_1, hidden_layer_2, last_layer])
    for i in range(len(data_set.data_points)):
        result = feed_forward_result(data_set.data_points[i].attributes, first_layer, [hidden_layer_1, hidden_layer_2], last_layer)
        print("===================")
        print("result:   ", result)
        print("Expected: ", samples[i])
        if samples[i] == result:
            hits += 1
    print(hits / len(data_set.data_points) * 100)
        



