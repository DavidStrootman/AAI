import math
import random
import csv
import time
from typing import List

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

def sigmoid(x) -> float:
    return 1 / (1 + math.e **-x)

class Input:
    def __init__ (self, value):
        self.a = value
        self.delta = 0

class NeuralNetwork:
    def __init__(self):
        self.first_layer = []
        self.hidden_layers = []
        self.last_layer = []

    def add_input_layer(self, nr_inputs: int) -> None:
        for neuron in range(nr_inputs):
            self.first_layer.append(Input(0.0))
    
    def add_hidden_layer(self, nr_neurons: int) -> None:
        self.hidden_layers.append([])
        index = len(self.hidden_layers) - 1
        for neuron in range(nr_neurons):
            self.hidden_layers[index].append(Neuron(None, None))

    def add_output_layer(self, nr_neurons: int) -> None:
        for neuron in range(nr_neurons):
            self.last_layer.append(Neuron(None, None))

    def join(self) -> None:
        index = 0
        last = len(self.hidden_layers) - 1
        if self.hidden_layers:
            for layer in self.hidden_layers:
                for neuron in layer:
                    if index == 0:
                        neuron.previous_layer = self.first_layer
                    elif index <= last:
                        neuron.previous_layer = self.hidden_layers[index - 1]
                    if index >= last:
                        neuron.next_layer = self.last_layer
                    else:
                        neuron.next_layer = self.hidden_layers[index + 1]
                        
                index += 1
            for neuron in self.last_layer:
                neuron.previous_layer = self.hidden_layers[len(self.hidden_layers) - 1]

            for neuron in self.last_layer:
                neuron.generate_weights()
            for layer in self.hidden_layers:
                for neuron in layer:
                    neuron.generate_weights()
        else:
            raise RuntimeError("No hidden layers: invalid operation")

    def train_network(self, data_set, sessions, correct_outputs=None) -> None:
        itteration = 0
        samples = []
        c = 1
        learning_constant = 0.1
        for i in range(sessions):
            last_c = c
            c= 0

            for j, data_point in enumerate(data_set.data_points):
                if correct_outputs == None:
                    sample_output = convert_to_output(data_point.classification_index, len(data_set.classifications))  
                else:
                    sample_output = correct_outputs[j]
                self.set_input_layer(data_point.attributes)
                c += self.train_network_session(learning_constant, sample_output)
            itteration += 1
            print("==========" + str(i), end="   ")
            
            #change learning constant based on error
            if c / len(data_set.data_points) > last_c / len(data_set.data_points):
                if not learning_constant > 2:
                    learning_constant *= 1.10
            else:
                if not learning_constant < 0.1:
                    learning_constant *= 0.90
            print("Learning constant: ", learning_constant, end="   ")
            print("Error value:       ", c / len(data_set.data_points))


        
    def train_network_session(self, learning_constant, correct_output) -> float:
        for hidden_layer in self.hidden_layers:
            self.feed_forward_layer(hidden_layer)

        self.feed_forward_layer(self.last_layer)
        c = self.update_last_layer(self.last_layer, correct_output)
        self.update_weights_layer(self.last_layer, learning_constant)

        for hidden_layer in self.hidden_layers:
            self.update_weights_layer(hidden_layer, learning_constant)
        return c 

    def feed_forward_layer(self, layer: list) -> None:
        for neuron in layer:
            neuron.update_a() 
    
    def update_weights_layer(self, layer: list, learning_constant: float) -> None:
        for neuron in layer:
            neuron.update_weights(learning_constant)

    def set_input_layer(self, input_data: list) -> None:
        for i, neuron in enumerate(self.first_layer):
            neuron.a = input_data[i]

    def update_last_layer(self, last_layer: list, y_list: list) -> float:
        c = 0
        for i, neuron in enumerate(last_layer):
            c += neuron.calculate_last(y_list[i])
        return abs(c)

    def feed_forward_result(self, attributes) -> List[int]:
        self.set_input_layer(attributes)
        for hidden_layer in self.hidden_layers:
            self.feed_forward_layer(hidden_layer)
        self.feed_forward_layer(self.last_layer)
        results = []
        for neuron in self.last_layer:
            results.append(int(neuron.a >= 0.5))
            # results.append(neuron.a)
        return results

    def get_result(self, data_set: DataSet, correct_outputs: list) -> None:
        hits = 0
        samples = []
        #getting validation results
        for i, data_point in enumerate(data_set.data_points):
            if correct_outputs == None:
                sample_output = convert_to_output(data_point.classification_index, len(data_set.classifications))  
            else:
                sample_output = correct_outputs[i]
            samples.append(sample_output)
            result = self.feed_forward_result(data_set.data_points[i].attributes)
            print("===================")
            print("result:   ", result)
            print("Expected: ", samples[i])
            if samples[i] == result:
                hits += 1
        print("Score: ", hits, "/", len(data_set.data_points))
        print("Score: ", hits / len(data_set.data_points) * 100)

class Neuron:

    def __init__(self, previous_layer: list, next_layer: list):
        self.bias = 0.5
        self.weights = []
        self.a = 0
        self.z = 0
        self.delta = 0
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        self.generate_weights()

    def generate_weights(self) -> None:
        if(self.previous_layer != None):
            for i in range(len(self.previous_layer)):
                self.weights.append(random.randrange(-100, 100, 1) /100)

    def update_a(self) -> None:
        self.a = 0
        self.z = 0
        for i, input_i in enumerate(self.previous_layer):
            self.z += self.weights[i] * input_i.a + self.bias
        self.a = sigmoid(self.z)

    def calculate_last(self, y_value) -> float:
        self.update_a() 
        self.delta = (sigmoid(self.z) * (1 - sigmoid(self.z))) * (y_value - self.a)
        return y_value - self.a

    def get_delta_weights_next_layer(self) -> float:
        value = 0
        index = self.next_layer[0].previous_layer.index(self)
        for output_neuron in self.next_layer:
            value += output_neuron.weights[index] * output_neuron.delta
        return value

    def back_propegation(self) -> None:
        if self.next_layer is not None:
            self.delta = (sigmoid(self.z) * (1 - sigmoid(self.z))) * self.get_delta_weights_next_layer()

    def update_weights(self, learning_constant: float) -> None:
        self.back_propegation()
        for i, input_i in enumerate(self.previous_layer):
            self.weights[i] += learning_constant * self.delta * input_i.a
        self.bias += self.delta * learning_constant

def import_data_set(url: str, nr_features) -> DataSet:
    data_set = []
    classifications = []
    with open(url, newline='') as csvfile:
        csv_file = csv.reader(csvfile, delimiter=' ')
        for row in csv_file:
            l = row[0].split(',')
            if l[nr_features] not in classifications:
                classifications.append(l[nr_features])
            data_set.append(DataPoint(l[0:nr_features], classifications.index(l[nr_features])))
    return DataSet(data_set, classifications)

def normalize_data(data_set: DataSet) -> DataSet:
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

def convert_to_output(index: int, list_size: int) -> List[int]:
    return [1 if i == index else 0 for i in range(list_size)] # Pythonic magic



    
if __name__ == "__main__":

    iris_data_set = import_data_set("dataset.csv", 4)
    iris_data_set = normalize_data(iris_data_set)

    iris_validation_set = import_data_set("validation_set.csv", 4)
    iris_validation_set = normalize_data(iris_validation_set)

    adder_data_set = import_data_set("adder_dataset.csv", 3)
    expected_output_adder = [[0,0], [0,0], [0,0],  [0,1],  [0,1],  [1,0],  [0,1],  [1,0],  [1,0],  [1,1],  [1,1],  [1,1],  [1,1]]

    # Adder 
    network=NeuralNetwork()
    network.add_input_layer(3)
    network.add_hidden_layer(3)
    network.add_output_layer(2)
    network.join()
    network.train_network(adder_data_set, 3000, expected_output_adder)
    network.get_result(adder_data_set, expected_output_adder)

    time.sleep(3)

    # Iris data
    network=NeuralNetwork()
    network.add_input_layer(4)
    network.add_hidden_layer(4)
    network.add_hidden_layer(4)
    network.add_output_layer(3)
    network.join()
    network.train_network(iris_data_set, 1000, None)
    network.get_result(iris_validation_set, None)


