import math
import random
import csv
import time
class DataSet:
    """Data set class that contains a list with datapoints and corresponding classifications.
    """
    def __init__(self, data_points: list, classifications: list):
        self.data_points = data_points
        self.classifications = classifications

class DataPoint:
    """Data point class with attributes and classifaction specified.
    """
    def __init__(self, attributes: list, classification_index: int):
        self.attributes = attributes
        self.classification_index = classification_index
        if type(attributes[0]) == str:
            for i, attribute in enumerate(attributes):
                self.attributes[i] = float(attribute)

def sigmoid(x):
    """Returns the sigmoid of x
    """
    return 1 / (1 + math.e **-x)

class Input:
    """A fake neuron that only acts as an input
    """
    def __init__ (self, value):
        self.a = value
        self.delta = 0
    
    def calculate_a(self):
        return

    def get_delta_weights_next_layer(self):
        return 0

class NeuralNetwork:
    """A neural network that can be trained
    """
    def __init__(self):
        self.first_layer = []
        self.hidden_layers = []
        self.last_layer = []

    def add_input_layer(self, nr_inputs: int):
        """Adds an input layer to the neural network.

        Args:
            nr_inputs (int): number of input neuron.
        """
        for neuron in range(nr_inputs):
            self.first_layer.append(Input(0.0))
    
    def add_hidden_layer(self, nr_neurons: int):
        """Adds an hidden layer.

        Args:
            nr_neurons (int): number of neurons.
        """
        self.hidden_layers.append([])
        index = len(self.hidden_layers) - 1
        for neuron in range(nr_neurons):
            self.hidden_layers[index].append(Neuron(None, None))

    def add_output_layer(self, nr_neurons: int):
        """Adds an output layer.

        Args:
            nr_neurons (int): number of output neurons.
        """
        for neuron in range(nr_neurons):
            self.last_layer.append(Neuron(None, None))

    def join(self):
        """Joins the hiddenlayers, input layers and output layers together.

        Returns:
            bool: Returns wether the operation was succesfully completed or not.
        """
        index = 0
        last = len(self.hidden_layers) - 1
        if self.hidden_layers != None:
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
            return 1
        else:
            print("No hiddenlayers invalid operation")
            return 0

    def train_network(self, data_set, sessions, correct_outputs=None):
        """Trains the network a x amount of times based on the given dataset. 

        Args:
            data_set (DataSet): the training dataset.
            sessions (int): amount of sessions/iterations.
            correct_outputs (list, optional): If this list is given then it 
            will train the network based on these outputs. Defaults to None.
        """
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
                    sample_output = correct_output[j]
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


        
    def train_network_session(self, learning_constant, correct_output):
        """Applies one session of feedforwarding and backward propagation to the neural network.

        Args:
            learning_constant (float): The value that depicts how much the weigths should change by a factor
            correct_output (list): The list that contains the sample result.

        Returns:
            float: returns the error factor.
        """
        for hidden_layer in self.hidden_layers:
            self.feed_forward_layer(hidden_layer)

        self.feed_forward_layer(self.last_layer)
        c = self.update_last_layer(self.last_layer, correct_output)
        self.update_weights_layer(self.last_layer, learning_constant)

        for hidden_layer in self.hidden_layers:
            self.update_weights_layer(hidden_layer, learning_constant)
        return c 

    def feed_forward_layer(self, layer: list):
        """Executes feed forward on selected neuron list.

        Args:
            layer (list): A list with neurons.
        """
        for neuron in layer:
            neuron.update_a() 
    
    def update_weights_layer(self, layer: list, learning_constant: float):
        """Executes weight update on selected neuron list.

        Args:
            layer (list): A list with neurons.
            learning_constant (float): the constant that should be multiplied with the outcome of the weight update value.
        """
        for neuron in layer:
            neuron.update_weights(learning_constant)

    def set_input_layer(self, input_data: list):
        """Sets the input data to the first layer input neurons.

        Args:
            input_data (list): The sample input data.
        """
        for i, neuron in enumerate(self.first_layer):
            neuron.a = input_data[i]

    def update_last_layer(self, last_layer: list, y_list: list):
        """Executes update last layer on selected last layer.

        Args:
            last_layer (list): A list with neurons
            y_list (list): A list with output values with the size of last_layer.

        Returns:
            C: Error value.
        """
        c = 0
        for i, neuron in enumerate(last_layer):
            c += neuron.calculate_last(y_list[i])
        return abs(c)

    def feed_forward_result(self, attributes):
        """Executes feedforward and returns the results based on input attributes.

        Args:
            attributes ([type]): Attributes of a sample.

        Returns:
            list: Returns output list of the feedforwarding of the neural network.
        """
        self.set_input_layer(attributes)
        for hidden_layer in self.hidden_layers:
            self.feed_forward_layer(hidden_layer)
        self.feed_forward_layer(self.last_layer)
        results = []
        for neuron in self.last_layer:
            results.append(int(neuron.a >= 0.5))
            # results.append(neuron.a)
        return results

    def get_result(self, data_set: DataSet, correct_outputs: list):
        hits = 0
        samples = []
        #getting validation results
        for i, data_point in enumerate(data_set.data_points):
            if correct_outputs == None:
                sample_output = convert_to_output(data_point.classification_index, len(data_set.classifications))  
            else:
                sample_output = correct_output[i]
            samples.append(sample_output)
            result = self.feed_forward_result(data_set.data_points[i].attributes)
            print("===================")
            print("result:   ", result)
            print("Expected: ", samples[i])
            if samples[i] == result:
                hits += 1
        print("Score: ", hits / len(data_set.data_points) * 100)

class Neuron:
    """A neuron node in a neural network
    """

    def __init__(self, previous_layer: list, next_layer: list):
        self.bias = 0.5
        self.weights = []
        self.a = 0
        self.z = 0
        self.delta = 0
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        self.generate_weights()

    def generate_weights(self):
        """Add random numbers between -1 & 1 to the weights to initialise them.
        """
        if(self.previous_layer != None):
            for i in range(len(self.previous_layer)):
                self.weights.append(random.randrange(-100, 100, 1) /100)

    def update_a(self):
        """Calculate the A value for the neuron based on the input and weights.
        """
        self.a = 0
        self.z = 0
        for i, input_i in enumerate(self.previous_layer):
            self.z += self.weights[i] * input_i.a + self.bias
        self.a = sigmoid(self.z)

    def calculate_last(self, y_value):
        """Calculate delta for the last layer for backpropagation and returns the error margin.

        Args:
            y_value ([type]): The expected output of the neuron.

        Returns:
            float: Error margin.
        """
        self.update_a() 
        self.delta = (sigmoid(self.z) * (1 - sigmoid(self.z))) * (y_value - self.a)
        return (y_value - self.a)

    def get_delta_weights_next_layer(self):
        """Returns the delta of the neuron based on the delta's multiplied by 
        the weights of connected neurons in the next layer for backpropagation of previous layers.

        Returns:
            float: returns delta of the neuron based on the delta's multiplied by 
        the weights of connected neurons in the next layer.
        """
        value = 0
        index = self.next_layer[0].previous_layer.index(self)
        for output_neuron in self.next_layer:
            value += output_neuron.weights[index] * output_neuron.delta
        return value

    def back_propegation(self):
        """Does backpropagation by calculating the delta's of previous layers.
        """
        if self.next_layer is not None:
            self.delta = (sigmoid(self.z) * (1 - sigmoid(self.z))) * self.get_delta_weights_next_layer()

    def update_weights(self, learning_constant: float):
        """Update weights based on backpropagation results.

        Args:
            learning_constant (float): The factor that determines how much the weights should change.
        """
        self.back_propegation()
        for i, input_i in enumerate(self.previous_layer):
            self.weights[i] += learning_constant * self.delta * input_i.a
        self.bias += self.delta * learning_constant










def import_data_set(url: str, nr_features):
    """Imports a given data set based on the number of feautures. 
    The classification should come after the last feature"""
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

def normalize_data(data_set: DataSet):
    """Normalizes the dataset attributes to be a number inbetween 0 and 1.

    Args:
        data_set (DataSet): The given dataset that should be altered.

    Returns:
        DataSet: Returns the altered dataset with normalised datapoints.
    """
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
    """Makes a binary list and sets the indexest bit.

    Args:
        index (int): The index of the bit that should be set.
        list_size (int): the size of the output list.

    Returns:
        List: returns a binary list with bit index set.
    """
    output = []
    for i in range(list_size):
        if i == index:
            output.append(1)
        else:
            output.append(0)
    return output


    
if __name__ == "__main__":

    iris_data_set = import_data_set("dataset.csv", 4)
    iris_data_set = normalize_data(iris_data_set)

    iris_validation_set = import_data_set("validation_set.csv", 4)
    iris_validation_set = normalize_data(iris_validation_set)

    adder_data_set = import_data_set("adder_dataset.csv", 3)

    # Adder 
    network=NeuralNetwork()
    network.add_input_layer(3)
    network.add_hidden_layer(3)
    network.add_output_layer(2)
    network.join()
    correct_output = [[0,0], [0,0], [0,0],  [0,1],  [0,1],  [1,0],  [0,1],  [1,0],  [1,0],  [1,1],  [1,1],  [1,1],  [1,1]]
    network.train_network(adder_data_set, 3000, correct_output)
    network.get_result(adder_data_set, correct_output)

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


