import math
import random
import csv

def sigmoid(x):
    """Returns the sigmoid of x
    """


class Input:
    """A fake neuron that only acts as an input
    """


class NeuralNetwork:
    """A neural network that can be trained
    """


    def add_input_layer(self, nr_inputs: int):
        """Adds an input layer to the neural network.

        Args:
            nr_inputs (int): number of input neuron.
        """
    
    def add_hidden_layer(self, nr_neurons: int):
        """Adds an hidden layer.

        Args:
            nr_neurons (int): number of neurons.
        """


    def add_output_layer(self, nr_neurons: int):
        """Adds an output layer.

        Args:
            nr_neurons (int): number of output neurons.
        """


    def join(self):
        """Joins the hiddenlayers, input layers and output layers together.

        Returns:
            bool: Returns wether the operation was succesfully completed or not.
        """
        

    def train_network(self, data_set, sessions, correct_outputs=None):
        """Trains the network a x amount of times based on the given dataset. 

        Args:
            data_set (DataSet): the training dataset.
            sessions (int): amount of sessions/iterations.
            correct_outputs (list, optional): If this list is given then it 
            will train the network based on these outputs. Defaults to None.
        """

    def train_network_session(self, learning_constant, correct_output):
        """Applies one session of feedforwarding and backward propagation to the neural network.

        Args:
            learning_constant (float): The value that depicts how much the weigths should change by a factor
            correct_output (list): The list that contains the sample result.

        Returns:
            float: returns the error factor.
        """

    def feed_forward_layer(self, layer: list):
        """Executes feed forward on selected neuron list.

        Args:
            layer (list): A list with neurons.
        """
    
    def update_weights_layer(self, layer: list, learning_constant: float):
        """Executes weight update on selected neuron list.

        Args:
            layer (list): A list with neurons.
            learning_constant (float): the constant that should be multiplied with the outcome of the weight update value.
        """

    def set_input_layer(self, input_data: list):
        """Sets the input data to the first layer input neurons.

        Args:
            input_data (list): The sample input data.
        """

    def update_last_layer(self, last_layer: list, y_list: list):
        """Executes update last layer on selected last layer.

        Args:
            last_layer (list): A list with neurons
            y_list (list): A list with output values with the size of last_layer.

        Returns:
            C: Error value.
        """

    def feed_forward_result(self, attributes):
        """Executes feedforward and returns the results based on input attributes.

        Args:
            attributes ([type]): Attributes of a sample.

        Returns:
            list: Returns output list of the feedforwarding of the neural network.
        """

class Neuron:
    """A neuron node in a neural network
    """
        self.generate_weights()

    def generate_weights(self):
        """Add random numbers between -1 & 1 to the weights to initialise them.
        """


    def update_a(self):
        """Calculate the A value for the neuron based on the input and weights.
        """


    def calculate_last(self, y_value):
        """Calculate delta for the last layer for backpropagation and returns the error margin.

        Args:
            y_value ([type]): The expected output of the neuron.

        Returns:
            float: Error margin.
        """


    def get_delta_weights_next_layer(self):
        """Returns the delta of the neuron based on the delta's multiplied by 
        the weights of connected neurons in the next layer for backpropagation of previous layers.

        Returns:
            float: returns delta of the neuron based on the delta's multiplied by 
        the weights of connected neurons in the next layer.
        """


    def back_propegation(self):
        """Does backpropagation by calculating the delta's of previous layers.
        """

    def update_weights(self, learning_constant: float):
        """Update weights based on backpropagation results.

        Args:
            learning_constant (float): The factor that determines how much the weights should change.
        """



class DataSet:
    """Data set class that contains a list with datapoints and corresponding classifications.
    """

class DataPoint:
    """Data point class with attributes and classifaction specified.
    """

def import_data_set(url: str, nr_features):
    """Imports a given data set based on the number of feautures. 
    The classification should come after the last feature"""

def normalize_data(data_set: DataSet):
    """Normalizes the dataset attributes to be a number inbetween 0 and 1.

    Args:
        data_set (DataSet): The given dataset that should be altered.

    Returns:
        DataSet: Returns the altered dataset with normalised datapoints.
    """

def convert_to_output(index: int, list_size: int):
    """Makes a binary list and sets the indexest bit.

    Args:
        index (int): The index of the bit that should be set.
        list_size (int): the size of the output list.

    Returns:
        List: returns a binary list with bit index set.
    """



