import numpy as np
import random
from collections import Counter
from copy import copy
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt



def import_data(filename: str, year: int) -> Tuple[np.ndarray, List[str]]:
    # (): For some reason this line fixes a bug in liveshare that causes docstrings to be inverted
    """ import a csv file for use by k_nn

    Args:
        filename (str): name of the file to import
        year (int): the year the data in the file is from
    Returns:
        Tuple[np.ndarray, List[str]]: ndarray containing data and a list containing classifications
    """
   
def find_nearest_cluster(d_point: DataPoint, clusters: List[Cluster]) -> Cluster:
    """ finds the nearest cluster to the datapoint

    Args:
        d_point (DataPoint): target 
        clusters (List[Cluster]): list of clusters

    Return:
        nearest_cluster(Cluster): the nearest cluster
    """
 
def k_means(k: int = None, training_set: List[DataPoint] = None, in_clusters: Optional[List[Cluster]] = None):
    """Sorting 

    Args:
        k(int): amount of clusters
        training_set(List[DataPoint]: 
        in_cluster(Optional[List[Cluster]]):

    Return:
        clusters(List[Cluster]):
        changes(bool)
    """

def normalize_features(d_set: np.ndarray, input_values: Optional[List[float]] = None) -> List[float]:
    # (): For some reason this line fixes a bug in liveshare that causes docstrings to be inverted
    """ normalize any data set to a percentage range (0-100)
    if input_values is specified, it will use these values as the maximum of the range

    Args:
        d_set (np.ndarray): data set to normalize
        input_values (List[float], optional): optional list of maximums to overwrite data_set maximum values.
            Defaults to None.

    Returns:
        List[float]: list of feature maximum values
    """

