import numpy as np
import random
from typing import List, Optional, Tuple
from collections import Counter


def import_data(filename: str, year: int) -> Tuple[np.ndarray, List[str]]:
    #(): For some reason this line fixes a bug in liveshare that causes docstrings to be inverted
    """ import a csv file for use by k_nn

    Args:
        filename (str): name of the file to import
        year (int): the year the data in the file is from
    Returns:
        Tuple[np.ndarray, List[str]]: ndarray containing data and a list containing classifications
    """
    data: np.ndarray = np.genfromtxt(filename, delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                                     converters={5: lambda s: 0 if s == b"-1" else float(s),
                                                 7: lambda s: 0 if s == b"-1" else float(s)})
      
    dates: np.ndarray = np.genfromtxt(filename, delimiter=';', usecols=[0])
    labels: List[str] = []
    for label in dates:
        if label < int(str(year) + '0301'):
            labels.append('winter')
        elif int(str(year) + '0301') <= label < int(str(year) + '0601'):
            labels.append('lente')
        elif int(str(year) + '0601') <= label < int(str(year) + '0901'):
            labels.append('zomer')
        elif int(str(year) + '0901') <= label < int(str(year) + '1201'):
            labels.append('herfst')
        else:  # from 01-12 to end of year
            labels.append('winter')

    return data, labels


def find_nearest_centroid_index(data_point: List[float], centroids: List[List[float]]):
    nearest_centroid_index = []
    shortest_distance = float('-inf')
    for centroid_index, centroid in enumerate(centroids):
        distance: int = 0
        for feature in centroid:
            distance += pow(feature - data_point[centroid_index], 2)
        if distance < shortest_distance:
            shortest_distance = distance
            nearest_centroid_index = centroid_index
    return nearest_centroid_index



def k_means(k: int, training_set: np.ndarray):
    centroids: List[List] = [[]]
    clusters: List[List[List[float]]] = [[[]] for _ in range(k)]
    for _ in range(k):
        centroids.append(random.choice(training_set))  # TODO: might be worthwile to make sure the same point does not get chosen twice
    
    for point in training_set:
        nearest_centroid_index = find_nearest_centroid_index(point, centroids)
        clusters[nearest_centroid_index].append(point)
    
    for cluster_index, cluster in enumerate(clusters):
        feature_list = [0 for _ in range(len(cluster[0]))]
        for point in cluster:
            for feature_index, feature_val in enumerate(point):
                feature_list[feature_index] += feature_val
                
        centroids[cluster_index] = [value / len(cluster) for value in feature_list]  # get mean value
        
            
def normalize_features(d_set: np.ndarray) -> List[float]:
    #(): For some reason this line fixes a bug in liveshare that causes docstrings to be inverted
    """ normalize any data set to a percentage range (0-100)
    if input_values is specified, it will use these values as the maximum of the range

    Args:
        d_set (np.ndarray): data set to normalize
        input_values (List[float], optional): optional list of maximums to overwrite data_set maximum values.
            Defaults to None.
    """
    for feature_index in range(len(d_set[0])):  # first data point
        feature_max: float = 0
        for data_point in d_set:
            value = data_point[feature_index]
            feature_max = value if value > feature_max else feature_max
        for data_point in d_set:
            data_point[feature_index] *= 100 / feature_max


if __name__ == "__main__":
    # import data sets
    data_set, _ = import_data('dataset1.csv', 2000)

    # normalize data
    normalize_features(data_set)
