import numpy as np
import random
from typing import List, Tuple, Optional


class DataPoint:
    def __init__(self, features: List, classification: str=None):
        self.features = features
        self.classification = classification


class Cluster:
    def __init__(self, centroid: DataPoint, data_points: List[DataPoint]):
        self.centroid = centroid
        self.data_points = data_points


def import_data(filename: str, year: int) -> Tuple[np.ndarray, List[str]]:
    # (): For some reason this line fixes a bug in liveshare that causes docstrings to be inverted
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


def find_nearest_cluster(d_point: DataPoint, clusters: List[Cluster]) -> Cluster:
    # TODO: update function to return cluster instead of centroid
    nearest_cluster = None
    shortest_distance = float('inf')

    for cluster in clusters:
        distance: int = 0
        for feature_index, feature in enumerate(cluster.centroid.features):
            distance += pow(feature - d_point.features[feature_index], 2)
        if distance < shortest_distance:
            shortest_distance = distance
            nearest_cluster = cluster

    return nearest_cluster


def k_means(k: int, training_set: List[DataPoint], in_clusters: Optional[List[Cluster]] = None):
    changes = False
    # centroids: List[DataPoint] = []
    clusters = in_clusters
    if not clusters:
        clusters: List[Cluster ]= []
    for _ in range(k):
        # TODO: might be worthwhile to make sure the same point does not get chosen twice
        clusters.append(Cluster(random.choice(training_set), []))
    for point in training_set:
        nearest_cluster = find_nearest_cluster(point, clusters)
        nearest_cluster.data_points.append(point)

    for cluster_index, cluster in enumerate(clusters):
        feature_list = [0 for _ in range(len(cluster.centroid.features))]
        for point in cluster.data_points:
            for feature_index, feature_val in enumerate(point.features):
                feature_list[feature_index] += feature_val
                
        clusters[cluster_index].centroid.features = [value / len(cluster.centroid.features) for value in feature_list]  # set centroid to mean of data points

    return clusters, changes


def normalize_features(d_set: np.ndarray):
    # (): For some reason this line fixes a bug in liveshare that causes docstrings to be inverted
    """ normalize any data set to a percentage range (0-100)
    if input_values is specified, it will use these values as the maximum of the range

    Args:
        d_set (np.ndarray): data set to normalize
    """
    for feature_index in range(len(d_set[0])):  # first data point
        feature_max: float = 0
        for point in d_set:
            value = point[feature_index]
            feature_max = value if value > feature_max else feature_max
        for point in d_set:
            point[feature_index] *= 100 / feature_max


if __name__ == "__main__":
    # import data sets
    data_set, data_labels = import_data('dataset1.csv', 2000)

    # normalize data
    normalize_features(data_set)
    data_set = list(data_set)
    for data_point_index, data_point in enumerate(data_set):
        data_set[data_point_index] = DataPoint(data_point, data_labels[data_point_index])

    clusters, changes_occurred = k_means(4, data_set)

    for cluster_index, cluster in enumerate(clusters):
        print("############# CLUSTER " + str(cluster_index) + " START #############")
        for point in cluster.data_points:
            print(point.classification)
