import numpy as np
import random
from collections import Counter
from copy import copy
from typing import List, Tuple, Optional


class DataPoint:
    def __init__(self, features: List, classification: str = None):
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


def k_means(k: int = None, training_set: List[DataPoint] = None, in_clusters: Optional[List[Cluster]] = None):
    changes = False
    clusters = in_clusters

    if not clusters:
        # Initial run
        # Distribute data points from training set
        clusters: List[Cluster] = []
        chosen_points = []
        for _ in range(k):
            while True:  # Simulated do while loop
                random_point = random.choice(training_set)
                if random_point not in chosen_points:
                    clusters.append(Cluster(copy(random_point), []))
                    chosen_points.append(random_point)
                    break
        for point in training_set:
            nearest_cluster = find_nearest_cluster(point, clusters)
            nearest_cluster.data_points.append(point)
    else:
        # subsequent runs
        # Move data points between clusters based on new nearest centroid
        for cluster in clusters:
            for point in cluster.data_points:
                nearest_cluster = find_nearest_cluster(point, clusters)
                if nearest_cluster is not cluster:
                    cluster.data_points.remove(point)
                    nearest_cluster.data_points.append(point)
                    changes = True

    # Recalculate centroids
    for cluster_index, cluster in enumerate(clusters):
        feature_list = [0 for _ in range(len(cluster.centroid.features))]
        for point in cluster.data_points:
            for feature_index, feature_val in enumerate(point.features):
                feature_list[feature_index] += feature_val

        clusters[cluster_index].centroid.features = [value / len(cluster.data_points) for value in feature_list]  # set centroid to mean of data points
    return clusters, changes


# def normalize_features(d_set: np.ndarray):
#     # (): For some reason this line fixes a bug in liveshare that causes docstrings to be inverted
#     """ normalize any data set to a percentage range (0-100)
#     if input_values is specified, it will use these values as the maximum of the range
#
#     Args:
#         d_set (np.ndarray): data set to normalize
#     """
#     for feature_index in range(len(d_set[0])):  # first data point
#         feature_max: float = 0
#         for point in d_set:
#             value = point[feature_index]
#             feature_max = value if value > feature_max else feature_max
#         for point in d_set:
#             point[feature_index] *= 100 / feature_max

def normalize_features(d_set: np.ndarray, input_values: Optional[List[float]] = None) -> List[float]:
    """ normalize any data set to a percentage range (0-100)
    if input_values is specified, it will use these values as the maximum of the range

    Args:
        d_set (np.ndarray): data set to normalize
        input_values (List[float], optional): optional list of maximums to overwrite data_set maximum values.
            Defaults to None.

    Returns:
        List[float]: list of feature maximum values
    """
    max_values: List[float] = []

    for feature_index in range(len(d_set[0])):  # first data point
        feature_max: float = 0
        for data_point in d_set:
            value = data_point[feature_index]
            feature_max = value if value > feature_max else feature_max
        if input_values:
            feature_max = input_values[feature_index]
        for data_point in d_set:
            data_point[feature_index] *= 100 / feature_max
            data_point[feature_index] = int(data_point[feature_index])
        max_values.append(feature_max)
    return max_values


if __name__ == "__main__":
    # seed random
    random.seed(0)
    # import data sets
    data_set, data_labels = import_data('dataset1.csv', 2000)
    validation_set, validation_labels = import_data('validation1.csv', 2001)

    # normalize data
    max_values = normalize_features(data_set)
    normalize_features(validation_set, max_values)

    data_set = list(data_set)
    validation_set = list(validation_set)

    for data_point_index, data_point in enumerate(data_set):
        data_set[data_point_index] = DataPoint(data_point, data_labels[data_point_index])

    for validation_point_index, validation_point in enumerate(validation_set):
        validation_set[validation_point_index] = DataPoint(validation_point,
                                                           validation_labels[validation_point_index])
    for k in range(1, 31):
        clusters, _ = k_means(k=k, training_set=data_set)

        while True:  # simulated do while loop
            clusters, changes_occurred = k_means(in_clusters=clusters)
            if not changes_occurred:
                break

        for cluster in clusters:
            point_classifications = []
            for point in cluster.data_points:
                point_classifications.append(point.classification)
            counter = Counter(point_classifications)

            most_common_classification = counter.most_common(1)
            cluster.centroid.classification = most_common_classification[0][0]
            # cluster.centroid.classification = most_common_classification[0][0] if len(most_common) > 0 else None

        matches = 0
        for validation_point in validation_set:
            nearest_classification = find_nearest_cluster(validation_point, clusters).centroid.classification
            if validation_point.classification is nearest_classification:
                matches += 1

        print(matches * 100 / len(validation_set))

        # for cluster_index, cluster in enumerate(clusters):
        #     print("############# CLUSTER " + str(cluster_index) + " START #############")
        #     for point in cluster.data_points:
        #         print(point.classification)

