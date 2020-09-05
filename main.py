import numpy as np
from typing import List, Optional, Tuple


def import_data(filename: str, year: int) -> Tuple[np.ndarray, List[str]]:
    data: np.ndarray = np.genfromtxt(filename, delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                                     converters={5: lambda s: 0 if s == b"-1" else float(s),
                                                 7: lambda s: 0 if s == b"-1" else float(s)})

    dates: np.ndarray = np.genfromtxt(filename, delimiter=';', usecols=[0])
    labels = []
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


class DataPoint:
    def __init__(self, classification: Optional[str], features: List):
        self.classification: str = classification
        self.features: List[int] = features


def k_nearest_neighbour(k: int, test_point: List[int], training_set: np.ndarray, labels):
    normalize_features(training_set)
    data_point_list: List[Tuple[int, str]] = list((float('inf'), "") for _ in range(k))  # TODO: create empty datapoints or something to compare to
    for i, data_point in enumerate(training_set):
        distance: int = 0
        for j, feature in enumerate(data_point):
            distance += abs(feature - test_point[i])
        for j, val in enumerate(data_point_list):
            if distance < val[0]:
                data_point_list[i] = (distance, labels[i])


def normalize_features(training_set) -> None:
    for i in range(len(training_set[0])):  # first data point
        feature_max = 0
        for data_point in training_set:
            value = data_point[i]
            feature_max = value if value > feature_max else feature_max

        for data_point in training_set:
            data_point[i] *= 100 / feature_max


if __name__ == '__main__':
    data_set, data_labels = import_data('dataset1.csv', 2000)
    validation_set, _ = import_data('validation1.csv', 2001)
    k_nearest_neighbour(4, data_set[0], validation_set, data_labels)
