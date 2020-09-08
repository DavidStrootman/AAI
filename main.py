import numpy as np
from typing import List, Optional, Tuple
from collections import Counter
import operator
import math

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

# Classifier algorithm that looks in the range of K nearest neighbours to determine what classification test_point is
def k_nearest_neighbour(k: int, test_point: List[int], training_set: np.ndarray, labels):
    data_point_list: List[Tuple[int, str]] = list((float('inf'), "") for _ in range(k))  # TODO: create empty datapoints or something to compare to

    for i, data_point in enumerate(training_set):
        distance: int = 0
        # calculate score for datapoint
        for j, feature in enumerate(data_point):
            distance += pow(feature - test_point[j], 2)
        for j, val in enumerate(data_point_list):
            if distance < val[0]:
                data_point_list[j] = (distance, labels[i])
                break
        
    output_list = []
    for item in data_point_list:
        output_list.append(item[1])
        
        
    count = Counter(output_list)
    if len(count) > 1:
        most_common = count.most_common(2)[0][0]
        return most_common
    return count.most_common(1)[0][0]


def normalize_features(data_set, input_values=None) -> List[int]:
    max_values = []

    for i in range(len(data_set[0])):  # first data point
        feature_max = 0
        for data_point in data_set:
            value = data_point[i]
            feature_max = value if value > feature_max else feature_max
        if input_values:
            feature_max = input_values[i]
        for data_point in data_set:
            data_point[i] *= 100 / feature_max
        max_values.append(feature_max)
    return max_values


if __name__ == '__main__':
    data_set, data_labels = import_data('dataset1.csv', 2000)
    validation_set, validation_labels = import_data('validation1.csv', 2001)
    days_set, _ = import_data('days.csv', 0)
    highest_k = 0
    highest_value = 0

    normalize_range = normalize_features(data_set)
    normalize_features(validation_set, normalize_range)
    normalize_features(days_set, normalize_range)

    for k in range (1, len(validation_set)):
        classifications = []
        for i in range(len(validation_set)):
            classifications.append(k_nearest_neighbour(k=k, test_point=validation_set[i], training_set=data_set, labels=data_labels))

        matches = 0
        for i, c in enumerate(classifications):
            if classifications[i] == validation_labels[i]:
                matches += 1
        res = matches * 100 / len(validation_labels) 
        if res > highest_value:
            highest_value = res
            highest_k = k
    print("k:", highest_k, " value:", int(highest_value), "%")
    out = []
    for test_point in days_set:
        out.append(k_nearest_neighbour(highest_k, test_point, data_set, data_labels))
    print(out)
