import numpy as np
from typing import List, Optional, Tuple
from collections import Counter


def import_data(filename: str, year: int) -> Tuple[np.ndarray, List[str]]:  
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


def k_nearest_neighbour(k: int, t_point: List[int], training_set: np.ndarray, labels: List[str]):
    """calculates the k nearest neighbour points to t_point from training_set

    Args:
        k (int):  nearest neighbour count
        t_point (List[int]): a list of features that form a data point
        training_set (np.ndarray): the training set from which to find the nearest points
        labels (List[str]): list of labels to assign to t_point from

    Returns:
        str: returns the classification of the t_point as a string
    """
    data_point_list: List[Tuple[int, str]] = list((float('inf'), "") for _ in range(k))

    for i, data_point in enumerate(training_set):
        distance: int = 0
        # calculate score for datapoint
        for j, feature in enumerate(data_point):
            distance += pow(feature - t_point[j], 2)
        for j, val in enumerate(data_point_list):
            if distance < val[0]:
                data_point_list[j] = (distance, labels[i])
                break

    output_list: List = []
    for item in data_point_list:
        output_list.append(item[1])

    count = Counter(output_list)
    if len(count) > 1:
        most_common = count.most_common(2)[0][0]
        return most_common
    return count.most_common(1)[0][0]


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

    for i in range(len(d_set[0])):  # first data point
        feature_max: float = 0
        for data_point in d_set:
            value = data_point[i]
            feature_max = value if value > feature_max else feature_max
        if input_values:
            feature_max = input_values[i]
        for data_point in d_set:
            data_point[i] *= 100 / feature_max
        max_values.append(feature_max)
    return max_values


def find_best_k(v_set: np.ndarray, v_labels: List[str], t_set: np.ndarray, t_labels: List[str]):
    """find the best k in range of validation set

    Args:
        v_set (np.ndarray): array of points to validate
        v_labels (List[str]): labels for the validation set
        t_set (np.ndarray): array of points to test against
        t_labels (List[str]): labels for the test set

    Returns:
        int: highest k value found
    """
    highest_k: int = 0
    highest_value: float = 0
    for k in range(1, len(v_set)):
        classifications = []
        for i in range(len(v_set)):
            classifications.append(k_nearest_neighbour(k, v_set[i],
                                                       t_set, t_labels))

        matches: int = 0
        for i, c in enumerate(classifications):
            if classifications[i] == v_labels[i]:
                matches += 1
        res: float = matches * 100 / len(v_labels)
        if res > highest_value:
            highest_value = res
            highest_k = k
    print("k:", highest_k, " value:", int(highest_value), "%")
    return highest_k


if __name__ == '__main__':
    # import data sets
    data_set, data_labels = import_data('dataset1.csv', 2000)
    validation_set, validation_labels = import_data('validation1.csv', 2001)
    days_set, _ = import_data('days.csv', 0)

    # normalize data
    normalize_range: List[float] = normalize_features(data_set)
    normalize_features(validation_set, normalize_range)
    normalize_features(days_set, normalize_range)

    # find best k
    best_k: int = find_best_k(validation_set, validation_labels, data_set, data_labels)

    # classify days set
    out: List[str] = []
    for test_point in days_set:
        out.append(k_nearest_neighbour(best_k, test_point, data_set, data_labels))
    print(out)
