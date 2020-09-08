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
