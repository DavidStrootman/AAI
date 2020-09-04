import numpy as np


def import_data(filename: str, year: int):
    data = np.genfromtxt(filename, delimiter=';', usecols=[1, 2, 3, 4, 5, 6, 7],
                         converters={5: lambda s: 0 if s == b"-1" else float(s),
                                     7: lambda s: 0 if s == b"-1" else float(s)})

    dates = np.genfromtxt(filename, delimiter=';', usecols=[0])
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


import_data('dataset1.csv', 2000)
import_data('validation1.csv', 2001)

# training examples {xi, yi} {attributes, labels}

# D{point, training examples}
# k = amount of points to use for classification
