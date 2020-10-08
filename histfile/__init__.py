import numpy as np
from matplotlib import pyplot

files_ls = []


def init_array(cols=2) -> list:
    """
    Help function to generate empty n-d array
    :param cols: number of columns in input file
    :return: list of lists for each column
    """
    return [[] for _ in range(cols)]


def array_from_file(file_path, separator="\t", as_int=False, as_float=False) -> np.ndarray:
    """
    Function to convert files into n-d numpy arrays
    :param file_path: absolute or relative file path
    :param separator: columns separator, tab by default
    :param as_int: convert array to array of ints
    :param as_float: convert array to array of floats
    :return: final numpy array
    """
    with open(file_path, "r") as file:
        data = file.read()
        data_lines = data.splitlines()
        if not len(data_lines):
            return np.asarray([])
        data_f_line = data_lines[0]
        if not len(data_f_line):
            return np.asarray([])
        cols = len(data_f_line.split(separator))
        result = init_array(cols)

        for row in range(len(data_lines)):
            data_row = data_lines[row].split(separator)
            for idx in range(min(len(data_row), cols)):
                result[idx].append(data_row[idx])

        result = np.asarray(result)

        try:
            if as_int:
                try:
                    result = result.astype(np.int)
                except ValueError:
                    result = result.astype(np.float)
                    result = result.astype(np.int)
            elif as_float:
                result = result.astype(np.float)
        except ValueError:
            pass
    return result


def find_max_by_cols(array: np.ndarray) -> np.ndarray:
    return np.amax(array, axis=1)


def find_min_by_cols(array: np.ndarray) -> np.ndarray:
    return np.amin(array, axis=1)


def make_hist(array: np.ndarray, as_prob=False) -> None:
    """
    Function to draw plots (bar for each value)
    :param array: numpy n-d array
    :param as_prob: draw plot in probability scale
    :return: None
    """
    uniq, counts = np.unique(array, return_counts=True)
    ### debug
    # print(dict(zip(uniq, counts)))
    weights = np.ones(len(array))
    if as_prob:
        weights = np.ones(len(array)) / len(array)
    pyplot.hist(array, alpha=0.5, bins=len(uniq), weights=weights)
    pyplot.show()
