import numpy as np
from matplotlib import pyplot
from string import ascii_lowercase, digits
import random

files_ls = []


# using object

class CustomHistData:

    def __init__(self, array=None):
        self.array = array

    def update_array(self, array: np.ndarray):
        """
        Update n-d array from the numpy object
        :param array:
        :return:
        """
        self.array = array

    def upload_data(self, path: str, separator="\t", as_int=False, as_float=False):
        """
        Upload n-d array from the file (with conversion to the numpy array)
        :param path: Absolute file path
        :param separator: columns separator
        :param as_int: convert array to array of ints
        :param as_float: convert array to array of floats
        :return: None
        """
        with open(path, "r") as file:
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
        self.array = result

    def draw_plots(self, as_prob=False):
        """
            Method to draw plots (bar for each value)
            :param as_prob: draw plot in probability scale
            :return: None
            """
        for column in self.array:
            uniq, counts = np.unique(column, return_counts=True)
            ### debug
            # print(dict(zip(uniq, counts)))
            weights = np.ones(len(column))
            if as_prob:
                weights = np.ones(len(column)) / len(column)
            pyplot.hist(column, alpha=0.5, bins=len(uniq), weights=weights)
            pyplot.show()

    def get_min(self, by_row=False):
        if by_row:
            return np.amin(self.array, axis=0)
        else:
            return np.amin(self.array, axis=1)

    def get_max(self, by_row=False):
        if by_row:
            return np.amax(self.array, axis=0)
        else:
            return np.amax(self.array, axis=1)


class Generator:

    def __init__(self, data: str, sample_delta: int = 5000):
        """
        :param data: text to generate data on
        :param sample_delta: the width of the window to sample data
        """
        self.data: str = data
        self.data_set: str = ascii_lowercase + digits + " "
        self.size = len(self.data)
        start = random.randint(0, self.size - sample_delta)
        # generate sample from the text to use in the class methods to reduce processing time for huge datasets
        self.sample = self.data[start:start+sample_delta]
        self.frequencies = self.get_frequency()

    def null_approximation(self, length=100) -> str:
        """
        Generate null approximation (just randomize the data set)
        :param length: int - length of text to generate
        :return: generated text
        """
        return "".join(random.choice(self.data_set) for _ in range(length))

    def basic_approximation(self, length: int = 100) -> str:
        """
        Generate basic approximation, based on the frequency of occurrence char in the text
        :param length: int - length of text to generate
        :return: generated text
        """
        return "".join([np.random.choice(list(self.data_set),
                                         p=[self.get_probability(x) for x in self.data_set]) for _ in range(length)])

    def markov_model(self, level: int = 1, length: int = 100, start_sub: str = "") -> str:
        """
        Generate markov chain
        :param level: int - how many previous chains we need to examine
        :param length: int - length of text to generate
        :param start_sub: str - substring we need to start with
        :return: generated text
        """
        result = start_sub
        while len(result) < length:
            result += np.random.choice(list(self.data_set), replace=True,
                                       p=self.get_probability_pairs(result[-level:]))
        return result

    def get_substrings_len(self, start_sub) -> int:
        """
        Count len of all substrings, starting with provided substring
        :param start_sub:
        :return: int - amount of substrings
        """
        count = 0
        for char in self.data_set:
            count += self.sample.count(start_sub + char)
        return count

    def get_probability(self, substring):
        """

        :param substring: substring probability of which you want to receive
        :return: probability for this character in decimal point format
        """
        count = self.sample.count(substring)
        return count/len(self.sample)

    def get_frequency(self):
        result = {}
        for char in self.data_set:
            result[char] = self.data.count(char)
        return result

    def get_probability_pairs(self, start_subs) -> np.array:
        """
        Get probability of occurrence of this substring with any character of available.
        :param start_subs:
        :return: numpy array of
        """
        result = []
        subs_len = self.get_substrings_len(start_subs)
        if subs_len:
            for char in self.data_set:
                result.append(self.sample.count(start_subs + char)/subs_len)
        else:
            result = [self.get_probability(x) for x in self.data_set]
        result = np.array(result)
        result /= result.sum()
        return result


# functions

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
