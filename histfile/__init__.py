import numpy as np
from matplotlib import pyplot
from string import ascii_lowercase, digits
import random
from collections import defaultdict
import time

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


def weighted_choice(seq: tuple):
    """
    Random weighted generator
    :param seq: sequence of items to choose random with weight
                (("John", 34), ("Jack", 56), ...)
    :return: chosen item from the list
    """
    """Calculate accumulative sum"""
    total_prob = sum(item[1] for item in seq)
    """Generate random float number"""
    chosen = random.uniform(0, total_prob)
    cumulative = 0

    for item, probability in seq:
        """Sum probabilities until reaching generated random float"""
        cumulative += probability
        if cumulative > chosen:
            chosen_item = item
            break
    else:
        """If no item found - probably all probabilities are 0 - choose the random item"""
        chosen_item = seq[random.randint(0, len(seq)-1)][0]
    return chosen_item


class Generator:

    def __init__(self, data: str = None,
                 path: str = None,
                 use_sample: bool = True,
                 sample_delta: int = 5000,
                 mode: str = "words"):
        """
        :param data: text to generate data on
        :param path: absolute of relative path to the file to load data from
        :param use_sample: flag to choose whether use sample or not
        :param sample_delta: the width of the window to sample data
        :param mode: words, char
        """
        if data:
            self.data: str = data
        elif path:
            self.data = open(path, "r").read()
        self.tokenized: list = []
        self.tokens: list = []
        self.words_count: int = 0
        self.mode = mode
        self.hashtable: defaultdict = defaultdict(lambda: 0)
        self.size = len(self.data)
        if self.mode == "words":
            self.tokenized = self.get_tokenized()
            self.tokens = set(self.tokenized)
            self.tokens.remove("")
            self.words_count = len(self.tokenized)
            self.generate_hashtable(description=True)
        else:
            self.tokens = list(ascii_lowercase + digits + " ")
        if use_sample:
            start = random.randint(0, self.size - sample_delta)
            # generate sample from the text to use in the class methods to reduce processing time for huge datasets
            self.sample = self.data[start:start + sample_delta]
        else:
            self.sample = self.data
        self.frequencies = self.get_frequency()

    def get_tokenized(self):
        return self.data.split(" ")

    def generate_hashtable(self, level: int = 1, description: bool = False):
        """
        Method to generate hashtable to have easy and fast access to the probabilities
        :param description:
        :param level: level of table to generate (probability for the chains of the words)
        :return: None
        """
        start_time = time.time()
        self.hashtable = defaultdict(lambda: 0)
        if self.mode == "words":
            for item in self.get_pairs(level=level):
                if not item:
                    continue
                self.hashtable[item] += 1
        if description:
            print(f"\t[v]Generated hashtable: level {level},"
                  f" length {len(self.hashtable)} in {time.time() - start_time} s")

    def get_hashtable_top(self, n: int = 5):
        result = dict(sorted(self.hashtable.items(), key=lambda x: x[1], reverse=True)[:n])
        return result

    def get_pairs(self, level: int = 1):
        """
        Generates pairs for the hashtable with level
        :param level: level of table to generate (probability for the chains of the words)
        :return: None
        """
        for idx in range(0, self.words_count - level):
            for level_ in range(1, level+1):
                yield " ".join(self.tokenized[idx:idx + level_])
        # for idx in range(0, self.words_count - level):
        #     yield " ".join(self.tokenized[idx:idx + level])

    def null_approximation(self, length=100) -> str:
        """
        Generate null approximation (just randomize the data set)
        :param length: int - length of text to generate
        :return: generated text
        """
        separator = ""
        if self.mode == "words":
            separator = " "
        result = separator.join(random.choice(self.tokens) for _ in range(length))
        return result

    def basic_approximation(self, length: int = 100) -> str:
        """
        Generate basic approximation, based on the frequency of occurrence char in the text
        :param length: int - length of text to generate
        :return: generated text
        """
        result = []
        separator = ""
        if self.mode == "words":
            separator = " "
            while sum([len(x) for x in result]) < length:
                tmp = weighted_choice(seq=tuple(zip(list(self.tokens),
                                                    [self.hashtable[x] for x in self.tokens])))
                result.append(tmp)
        elif self.mode == "char":
            result = [random.choices(list(self.tokens),
                                     weights=[self.get_probability(x) for x in self.tokens],
                                     k=1)[0]
                      for _ in range(length)]
        return separator.join(result)

    def markov_model(self, level: int = 1, length: int = 100, start_sub: str = "") -> str:
        """
        Generate markov chain
        :param level: int - how many previous chains we need to examine
        :param length: int - length of text to generate
        :param start_sub: str - substring we need to start with
        :return: generated text
        """
        result = []
        separator = ""
        if self.mode == "words":
            separator = " "
            result = start_sub.split(separator) if start_sub else []
            self.generate_hashtable(level=level + 1, description=True)

            while sum([len(x) for x in result]) < length:
                weights = [self.hashtable[" ".join(result[-level:] + [x])] for x in self.tokens]
                tmp = weighted_choice(tuple(zip(self.tokens, weights)))
                result.append(tmp)
        elif self.mode == "char":
            result = list(start_sub)
            while len(result) < length:
                result.append(random.choices(tuple(self.tokens),
                                             weights=self.get_probability_pairs(result[-level:]),
                                             k=1)[0])
        return separator.join(result)

    def get_substrings_len(self, start_sub) -> int:
        """
        Count len of all substrings, starting with provided substring
        :param start_sub:
        :return: int - amount of substrings
        """
        count = 0
        for char in self.tokens:
            count += self.sample.count(start_sub + char)
        return count

    def get_probability(self, substring):
        """

        :param substring: substring probability of which you want to receive
        :return: probability for this character in decimal point format
        """
        count = self.sample.count(substring)
        return count / len(self.sample)

    def get_frequency(self, show=False):
        result = {}
        if self.mode == "char":
            for char in self.tokens:
                result[char] = self.data.count(char)
        elif self.mode == "words":
            result = dict(self.hashtable)
        if show:
            if self.mode == "words":
                result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:40])
                pyplot.xticks(rotation=70)
            pyplot.bar(list(result.keys()), list(result.values()))
            pyplot.show()
        return result

    def get_probability_pairs(self, start_subs: list) -> list:
        """
        Get probability of occurrence of this substring with any character of available.
        :param start_subs:
        :return: list
        """
        result = []
        subs_len = self.get_substrings_len(start_subs)
        if subs_len:
            for char in self.tokens:
                result.append(self.sample.count(start_subs + char) / subs_len)
        else:
            result = [self.get_probability(x) for x in self.tokens]
        res_sum = sum(result)
        finish = [x / res_sum for x in result]
        return finish


# functions

def init_array(cols=2) -> list:
    """
    Help function to generate empty n-d array
    :param cols: number of columns in input file
    :return: list of lists for each column
    """
    return [[] for _ in range(cols)]


def array_from_file(path, separator="\t", as_int=False, as_float=False) -> np.ndarray:
    """
    Function to convert files into n-d numpy arrays
    :param path: absolute or relative file path
    :param separator: columns separator, tab by default
    :param as_int: convert array to array of ints
    :param as_float: convert array to array of floats
    :return: final numpy array
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
