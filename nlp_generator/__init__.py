from matplotlib import pyplot
import random
from collections import defaultdict
import math


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
                 sample_delta: int = None,
                 mode: str = "words"):
        """
        :param data: text to generate data on
        :param path: absolute of relative path to the file to load data from
        :param use_sample: flag to choose whether use sample or not
        :param sample_delta: the width of the window to sample data
        :param mode: words, char
        """
        if path:
            data = open(path, "r").read()
        if not data:
            raise AttributeError("The data string is empty or None")

        if use_sample:
            size = len(data)
            if not sample_delta:
                sample_delta = size // 5
            start: int = random.randint(0, size - sample_delta)
            # generate sample from the text to use in the class methods to reduce processing time for huge datasets
            self.data: str = data[start:start + sample_delta]
        else:
            self.data: str = data
        self.size: int = len(data)
        self.tokenized: list = []
        self.tokens: list = []
        self.mode: str = mode

        self.hashtable: dict = {}
        if self.mode == "words":
            self.separator = " "
            self.tokenized = self.data.split(" ")
            self.tokens = list(set(self.tokenized))
        else:
            self.separator = ""
            self.tokenized = list(self.data)
            self.tokens = set(self.tokenized)

    def get_entropy(self, level=0) -> float:
        """
        :param level: conditional level (0 - basic entropy)
        :return: entropy of the text
        """
        result = 0
        hashtable, secondary_hashtable = self.get_transition_probabilities(level=level + 1)
        for char, prob in hashtable.items():
            result += prob * math.log2(prob / secondary_hashtable.get(char[:-1], 1))
        return -result

    def get_hashtable_top(self, n: int = 5) -> dict:
        """
        Get top-n elements from the hashtable
        :param n: number of elements to get
        :return: dict {element: probability}
        """
        result = dict(sorted(self.hashtable.items(), key=lambda x: x[1], reverse=True)[:n])
        return result

    def null_approximation(self, length=100) -> str:
        """
        Generate null approximation (just randomize the data set)
        :param length: int - length of text to generate
        :return: generated text
        """
        tokens: list = list(self.tokens)
        result = []
        while len(self.separator.join(result)) < length:
            result += random.choice(tokens)
        return self.separator.join(result)

    def basic_approximation(self, length: int = 100) -> str:
        """
        Generate basic approximation, based on the frequency of occurrence char in the text
        :param length: int - length of text to generate
        :return: generated text
        """
        result = []
        self.get_transition_probabilities()
        while len(self.separator.join(result)) < length:
            tmp = weighted_choice(
                seq=tuple(
                    zip(
                        list(self.tokens),
                        [self.hashtable.get((x,), 0) for x in self.tokens]
                    )
                )
            )
            result.append(tmp)
        return self.separator.join(result)

    def markov_model(self, level: int = 1, length: int = 100, start_sub: str = "") -> str:
        """
        Generate markov chain
        :param level: int - how many previous chains we need to examine
        :param length: int - length of text to generate
        :param start_sub: str - substring we need to start with
        :return: generated text
        """
        result = []
        hashtable, secondary_hashtable = self.get_transition_probabilities(level=level + 1)
        if self.mode == "words":
            result = start_sub.split(self.separator) if start_sub else []
        elif self.mode == "char":
            result = list(start_sub)
        while len(self.separator.join(result)) < length:
            weights = [
                hashtable.get(tuple(result[-level:] + [x]), 0.0) /
                secondary_hashtable.get(tuple(result[-level:]), 0.0) if
                secondary_hashtable.get(tuple(result[-level:]), 0.0) > 0 else 0.0
                for x in self.tokens
            ]

            tmp = weighted_choice(tuple(zip(self.tokens, weights)))

            result.append(tmp)
        return self.separator.join(result)

    def show_top_hashtable(self, n=40) -> None:
        """
        Show the bar plot of the most frequent items in the chain
        :param n: number of items to show on the plot (up to 100)
        :return:
        """
        if n > 50:
            raise AttributeError("Provide number of items to show up to 50 to have readable plot")
        hashtable_top = self.get_hashtable_top(n)
        pyplot.xticks(rotation=75)
        pyplot.bar(list(hashtable_top.keys()), list(hashtable_top.values()))
        pyplot.show()

    def get_transition_probabilities(self, level=1) -> tuple:
        """
        Generate transition probabilities with provided level
        :param level: level of dependence
        :return: (main hashtable, secondary hashtable with lower level dependence for calculations)
        """
        self.hashtable = {}
        hashtable = defaultdict(lambda: 0)
        secondary_hashtable = defaultdict(lambda: 0)
        for idx in range(len(self.tokenized) - level + 1):
            # get probabilities for the current chain level and level higher
            # for idx_tmp in range(level):
            #     hashtable[tuple(self.tokenized[idx:idx+idx_tmp+1])] += 1
            hashtable[tuple(self.tokenized[idx:idx + level])] += 1
            secondary_hashtable[tuple(self.tokenized[idx:idx + level - 1])] += 1

        sum_values = sum(hashtable.values())
        hashtable = {k: v/sum_values for k, v in hashtable.items()}
        sum_values_secondary = sum(secondary_hashtable.values())
        secondary_hashtable = {k: v/sum_values_secondary for k, v in secondary_hashtable.items()}

        self.hashtable = hashtable
        return hashtable, secondary_hashtable
