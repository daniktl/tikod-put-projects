# MIT License
#
# Copyright (c) 2020 Daniil Martsich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from matplotlib import pyplot
from string import ascii_lowercase, digits
import random
from collections import defaultdict
import time
import os


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
                                             weights=self.get_probability_pairs("".join(result[-level:])),
                                             k=1)[0])
        return separator.join(result)

    def get_substrings_len(self, start_sub: str) -> int:
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
                result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:35])
                pyplot.xticks(rotation=75)
            pyplot.bar(list(result.keys()), list(result.values()))
            pyplot.show()
        return result

    def get_probability_pairs(self, start_subs: str) -> list:
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


if __name__ == '__main__':
    generated_length = 200
    for filename in ["norm_wiki_sample.txt", "norm_hamlet.txt", "norm_romeo.txt"]:
        print(f"[*] Generate for {filename}")
        generator = Generator(path=os.path.join("data", filename), mode="words", use_sample=False)
        res = generator.basic_approximation(length=generated_length)
        print(res)
        res = generator.markov_model(length=generated_length, level=1)
        print(res)
        res = generator.markov_model(length=generated_length, level=2)
        print(res)
        res = generator.markov_model(length=generated_length, level=2, start_sub="probability")
        print(res)
