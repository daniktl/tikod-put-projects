from matplotlib import pyplot
import random
from collections import defaultdict
import math
import time
import os
import sys

DATA_DIR = "data"


class CLIColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def custom_print(text: str, style: str = ""):
    if style == "h":
        print(f"{CLIColors.HEADER}{text}{CLIColors.ENDC}")
    elif style == "w":
        print(f"{CLIColors.WARNING}{text}{CLIColors.ENDC}")
    elif style == "e":
        print(f"{CLIColors.FAIL}{text}{CLIColors.ENDC}")
    elif style == "g":
        print(f"{CLIColors.OKGREEN}{text}{CLIColors.ENDC}")
    elif style == "c":
        print(f"{CLIColors.OKCYAN}{text}{CLIColors.ENDC}")
    elif style == "b":
        print(f"{CLIColors.BOLD}{text}{CLIColors.ENDC}")
    else:
        print(f"{text}")


def assignment_wrapper(func):
    def inner():
        start = time.time()
        custom_print(f"Running {func.__name__}...\n", style="b")
        res = func()
        custom_print(f"\nExecution finished in {time.time() - start}s.\n", style="b")
        return res
    return inner


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
            hashtable[tuple(self.tokenized[idx:idx + level])] += 1
            secondary_hashtable[tuple(self.tokenized[idx:idx + level - 1])] += 1
        sum_values = sum(hashtable.values())
        hashtable = {k: v/sum_values for k, v in hashtable.items()}
        sum_values_secondary = sum(secondary_hashtable.values())
        secondary_hashtable = {k: v/sum_values_secondary for k, v in secondary_hashtable.items()}

        self.hashtable = hashtable
        return hashtable, secondary_hashtable


def get_average_word_length(text):
    words = text.split()
    if not words:
        return 0
    full_length = sum([len(word) for word in words])
    custom_print(f"\tAverage word count is: {full_length/len(words)}", style="g")


@assignment_wrapper
def zad1():
    use_sample = False
    generated_length = 300
    for filename in ["norm_wiki_sample.txt"]:
        custom_print(f"[*] Generate for {filename}", style="h")
        with open(os.path.join(DATA_DIR, filename), "r") as file:
            data = file.read()
        generator = Generator(data, mode="char", use_sample=use_sample)
        custom_print("\t[+] null approximation", style="c")
        res = generator.null_approximation(length=generated_length)
        print("\t" + res)
        get_average_word_length(res)
        custom_print("\t[+] basic approximation", style="c")
        res = generator.basic_approximation(length=generated_length)
        print("\t" + res)
        get_average_word_length(res)
        for level in [1, 3, 5]:
            custom_print(f"\t[+] markov chain level {level}", style="c")
            if level == 5:
                start_sub = "probability"
            else:
                start_sub = ""
            res = generator.markov_model(level=5, length=generated_length, start_sub=start_sub)
            print("\t" + res)
            get_average_word_length(res)


def zad2():
    use_sample = True
    generated_length = 200
    for filename in ["norm_wiki_sample.txt"]:
        custom_print(f"\n[*] Generating NLP with: {filename}", style="h")
        generator = Generator(path=os.path.join(DATA_DIR, filename), mode="words", use_sample=use_sample)
        res = generator.basic_approximation(length=generated_length)
        custom_print(f"\t [WORD] basic approximation:", style="c")
        print(f"\t{res}")
        for level in [1, 2]:
            res = generator.markov_model(length=generated_length, level=level)
            custom_print(f"\t [WORD] markov {level}:", style="c")
            print(f"\t{res}")
        res = generator.markov_model(length=generated_length, level=2, start_sub="probability")
        custom_print(f"\t [WORD] markov {level} with \"probability\":", style="c")
        print(f"\t{res}")


def get_entropy(text: str) -> float:
    generator = Generator(data=text, mode="char", use_sample=False)
    res = generator.get_entropy()
    return res


@assignment_wrapper
def zad3a():
    # choose whether you want to use sample to generate hash tables or not
    #       (reduce memory requirements but generates approximate result)
    use_sample = False
    files = ["norm_wiki_en.txt"]
    for filename in files:
        custom_print(f"[*] Processing {filename} with sample: {use_sample}", style="h")
        for mode in ["char", "words"]:
            custom_print(f"\tmode {mode.upper()}", style="b")
            generator = Generator(path=os.path.join(DATA_DIR, filename), mode=mode, use_sample=use_sample)
            # custom_print(f"\tentropy of input_data is: {generator.get_entropy()}", style="g")
            for level in range(1, 6):
                custom_print(f"\tconditional entropy level {level} is: {generator.get_entropy(level=level)}",
                             style="g")
            res = generator.null_approximation(length=10000)
            custom_print(f"\tentropy of null approximation is: {get_entropy(text=res)}", style="g")


@assignment_wrapper
def zad3b():
    use_sample = True
    files = ["norm_wiki_en.txt", "norm_wiki_la.txt"] + [f"sample{x}.txt" for x in range(6)]
    for filename in files:
        custom_print(f"[*] Processing {filename} with sample: {use_sample}", style="h")
        for mode in ["char", "words"]:
            custom_print(f"\tmode {mode.upper()}", style="b")
            generator = Generator(path=os.path.join(DATA_DIR, filename), use_sample=use_sample, mode=mode)
            for level in [0, 1, 2, 3, 4, 5]:
                custom_print(f"\tconditional: {level}\t{generator.get_entropy(level=level)}", style="g")


    # sample0.txt: False (entropy and conditional entropy of the level 1 has the same value)
    # sample1.txt: True
    # sample2.txt: False ? (too much repeating small words)
    # sample3.txt: True
    # sample4.txt: False (conditional entropy of words for 4, 5 is equal to 0,
    #                       conditional entropy of chars iis the same for the level 0-3)
    # sample5.txt: False (each word repeats 16 times??)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_DIR = sys.argv[1]
    if not os.path.exists(DATA_DIR):
        print(f"To run this code be sure to place you data in the directory {DATA_DIR}/ or append"
              f" directory path as argument after the script name: martsich.py [DIRECTORY]")
        quit(1)
    else:
        print(f"Found {DATA_DIR} directory")
    # zad1()
    # zad2()
    zad3a()
    zad3b()
