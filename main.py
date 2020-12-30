import os
import time

import nlp_generator


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


def get_average_word_length(text):
    words = text.split()
    if not words:
        return 0
    full_length = sum([len(word) for word in words])
    custom_print(f"\tAverage word count is: {full_length/len(words)}", style="g")


@assignment_wrapper
def zad1():
    generated_length = 300
    for filename in ["norm_wiki_sample.txt", "norm_hamlet.txt", "norm_romeo.txt"]:
        custom_print(f"[*] Generate for {filename}", style="h")
        with open(f"data/{filename}", "r") as file:
            data = file.read()
        generator = nlp_generator.Generator(data, mode="char")
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


@assignment_wrapper
def zad2():
    use_sample = True
    generated_length = 200
    for filename in ["norm_wiki_sample.txt", "norm_hamlet.txt", "norm_romeo.txt"]:
        custom_print(f"\n[*] Generating NLP with: {filename}", style="h")
        generator = nlp_generator.Generator(path=os.path.join("data", filename), mode="words", use_sample=use_sample)
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
    generator = nlp_generator.Generator(data=text, mode="char", use_sample=False)
    res = generator.get_entropy()
    return res


@assignment_wrapper
def zad3a():
    # choose whether you want to use sample to generate hash tables or not
    #       (reduce memory requirements but generates approximate result)
    use_sample = True
    files = ["norm_wiki_en.txt", "norm_wiki_la.txt"]
    for filename in files:
        custom_print(f"[*] Processing {filename}", style="h")
        custom_print(f"\tmode CHAR", style="b")
        generator = nlp_generator.Generator(path=os.path.join("data", filename), mode="char", use_sample=use_sample)
        custom_print(f"\tentropy of input_data is: {generator.get_entropy()}", style="g")
        res = generator.null_approximation(length=7000)
        custom_print(f"\tentropy of null approximation is: {get_entropy(text=res)}", style="g")
        generator = nlp_generator.Generator(path=os.path.join("data", filename), mode="words", use_sample=use_sample)
        for level in range(2, 6):
            custom_print(f"\tconditional entropy level {level} is: {generator.get_entropy(level=level)}",
                         style="g")
        custom_print(f"\tmode WORD", style="b")
        custom_print(f"\tentropy of input_data is: {generator.get_entropy()}", style="g")
        res = generator.null_approximation(length=7000)
        custom_print(f"\tentropy of null approximation is: {get_entropy(text=res)}", style="g")
        for level in range(2, 6):
            custom_print(f"\tconditional entropy level {level} is: {generator.get_entropy(level=level)}", style="g")


@assignment_wrapper
def zad3b():
    use_sample = True
    files = ["norm_wiki_en.txt", "norm_wiki_la.txt", "norm_wiki_nv.txt"] + [f"sample{x}.txt" for x in range(6)]
    for filename in files:
        custom_print(f"[*] Processing {filename}", style="h")
        custom_print(f"\tmode WORDS", style="b")
        generator = nlp_generator.Generator(path=os.path.join("data", filename), use_sample=use_sample, mode="words")
        for level in [0, 1, 2, 3, 4, 5]:
            custom_print(f"\tconditional: {level}\t{generator.get_entropy(level=level)}", style="g")
        custom_print(f"\tmode CHAR", style="b")
        generator = nlp_generator.Generator(path=os.path.join("data", filename), use_sample=use_sample, mode="char")
        for level in [0, 1, 2, 3, 4, 5]:
            custom_print(f"\tconditional: {level}\t{generator.get_entropy(level=level)}", style="g")

    # sample0.txt: False (entropy and conditional entropy of the level 1 has the same value)
    # sample1.txt: True
    # sample2.txt: False ? (too much repeating small words)
    # sample3.txt: True
    # sample4.txt: False (conditional entropy of words for 4, 5 is equal to 0,
    #                       conditional entropy of chars iis the same for the level 0-3)
    # sample5.txt: False (each word repeats 16 times??)


def test():
    generated_length = 300
    for filename in ["norm_wiki_sample.txt"]:
        custom_print(f"[*] Generate for {filename}", style="h")
        with open(f"data/{filename}", "r") as file:
            data = file.read()
        for mode in ["char"]:
            generator = nlp_generator.Generator(data, mode=mode, use_sample=True)
            for level in [1, 2, 5, 10]:
                custom_print(f"\t[+] markov chain level {level}", style="c")
                start_sub = ""
                res = generator.markov_model(level=5, length=generated_length, start_sub=start_sub)
                print("\t" + res)
                get_average_word_length(res)


if __name__ == '__main__':
    # zad1()
    # zad2()
    # zad3a()
    # zad3b()
    test()
