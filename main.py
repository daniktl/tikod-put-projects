import nlp_generator, os


def examples():
    # Example 1
    filename = "disc.dat"
    array_ = nlp_generator.array_from_file(f"data/{filename}", as_float=True)
    maxs = nlp_generator.find_max_by_cols(array_)
    mins = nlp_generator.find_min_by_cols(array_)
    print(maxs, mins)
    for col in range(len(array_)):
        nlp_generator.make_hist(array_[col], as_prob=True)
    # Example 2
    hist_data = nlp_generator.CustomHistData()
    hist_data.upload_data(path=f'data/{filename}', as_float=True)
    maxs = hist_data.get_max()
    mins = hist_data.get_min()
    print(maxs, mins)
    hist_data.draw_plots()


def get_average_word_length(text):
    words = text.split()
    if not words:
        return 0
    full_length = sum([len(word) for word in words])
    return full_length/len(words)


def zad1():
    generated_length = 300
    sample_delta = 30000
    for filename in ["norm_wiki_sample.txt", "norm_hamlet.txt", "norm_romeo.txt"]:
        print(f"[*] Generate for {filename}")
        with open(f"data/{filename}", "r") as file:
            data = file.read()
        generator = nlp_generator.Generator(data, sample_delta=sample_delta)
        print(generator.frequencies)
        print("\t[+] null approximation")
        res = generator.null_approximation()
        print("\t" + res)
        print(f"\t average word length: {get_average_word_length(res)}".upper())
        print("\t[+] basic approximation")
        res = generator.basic_approximation(length=generated_length)
        print("\t" + res)
        print(f"\t average word length: {get_average_word_length(res)}".upper())
        for level in [1, 3, 5]:
            print(f"\t[+] markov chain level {level}")
            if level == 5:
                start_sub = "probability"
            else:
                start_sub = ""
            res = generator.markov_model(level=5, length=generated_length, start_sub=start_sub)
            print("\t" + res)
            print(f"\t average word length: {get_average_word_length(res)}".upper())


def zad2():
    generated_length = 200
    for filename in ["norm_wiki_sample.txt", "norm_hamlet.txt", "norm_romeo.txt"]:
        print(f"[*] Generate for {filename}")
        generator = nlp_generator.Generator(path=os.path.join("data", filename), mode="words", use_sample=False)
        res = generator.basic_approximation(length=generated_length)
        print(res)
        print(f"\t average word length: {get_average_word_length(res)}".upper())
        res = generator.markov_model(length=generated_length, level=1)
        print(res)
        print(f"\t average word length: {get_average_word_length(res)}".upper())
        res = generator.markov_model(length=generated_length, level=2)
        print(res)
        print(f"\t average word length: {get_average_word_length(res)}".upper())
        res = generator.markov_model(length=generated_length, level=2, start_sub="probability")
        print(res)
        print(f"\t average word length: {get_average_word_length(res)}".upper())


def get_entropy(text: str) -> float:
    generator = nlp_generator.Generator(data=text, mode="char", use_sample=False)
    res = generator.get_entropy()
    return res


def zad3():
    files = ["norm_wiki_en.txt", "norm_wiki_la.txt"]
    for filename in files:
        print(f"[*] Processing {filename}")
        generator = nlp_generator.Generator(path=os.path.join("data", filename), mode="char", use_sample=True)
        print(f"\t [CHAR] entropy of input_data is: {generator.get_entropy()}")
        res = generator.null_approximation(length=7000)
        print(f"\t [CHAR] entropy of null approximation is: {get_entropy(text=res)}")
        generator = nlp_generator.Generator(path=os.path.join("data", filename), mode="words", use_sample=False)
        print(f"\t [WORD] entropy of input_data is: {generator.get_entropy()}")
        res = generator.null_approximation(length=7000)
        print(f"\t [WORD] entropy of null approximation is: {get_entropy(text=res)}")
        for level in range(2, 6):
            # generator.generate_hashtable(level=level, description=True)
            print(f"\t [WORD] conditional entropy level {level} is: {generator.get_entropy()}")

    return


def zad4():
    generated_length = 300
    files = ["norm_wiki_en.txt"]
    for filename in files:
        print(f"[*] Processing {filename}")
        print(f"\tmode WORDS")
        generator = nlp_generator.Generator(path=os.path.join("data", filename), use_sample=False, mode="words")
        for level in [0, 1, 2, 3, 4, 5]:
            print(f"\tconditional: {level}\t{generator.get_entropy(level=level)}")
        print(f"\tmode CHAR")
        generator = nlp_generator.Generator(path=os.path.join("data", filename), use_sample=False, mode="char")
        for level in [0, 1, 2, 3, 4, 5]:
            print(f"\tconditional: {level}\t{generator.get_entropy(level=level)}")


if __name__ == '__main__':
    zad4()
