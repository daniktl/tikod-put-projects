import histfile


def examples():
    # Example 1
    filename = "disc.dat"
    array_ = histfile.array_from_file(f"data/{filename}", as_float=True)
    maxs = histfile.find_max_by_cols(array_)
    mins = histfile.find_min_by_cols(array_)
    print(maxs, mins)
    for col in range(len(array_)):
        histfile.make_hist(array_[col], as_prob=True)
    # Example 2
    hist_data = histfile.CustomHistData()
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
    generated_length = 200
    sample_delta = 15000
    for filename in ["norm_wiki_sample.txt", "norm_hamlet.txt", "norm_romeo.txt"]:
        print(f"[*] Generate for {filename}")
        with open(f"data/{filename}", "r") as file:
            data = file.read()
        generator = histfile.Generator(data, sample_delta=sample_delta)
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


if __name__ == '__main__':
    zad1()
