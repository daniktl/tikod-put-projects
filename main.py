import histfile


def example():
    filename = "disc.dat"
    array_ = histfile.array_from_file(f"data/{filename}", as_float=True)
    maxs = histfile.find_max_by_cols(array_)
    mins = histfile.find_min_by_cols(array_)
    print(maxs, mins)
    for col in range(len(array_)):
        histfile.make_hist(array_[col], as_prob=True)


if __name__ == '__main__':
    example()
