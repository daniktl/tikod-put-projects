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


if __name__ == '__main__':
    examples()
