from histfile import *


def test_weighted_choice():
    tuples = (("Jacob", 0), ("Mark", 32), ("John", 22), ("Jack", 0))
    choice = weighted_choice(tuples)
    assert isinstance(choice, str)
    assert choice in ("Mark", "John")


def test_weighted_choice_zero():
    tuples = (("Jacob", 0), ("Mark", 0), ("John", 0), ("Jack", 0))
    choice = weighted_choice(tuples)
    assert isinstance(choice, str)
