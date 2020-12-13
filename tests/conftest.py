from nlp_generator import Generator
import os
import pytest


@pytest.fixture()
def generator_char():
    return Generator(path=os.path.join("data", "norm_hamlet.txt"), mode="char")


@pytest.fixture()
def generator_words():
    return Generator(path=os.path.join("data", "norm_hamlet.txt"), mode="words")
