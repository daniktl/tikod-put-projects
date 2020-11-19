from histfile import Generator
import os
import pytest


@pytest.mark.usefixtures("generator_char")
class TestChar:

    def test_generator_char_tokens(self, generator_char):
        assert len(generator_char.frequencies) == len(generator_char.tokens)

    def test_generator_char_null_approximation(self, generator_char):
        assert len(generator_char.null_approximation(length=100)) == 100

    def test_generator_char_basic_approximation(self, generator_char):
        assert len(generator_char.basic_approximation(length=100)) == 100


@pytest.mark.usefixtures("generator_char")
class TestWord:

    def test_generator_char_tokens(self, generator_words):
        assert len(generator_words.frequencies) == len(generator_words.tokens)

    def test_generator_char_basic_approximation(self, generator_words):
        length = len(generator_words.basic_approximation(length=100))
        assert 100 <= length <= length + len(sorted(generator_words.tokens, key=lambda x: len(x))[0])

    def test_generator_char_markov_1(self, generator_words):
        length = len(generator_words.markov_model(level=1, length=100))
        assert 100 <= length <= length + len(sorted(generator_words.tokens, key=lambda x: len(x))[0])
