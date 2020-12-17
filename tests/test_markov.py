from nlp_generator import Generator
import pytest


@pytest.mark.usefixtures("generator_char")
class TestChar:

    def test_generator_char_null_approximation(self, generator_char):
        assert len(generator_char.null_approximation(length=100)) == 100

    def test_generator_char_basic_approximation(self, generator_char):
        assert len(generator_char.basic_approximation(length=100)) == 100

    @pytest.mark.parametrize("level", [1, 2, 5, 10])
    def test_generator_char_markov(self, generator_char: Generator, level):
        assert len(generator_char.markov_model(level=level, length=100)) == 100


@pytest.mark.usefixtures("generator_char")
class TestWord:

    def test_generator_word_tokens(self, generator_words):
        generator_words.basic_approximation()
        assert len(generator_words.hashtable) == len(generator_words.tokens)

    def test_generator_word_basic_approximation(self, generator_words):
        length = len(generator_words.basic_approximation(length=100))
        assert 100 <= length <= length + len(sorted(generator_words.tokens, key=lambda x: len(x))[0])

    @pytest.mark.parametrize("level", [1, 2, 5])
    def test_generator_word_markov(self, generator_words, level):
        length = len(generator_words.markov_model(level=level, length=100))
        assert 100 <= length <= length + len(sorted(generator_words.tokens, key=lambda x: len(x))[0])
