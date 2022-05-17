"""Tests for main.py"""
from yes_i_hate_it.main import load_env
from yes_i_hate_it.main import TWITTER_API_KEY, TWITTER_API_SECRET
from yes_i_hate_it.main import TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET, TWITTER_BEARER_TOKEN


def test_load_env():
    """Test main.load_env function"""
    tokens = load_env()
    assert isinstance(tokens, dict)

    assert TWITTER_API_KEY in tokens
    assert isinstance(tokens[TWITTER_API_KEY], str)

    assert TWITTER_API_SECRET in tokens
    assert isinstance(tokens[TWITTER_API_SECRET], str)

    assert TWITTER_ACCESS_TOKEN in tokens
    assert isinstance(tokens[TWITTER_ACCESS_TOKEN], str)

    assert TWITTER_ACCESS_SECRET in tokens
    assert isinstance(tokens[TWITTER_ACCESS_SECRET], str)

    assert TWITTER_BEARER_TOKEN in tokens
    assert isinstance(tokens[TWITTER_BEARER_TOKEN], str)
