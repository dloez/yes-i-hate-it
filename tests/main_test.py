"""Tests for main.py"""
import pytest
import tweepy

from yes_i_hate_it.main import load_env
from yes_i_hate_it.main import get_tweets
from yes_i_hate_it.main import is_new_tweet
from yes_i_hate_it.main import is_football

from yes_i_hate_it.main import TWITTER_API_KEY, TWITTER_API_SECRET
from yes_i_hate_it.main import TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET, TWITTER_BEARER_TOKEN

from yes_i_hate_it.exceptions import ValueExceeded, ValueInferior


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


def test_get_tweets():
    """Test main.get_tweets"""
    max_results = 10
    user_name = 'javieff16YT'
    tweets = get_tweets(user_name, max_results)

    assert isinstance(tweets, list)
    assert len(tweets) == max_results

    assert isinstance(tweets[0], tweepy.tweet.Tweet)
    assert 'text' in tweets[0]
    assert isinstance(tweets[0].text, str)
    assert 'id' in tweets[0]
    assert isinstance(tweets[0].id, int)

    with pytest.raises(ValueExceeded):
        get_tweets(user_name, 102)

    with pytest.raises(ValueInferior):
        get_tweets(user_name, 1)


def test_is_new_test():
    """Test main.is_new_test"""
    max_results = 10
    user_name = 'javieff16YT'
    tweets = get_tweets(user_name, max_results)

    assert is_new_tweet(tweets[0])
    assert not is_new_tweet(tweets[0])

def test_is_football():
    """Test main.is_football"""
    test_phrases = (
        "Me gusta el fúrbo, odio a los arbitros",
        "muahhh lloroo, el madrid es una mierda muahhhh",
        "frase aleatoria",
        "Probablemente el futbol sea de las peores enfermedades del mundo"
    )

    assert is_football(test_phrases[0])
    assert is_football(test_phrases[1])
    assert not is_football(test_phrases[2])
    assert is_football(test_phrases[3])
