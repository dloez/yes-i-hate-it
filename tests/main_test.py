"""Tests for main.py"""
import random
import string
import pytest
import tweepy

from yes_i_hate_it.main import load_env
from yes_i_hate_it.main import get_tweets
from yes_i_hate_it.main import save_tweet_id
from yes_i_hate_it.main import load_tweet_id
from yes_i_hate_it.main import is_football
from yes_i_hate_it.main import request_vehicle_data
from yes_i_hate_it.main import reply_tweet

from yes_i_hate_it.main import TWITTER_API_KEY, TWITTER_API_SECRET
from yes_i_hate_it.main import TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET, TWITTER_BEARER_TOKEN
from yes_i_hate_it.main import MAX_TWEET_CHARS

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
    tweets = get_tweets(user_name, max_results).data

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


def test_save_tweet_id():
    """Test main.write_tweet_id"""
    tweet_id = 123123
    assert save_tweet_id(tweet_id)


def test_load_tweet_id():
    """Test main.load_tweet_id"""
    assert not load_tweet_id()

    tweet_id = 123123
    save_tweet_id(tweet_id)
    assert tweet_id == load_tweet_id()


def test_is_football():
    """Test main.is_football"""
    test_phrases = (
        ("Me gusta el furbo, odio a los arbitros", True),
        ("muahhh lloroo, el madrid es una mierda muahhhh", True),
        ("frase aleatoria", False),
        ("Probablemente el futbol sea de las peores enfermedades del mundo", True),
        ("Internet es una fuente de informacion masiva", False),
        ("LA ATLETIMEDIANETA VUELVE MAMADÍSIMA!!!", True)
    )

    for phrase in test_phrases:
        text, result = phrase
        assert is_football(text) == result


def test_request_vehicle_data():
    """Test main.request_vehicle_data"""
    data = request_vehicle_data()
    assert isinstance(data, str)
    assert len(data) <= MAX_TWEET_CHARS


def test_reply_tweet():
    """Test main.reply_tweet"""
    tweet_id = 1527050399559057409
    text = "".join(random.choice(string.ascii_lowercase) for i in range(10))
    assert reply_tweet(tweet_id, text)
