"""
Main file which will handle:
    - listen for new tweets
    - detect if last tweet is football related
    - if it is football related, fetch data from a random API
    - respond with the fetch data
"""
from typing import List
import pickle
import os
import re
import time
import random
import logging
import tweepy
import requests
from thefuzz import fuzz

from yes_i_hate_it.exceptions import ValueExceeded, ValueInferior

from yes_i_hate_it.config import TWITTER_API_KEY, TWITTER_API_SECRET
from yes_i_hate_it.config import TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
from yes_i_hate_it.config import TWITTER_BEARER_TOKEN
from yes_i_hate_it.config import OLD_TWEET_PATH
from yes_i_hate_it.config import KEY_WORDS, MIN_RATIO
from yes_i_hate_it.config import VEHICLE_REST_URL, VEHICLE_HEADER_TEMPLATE
from yes_i_hate_it.config import TARGET_DATA, MAX_TWEET_CHARS
from yes_i_hate_it.config import LOG_FILE, DISCORD_WEBHOOK


def load_env():
    """Load twitter tokens from environment variabless"""
    tokens = {
        TWITTER_API_KEY: os.getenv(TWITTER_API_KEY),
        TWITTER_API_SECRET: os.getenv(TWITTER_API_SECRET),
        TWITTER_ACCESS_TOKEN: os.getenv(TWITTER_ACCESS_TOKEN),
        TWITTER_ACCESS_SECRET: os.getenv(TWITTER_ACCESS_SECRET),
        TWITTER_BEARER_TOKEN: os.getenv(TWITTER_BEARER_TOKEN)
    }
    return tokens


def _init_tweepy():
    tokens = load_env()
    return tweepy.Client(
        bearer_token=tokens[TWITTER_BEARER_TOKEN],
        consumer_key=tokens[TWITTER_API_KEY],
        consumer_secret=tokens[TWITTER_API_SECRET],
        access_token=tokens[TWITTER_ACCESS_TOKEN],
        access_token_secret=tokens[TWITTER_ACCESS_SECRET]
    )


def get_tweets(user_name: str, max_results: int) -> List[tweepy.Tweet]:
    """Get 'x' amount of latests tweets from 'y' user"""
    # test max_result value
    if max_results < 5:
        raise ValueInferior
    if max_results > 100:
        raise ValueExceeded

    # init tweepy client object to call Twitter REST API
    client = _init_tweepy()

    # get user from given user_name
    user = client.get_user(username=user_name)

    # get latest amount of tweets from user_name
    tweets = client.get_users_tweets(id=user.data['id'], max_results=max_results)
    return list(tweets.data)


def is_new_tweet(new_tweet: tweepy.Tweet) -> bool:
    """
    Detect if given tweet is the latest user tweet by comparing
    given new_tweet with stored one.
    If there is not a stored tweet, store new one and return True
    """
    # verify if there is an old tweet stored
    if not OLD_TWEET_PATH.exists():
        # pylint: disable=no-member
        # above rule is required to avoid pylint errors on python3.10
        OLD_TWEET_PATH.parents[0].mkdir(exist_ok=True)

        with open(OLD_TWEET_PATH, 'wb') as handle:
            pickle.dump(new_tweet.id, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    # load old tweet
    with open(OLD_TWEET_PATH, 'rb') as handle:
        old_tweet_id = pickle.load(handle)

    if new_tweet.id != old_tweet_id:
        with open(OLD_TWEET_PATH, 'wb') as handle:
            pickle.dump(new_tweet.id, handle)
        return True
    return False


def is_football(text: str) -> bool:
    """
    Evaluate if given text is fooball or not looking at
    similarities between given text and config.KEY_WORDS
    """
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )

    for word in text.split(' '):
        # normalize words
        word = re.sub('[^0-9a-zA-Z]+', '', word).lower()
        for accent, no_accent in replacements:
            word = word.replace(accent, no_accent)

        for key_word in KEY_WORDS:
            ratio = fuzz.ratio(word, key_word)
            if ratio > MIN_RATIO:
                return True
    return False


def request_vehicle_data() -> str:
    """Get random vehicle data and format it to fit on a tweet"""
    # pylint: disable = broad-except
    try:
        data = requests.get(VEHICLE_REST_URL).json()
    except Exception:
        print("Vehicle REST API is unavailable!!")

    formated_data = VEHICLE_HEADER_TEMPLATE.format(data['make_and_model'], data['doors'])
    available_chars = MAX_TWEET_CHARS - len(formated_data)

    target = random.choice(TARGET_DATA)
    for spec in data[target]:
        spec += ', '
        remain_chars = available_chars - len(spec)
        if remain_chars >= 0:
            formated_data += spec
            available_chars = remain_chars
    return formated_data[:-2] + '.'


def reply_tweet(tweet_id: int, text: str) -> bool:
    """Reply to tweet with given text"""
    client = _init_tweepy()
    # pylint: disable = broad-except
    try:
        client.create_tweet(in_reply_to_tweet_id=tweet_id, text=text)
    except Exception:
        return False
    return True


def post_discord(message: str):
    """Post message to Discord Webhook"""
    data = {'content': message}
    requests.post(DISCORD_WEBHOOK, json=data)


def log(level: str, text: str):
    """Log to stdout, file and discord"""
    level_map = {
        'debug': logging.debug,
        'info':  logging.info,
        'error': logging.error
    }

    if level not in level_map:
        return

    level_map[level](text)
    post_discord(text)


def main():
    """main function"""
    # configure logging
    LOG_FILE.parents[0].mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

    user_name = 'Javieff16YT'
    log(INFO, f"Started, targeting {user_name}")

    while True:
        logging.info("Getting new tweets...")

        last_tweet = get_tweets(user_name=user_name, max_results=5)[0]
        if is_new_tweet(last_tweet):
            log(INFO, f"Found new tweet with ID {last_tweet.id}, evaluating...")
            if is_football(last_tweet.text):
                log(INFO, f"Tweet with ID {last_tweet.id} is football related, replying...")
                vehicle_data = request_vehicle_data()
                reply_tweet(tweet_id=last_tweet.id, text=vehicle_data)
                log(INFO, f"Replied to tweet URL: https://twitter.com/{user_name}/status/{last_tweet.id}")
            else:
                log(INFO, f"Tweet with ID {last_tweet.id} is not football related")
        time.sleep(5*60)


BUG = 'debug'
INFO = 'info'
ERROR = 'error'

if __name__ == '__main__':
    main()
