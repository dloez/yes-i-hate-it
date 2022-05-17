"""
Main file which will handle:
    - listen for new tweets
    - detect if last tweet is football related
    - if it is football related, fetch data from a random API
    - respond with the fetch data
"""
from typing import List
import os
import tweepy

from yes_i_hate_it.exceptions import ValueExceeded, ValueInferior


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


def get_tweets(user_name: str, max_results: int) -> List[tweepy.Tweet]:
    """Get 'x' amount of latests tweets from 'y' user"""
    # test max_result value
    if max_results < 5:
        raise ValueInferior
    if max_results > 100:
        raise ValueExceeded

    # init tweepy client object to call Twitter REST API
    tokens = load_env()
    client = tweepy.Client(tokens[TWITTER_BEARER_TOKEN])

    # get user from given user_name
    user = client.get_user(username=user_name)

    # get latest amount of tweets from user_name
    tweets = client.get_users_tweets(id=user.data['id'], max_results=max_results)
    return list(tweets.data)


def process_tweets(_tweets):
    """
    Get twitter raw API from tweets and returns a list fo tweets with:
        - tweet body
        - tweet id
        - tweet user id
    """


def main():
    """main function"""
    # print(get_tweets('Javieff16YT', 30))


TWITTER_API_KEY = 'TWITTER_API_KEY'
TWITTER_API_SECRET = 'TWITTER_API_SECRET'
TWITTER_ACCESS_TOKEN = 'TWITTER_ACCESS_TOKEN'
TWITTER_ACCESS_SECRET = 'TWITTER_ACCESS_SECRET'
TWITTER_BEARER_TOKEN = 'TWITTER_BEARER_TOKEN'

if __name__ == '__main__':
    main()
