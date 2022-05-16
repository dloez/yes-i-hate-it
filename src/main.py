"""
Main file which will handle:
    - listen for new tweets
    - detect if last tweet is football related
    - if it is football related, fetch data from a random API
    - respond with the fetch data
"""
import os


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

def main():
    """main function"""


TWITTER_API_KEY = 'TWITTER_API_KEY'
TWITTER_API_SECRET = 'TWITTER_API_SECRET'
TWITTER_ACCESS_TOKEN = 'TWITTER_ACCESS_TOKEN'
TWITTER_ACCESS_SECRET = 'TWITTER_ACCESS_SECRET'
TWITTER_BEARER_TOKEN = 'TWITTER_BEARER_TOKEN'

if __name__ == '__main__':
    main()
