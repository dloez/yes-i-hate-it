"""Store project configuration"""
from pathlib import Path


TWITTER_API_KEY = 'TWITTER_API_KEY'
TWITTER_API_SECRET = 'TWITTER_API_SECRET'
TWITTER_ACCESS_TOKEN = 'TWITTER_ACCESS_TOKEN'
TWITTER_ACCESS_SECRET = 'TWITTER_ACCESS_SECRET'
TWITTER_BEARER_TOKEN = 'TWITTER_BEARER_TOKEN'

OLD_TWEET_PATH = Path('./data/old_tweet_id.pickle')

KEY_WORDS = (
    "futbol", "arbitro", "porteria",
    "var", "atletico", "madrid"
)
MIN_RATIO = 80
