"""Store project configuration"""
from pathlib import Path
from sqlalchemy.orm import declarative_base


# General
DATA_PATH = Path('data')

# Twitter tokens
TWITTER_API_KEY = 'TWITTER_API_KEY'
TWITTER_API_SECRET = 'TWITTER_API_SECRET'
TWITTER_ACCESS_TOKEN = 'TWITTER_ACCESS_TOKEN'
TWITTER_ACCESS_SECRET = 'TWITTER_ACCESS_SECRET'
TWITTER_BEARER_TOKEN = 'TWITTER_BEARER_TOKEN'

# store latest tweet
OLD_TWEET_PATH = DATA_PATH / 'old_tweet_id.pickle'

# deletection of football text
KEY_WORDS = (
    "portero", "futbol", "gol",
    "balon", "defensa", "central",
    "lateral", "campo", "delantero",
    "disparo", "tiro", "asistencia",
    "centro", "corner", "descuento",
    "penalti", "panenka","caño",
    "equipo", "arbitro", "arbitraje",
    "atleti", "atletico", "madrid",
    "final", "champions", "vamos",
    "liga", "copa", "fifa",
    "ea", "handicap", "atletimedia"
)
MIN_RATIO = 80

# tweet response data params
VEHICLE_REST_URL = 'https://random-data-api.com/api/vehicle/random_vehicle'
VEHICLE_HEADER_TEMPLATE = "{}, {} doors. "
TARGET_DATA = ('car_options', 'specs')
MAX_TWEET_CHARS = 280

# logging
# pylint: disable = line-too-long
LOGS_PATH = Path('logs')
LOG_FILE = LOGS_PATH / 'main.log'
DISCORD_LOG_FILE = LOGS_PATH / 'discord_bot.log'
GATHER_LOG_FILE = LOGS_PATH / 'gather.log'
BOW_LOG_FILE = LOGS_PATH / 'bow.log'
DATASET_LOG_FILE = LOGS_PATH / 'dataset.log'

SLEEP_TIME_MINUTES = 5

# IA
GATHER_TWEETS_FROM_USERS = [
    ('Javieff16YT', 2000),
    ('adrian_lomar', 2000),
    ('neme2k', 2000),
    ('elwandis', 2000),
    ('AtletiMediaLive', 2000),
    ('MartitasArranz', 2000)
]
BASE = declarative_base()
TWEETS_DB_PATH = DATA_PATH / 'tweets.sqlite'

# discord
DISCORD_WEBHOOK = 'https://discord.com/api/webhooks/976894637438554143/f1rOdQUB-a7keC3pTlUZ2ABDbz0dR9uY-ikI_q5UsJMV9wQDAUepNEIeu_TflGCzfxD8'
DISCORD_TOKEN = 'DISCORD_TOKEN'
CATEGORY_ID = 979508900342685696
# CATEGORY_ID = 979112708530139200

# Dataset
DATASET_FILE_PATH = DATA_PATH / 'dataset.npz'
TRAIN_PERCENTAGE = 80
