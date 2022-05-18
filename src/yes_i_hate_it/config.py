"""Store project configuration"""
from pathlib import Path


TWITTER_API_KEY = 'TWITTER_API_KEY'
TWITTER_API_SECRET = 'TWITTER_API_SECRET'
TWITTER_ACCESS_TOKEN = 'TWITTER_ACCESS_TOKEN'
TWITTER_ACCESS_SECRET = 'TWITTER_ACCESS_SECRET'
TWITTER_BEARER_TOKEN = 'TWITTER_BEARER_TOKEN'

OLD_TWEET_PATH = Path('./data/old_tweet_id.pickle')

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

VEHICLE_REST_URL = 'https://random-data-api.com/api/vehicle/random_vehicle'
VEHICLE_HEADER_TEMPLATE = "{}, {} doors. "
TARGET_DATA = ('car_options', 'specs')
MAX_TWEET_CHARS = 280
