"""Download, store tweets and generate keyword from tweets"""
import logging
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Integer, Boolean
from sqlalchemy import create_engine

from yes_i_hate_it.config import TWEETS_DB_PATH
from yes_i_hate_it.config import GATHER_TWEETS_FROM_USERS
from yes_i_hate_it.config import BASE
from yes_i_hate_it.config import GATHER_LOG_FILE

from yes_i_hate_it.main import get_tweets


# pylint: disable = too-few-public-methods
class Tweet(BASE):
    """Tweet base clase to interact with database"""
    __tablename__ = 'tweets'

    tweet_id = Column(Integer, primary_key=True)
    text = Column(String)
    requested = Column(Boolean, default=False)
    is_football = Column(Boolean, default=False)
    processed = Column(Boolean, default=False)
    labeled = Column(Boolean, default=False)


class User(BASE):
    """User base clase to interact with database"""
    __tablename__ = 'users'

    twitter_user_name = Column(String, primary_key=True)
    tweets_amount = Column(Integer)


def check_user(user, amount, db_session):
    """Remove users from list which already are at the database"""
    db_user = db_session.query(User).get(user)
    if db_user:
        return True

    db_session.add(User(twitter_user_name=user, tweets_amount=amount))
    db_session.commit()
    return False


def gather_tweets(users, db_session):
    """Gather tweets from users"""
    for user, amount in users:
        if check_user(user, amount, db_session):
            logging.info("User %s was already requested, skipping...", user)
            continue

        logging.info("Getting %s tweets from %s", amount, user)
        pagination = ''
        for _ in range(0, amount, 100):
            twitter_data = get_tweets(
                user_name=user,
                max_results=100,
                pagination=pagination
            )

            tweets = twitter_data.data or []
            tweet_storage = []
            for tweet in tweets:
                tweet_storage.append(Tweet(
                    tweet_id=tweet.id,
                    text=tweet.text
                ))
            db_session.add_all(tweet_storage)
            db_session.commit()

            if 'next_token' not in twitter_data.meta:
                break
            pagination = twitter_data.meta['next_token']


def main():
    """Main function"""
    # pylint: disable = no-member
    GATHER_LOG_FILE.parents[0].mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(GATHER_LOG_FILE),
            logging.StreamHandler()
        ]
    )

    if not TWEETS_DB_PATH.parents[0].exists():
        TWEETS_DB_PATH.parents[0].mkdir(exist_ok=True)

    engine = create_engine(f'sqlite:///{TWEETS_DB_PATH}')
    BASE.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine)
    gather_tweets(GATHER_TWEETS_FROM_USERS, session_maker())


if __name__ == '__main__':
    main()
